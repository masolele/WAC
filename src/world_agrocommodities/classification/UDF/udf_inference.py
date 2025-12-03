import functools
import hashlib
import logging
import os
import sys
import tempfile
import threading
from typing import Any, Dict, Tuple
import logging
import numpy as np
import requests
import xarray as xr
from openeo.metadata import CollectionMetadata

sys.path.append("onnx_deps")
import onnxruntime as ort

# Constants for sanitization
_INF_REPLACEMENT = 1e6
_NEG_INF_REPLACEMENT = -1e6

NUM_THREADS = 2

_MODULE_CACHE_KEY = f"__model_cache_{__name__}"

logger = logging.getLogger(__name__)


# Global lock dictionary for thread-safe model downloading
_model_locks: Dict[str, threading.RLock] = {}
_model_locks_lock = threading.Lock()  # Lock for managing the lock dictionary


# =============================================================================
# Functions to download and load ONNX models with caching and thread-safety
# =============================================================================


def optimize_onnx_cpu_performance(num_threads):
    """CPU-specific ONNX optimizations."""
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = (
        num_threads  # TODO test setting to 1 due to sequential nature
    )

    # CPU-specific optimizations
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    return session_options, providers


# Cache for loaded models
def get_model_cache() -> Dict[str, Any]:
    """Get or create module-specific cache."""
    if not hasattr(sys, _MODULE_CACHE_KEY):
        setattr(sys, _MODULE_CACHE_KEY, {})
    return getattr(sys, _MODULE_CACHE_KEY)


def get_model_lock(model_id: str) -> threading.RLock:
    """Get or create a lock for a specific model ID (thread-safe)."""
    with _model_locks_lock:
        if model_id not in _model_locks:
            _model_locks[model_id] = threading.RLock()
        return _model_locks[model_id]


# Cache for downloaded models
def get_model_cache_path(model_id: str, cache_dir: str = "/tmp/onnx_models") -> str:
    """Get the cache path for a model, creating directory if needed."""
    os.makedirs(cache_dir, exist_ok=True)

    # Create a safe filename from model_id
    model_hash = hashlib.md5(model_id.encode()).hexdigest()
    return os.path.join(cache_dir, f"{model_hash}.onnx")


def download_model_with_lock(
    model_id: str,
    model_url: str,
    cache_dir: str = "/tmp/onnx_models",
    max_file_size_mb: int = 250,
) -> str:
    """
    Download model with thread locking to prevent concurrent downloads.
    """
    cache_path = get_model_cache_path(model_id, cache_dir)

    # Get the lock for this specific model
    lock = get_model_lock(model_id)

    with lock:
        # Check if model already exists in cache
        if os.path.exists(cache_path):
            logger.info(f"Using cached model: {cache_path}")
            return cache_path

        # Download the model
        logger.info(f"Downloading model {model_id} from {model_url}")

        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=".onnx", dir=cache_dir)

            try:
                with os.fdopen(temp_fd, "wb") as temp_file:
                    response = requests.get(model_url, stream=True, timeout=300)
                    response.raise_for_status()

                    # Download with size checking
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            downloaded_size += len(chunk)
                            if downloaded_size > max_file_size_mb * 1024 * 1024:
                                raise ValueError(
                                    f"Downloaded file exceeds size limit of {max_file_size_mb}MB"
                                )

                # Atomic move from temp file to final location
                os.rename(temp_path, cache_path)
                logger.info(f"Successfully downloaded and cached model: {cache_path}")

            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise ValueError(f"Error downloading model {model_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise

    return cache_path


def get_model_from_stac(
    model_id: str,
    stac_api_url: str = "https://stac.openeo.vito.be",
    cache_dir: str = "/tmp/onnx_models",
) -> Tuple[str, dict]:
    """Fetch model file and metadata from STAC API with caching."""
    try:
        collection_id = "world-agri-commodities-models"
        url = f"{stac_api_url}/collections/{collection_id}/items/{model_id}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        item = response.json()
        properties = item.get("properties", {})
        assets = item.get("assets", {})

        # Get model URL
        model_asset = assets.get("model")
        if not model_asset:
            raise ValueError(f"No model asset found for {model_id}")

        model_url = model_asset["href"]

        # Download model with caching and locking
        model_path = download_model_with_lock(model_id, model_url, cache_dir)

        metadata = {
            "input_bands": properties.get("input_channels", []),
            "output_classes": properties.get("output_classes", []),
            "output_shape": properties.get("output_shape", 0),
            "framework": properties.get("framework", "ONNX"),
            "region": properties.get("region", "Unknown"),
            "model_url": model_url,
            "cached_path": model_path,
        }

        logger.info(
            f"Retrieved model {model_id} with {len(metadata['output_classes'])} output classes"
        )
        return model_path, metadata

    except Exception as e:
        logger.error(f"Failed to fetch model from STAC: {e}")
        raise


def load_ort_session(model_id: str) -> Tuple[ort.InferenceSession, dict]:
    """Loads an ONNX model from STAC and returns session with metadata"""

    cache = get_model_cache()

    entry = cache.get(model_id)
    if entry is not None:
        logger.info(f"Using cached ONNX session for model {model_id}")
        return entry["session"], entry["metadata"]

    lock = get_model_lock(model_id)
    with lock:
        # Double-check cache inside lock
        entry = cache.get(model_id)
        if entry is not None:
            logger.info(f"Using cached ONNX session for model {model_id}")
            return entry["session"], entry["metadata"]

        # If not cached, load model from STAC
        model_path, metadata = get_model_from_stac(model_id)
        session_options, providers = optimize_onnx_cpu_performance(NUM_THREADS)

        try:
            session = ort.InferenceSession(
                model_path, session_options, providers=providers
            )
            # Cache the loaded session
            cache[model_id] = {"session": session, "metadata": metadata}
            logger.info(
                f"Loaded and cached ONNX model for {model_id} on path {model_path}"
            )
            return session, metadata
        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise

        # finally:
        #     # Clean up temporary file
        #     try:
        #         os.unlink(model_path)
        #     except OSError as e:
        #         raise RuntimeError(
        #             f"Failed to delete temporary model file: {model_path}"
        #         ) from e


class ONNXClassifier:
    """Handles ONNX model inference for classification."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.model_id = self.parameters.get("model_id")
        if not self.model_id:
            raise ValueError("model_id must be provided in context")

    def _preprocess_image(
        self,
        cube: xr.DataArray,
    ) -> Tuple[np.ndarray, Dict[str, xr.DataArray], np.ndarray]:
        """
        Prepare the input cube for inference:
        - Transpose to (y, x, bands)
        - Sanitize NaN/Inf
        - Return batch tensor, coords, and invalid-value mask
        """
        # Reorder dims
        if "t" in cube.dims:
            cube = cube.squeeze("t", drop=True)

        reordered = cube.transpose("y", "x", "bands")
        values = reordered.values.astype(np.float32)

        # Mask invalid entries
        mask_invalid = ~np.isfinite(values)

        # Replace NaN with 0, inf with large sentinel
        sanitized = np.where(
            np.isnan(values), 0.0, values
        )  # TODO validate if this is okay
        sanitized = np.where(np.isposinf(sanitized), _INF_REPLACEMENT, sanitized)
        sanitized = np.where(np.isneginf(sanitized), _NEG_INF_REPLACEMENT, sanitized)

        # Add batch dimension
        input_tensor = sanitized[None, ...]
        logger.info(f"Preprocessed tensor shape={input_tensor.shape}")
        return input_tensor, dict(reordered.coords), mask_invalid

    def _run_inference(
        self, session: ort.InferenceSession, input_name: str, input_tensor: np.ndarray
    ) -> np.ndarray:
        """Run ONNX session and remove batch dimension from output."""
        outputs = session.run(None, {input_name: input_tensor})
        pred = np.squeeze(outputs[0], axis=0)
        logger.info(f"Inference output shape={pred.shape}")
        return pred

    def _postprocess_output(
        self,
        pred: np.ndarray,  # Shape: [y, x, bands]
        coords: Dict[str, xr.DataArray],
        mask_invalid: np.ndarray,  # Shape: [y, x, bands]
    ) -> xr.DataArray:
        """
        Appends winning class index as new band to predictions:
        - Keeps original prediction values
        - Adds new band (-1 for invalid, 0..n-1 for winning class)
        """

        # Apply sigmoid
        # sigmoid_probs = expit(pred)  # shape [y, x, bands]

        # Optionally pick highest prob if needed
        # class_index = np.argmax(pred, axis=-1, keepdims=True)

        # Identify invalid pixels (any invalid in input bands)
        class_index = np.argmax(pred, axis=-1, keepdims=True)  # shape [y, x, 1]

        invalid_mask = np.any(mask_invalid, axis=-1, keepdims=True)
        class_index = np.where(invalid_mask, -1, class_index).astype(np.float32)

        # Update band coordinates
        new_band_coords = np.arange(pred.shape[-1] + 1)

        combined = np.concatenate([pred, class_index], axis=-1)

        return xr.DataArray(
            combined,
            dims=("y", "x", "bands"),
            coords={"y": coords["y"], "x": coords["x"], "bands": new_band_coords},
            attrs={"description": "Original preds, sigmoid probs, class index"},
        )

    def _apply_model_single_timestep(self, cube: xr.DataArray) -> xr.DataArray:
        """
        Full inference pipeline on a single timestep:
        - Read ONNX model input shape
        - If model expects 15 bands → drop NDRE/EVI by name
        - Preprocess → infer → postprocess
        """
        session, metadata = load_ort_session(self.model_id)

        input_bands = metadata["input_bands"]
        output_classes = metadata["output_classes"]

        logger.info(f"Running inference for model {self.model_id}")
        logger.info(f"Input bands: {input_bands}")
        logger.info(f"Output bands: {output_classes}")

        # Validate input bands match expectations
        cube_bands = list(cube.coords["bands"].values)
        if cube_bands != input_bands:
            logger.warning(f"Band mismatch. Cube: {cube_bands}, Model: {input_bands}")
            cube = cube.sel(bands=input_bands)

        input_tensor, coords, mask_invalid = self._preprocess_image(cube)
        input_name = session.get_inputs()[0].name
        raw_pred = self._run_inference(session, input_name, input_tensor)

        return self._postprocess_output(raw_pred, coords, mask_invalid)

    def apply_model(self, cube: xr.DataArray) -> xr.DataArray:
        """
        Apply ONNX model per timestep in the datacube.
        """
        logger.info(f"Applying model from STAC: {self.model_id}")

        if "t" in cube.dims:
            logger.info("Applying model per timestep via groupby-map.")

            # Use isel to handle time dimension properly
            def process_timestep(da):
                return self._apply_model_single_timestep(da)

            return cube.groupby("t").map(process_timestep)
        else:
            logger.info("Single timestep: applying model once.")
            return self._apply_model_single_timestep(cube)


# =============================================================================
# ACTUAL UDF
# =============================================================================


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    model_id = context.get("model_id")
    _, metadata_dict = load_ort_session(model_id)

    output_classes = metadata_dict["output_classes"] + ["ARGMAX"]
    logger.info(f"Applying metadata with output classes: {output_classes}")
    return metadata.rename_labels(dimension="bands", target=output_classes)


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Apply ONNX model per timestep in the datacube.
    """

    classifier = ONNXClassifier(context)
    # Ensure correct dimension order
    cube = cube.transpose("y", "x", "bands", "t")

    output = classifier.apply_model(cube)
    return output
