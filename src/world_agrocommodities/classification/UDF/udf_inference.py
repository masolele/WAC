import functools
import hashlib
import logging
import os
import sys
import tempfile
import threading
from typing import Any, Dict, List, Tuple
import logging
import numpy as np
import requests
import xarray as xr
from pyproj import Transformer
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData

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


@functools.lru_cache(maxsize=1)
def get_model_metadata_from_stac(
    model_id: str, stac_api_url: str = "https://stac.openeo.vito.be"
) -> dict:
    """Fetch model metadata from STAC API"""
    try:
        # Get collection and item
        collection_id = "world-agri-commodities-models"
        url = f"{stac_api_url}/collections/{collection_id}/items/{model_id}"

        response = requests.get(url)
        response.raise_for_status()

        item = response.json()
        properties = item.get("properties", {})

        logger.info(f"Retrieved model metadata for {model_id}")
        return {
            "input_bands": properties.get("input_channels", []),
            "input_shape": properties.get("input_shape", 0),
        }

    except Exception as e:
        logger.error(f"Failed to fetch model metadata: {e}")
        raise


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


# =============================================================================
# Lon-Lat calculation
# =============================================================================
class LonLatCalculator:
    def _calc_lonlat(self, cube: xr.DataArray, crs: str | int) -> xr.DataArray:
        """
        Calculates lat/lon values for the given spatial extent.

        Args:
            cube: Input data cube with spatial dimensions.
        Returns:
            DataArray: Cube with latitude and longitude as bands.
        """
        cube = cube.transpose("bands", "t", "y", "x")

        logger.info(f"EPSG code determined for feature extraction: {crs}")

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        X, Y = np.meshgrid(cube.x.values, cube.y.values)
        lon_grid, lat_grid = transformer.transform(X, Y)

        logger.info(f"Transformed longitudes range: {lon_grid.min()}, {lon_grid.max()}")
        logger.info(f"Transformed latitudes range: {lat_grid.min()}, {lat_grid.max()}")

        t_len = cube.sizes["t"]
        lon_t = np.broadcast_to(lon_grid, (t_len,) + lon_grid.shape)  # (t, y, x)
        lat_t = np.broadcast_to(lat_grid, (t_len,) + lat_grid.shape)  # (t, y, x)

        combined = xr.DataArray(
            data=np.stack([lon_t, lat_t], axis=0),  # shape: (2, t, y, x)
            dims=("bands", "t", "y", "x"),
            coords={
                "bands": ["lon", "lat"],
                "t": cube.coords["t"],
                "x": cube.coords["x"],
                "y": cube.coords["y"],
            },
        )

        return combined

    def append_lonlat(self, cube: xr.DataArray, crs: str | int) -> xr.DataArray:
        """
        Appends lon/lat bands to the input cube.

        Args:
            cube: Input data cube with spatial dimensions.
        Returns:
            DataArray: Cube with lon/lat bands appended.
        """
        lonlat = self._calc_lonlat(cube, crs)
        combined = xr.concat([cube, lonlat], dim="bands")
        return combined


# =============================================================================
# Pre-classification normalization
# =============================================================================
class Normalizer:
    """Handles normalization of input data based on model metadata."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.model_id = self.parameters.get("model_id")
        if not self.model_id:
            raise ValueError("model_id must be provided in context")
        self.NORMALIZE_FUNCS = {
            "optical": self._normalize_optical,
            "linear": self._normalize_linear,
        }

    def _get_normalization_specs(self, input_bands: List[str]) -> dict:
        """Dynamically generate normalization specs based on input bands"""
        specs = {
            "optical": {
                "B02": (1.7417268007636313, 2.023298706048351),
                "B03": (1.7261204997060209, 2.038905204308012),
                "B04": (1.6798346251414997, 2.179592821212937),
                "B05": (2.3828939530384052, 2.7578332604178284),
                "B06": (1.7417268007636313, 2.023298706048351),
                "B07": (1.7417268007636313, 2.023298706048351),
                "B08": (1.7417268007636313, 2.023298706048351),
                "B11": (1.7417268007636313, 2.023298706048351),
                "B12": (1.7417268007636313, 2.023298706048351),
            },
            "linear": {
                "NDVI": (0, 1),
                "NDRE": (-1, 1),
                "EVI": (-1, 1),
                "VV": (-25, 0),
                "VH": (-30, -5),
                "DEM": (-400, 8000),
                "lon": (-180, 180),
                "lat": (-60, 60),
            },
        }

        # Filter specs to only include bands that are actually in the input
        filtered_specs = {"optical": {}, "linear": {}}

        for band in input_bands:
            if band in specs["optical"]:
                filtered_specs["optical"][band] = specs["optical"][band]
            elif band in specs["linear"]:
                filtered_specs["linear"][band] = specs["linear"][band]

        return filtered_specs

    def _normalize_optical(
        self, arr: np.ndarray, min_spec: float, max_spec: float
    ) -> np.ndarray:
        """Log-based normalization for optical bands."""
        arr = np.log(arr * 0.005 + 1)
        arr = (arr - min_spec) / (max_spec)
        arr = np.exp(arr * 5 - 1)
        return arr / (arr + 1)

    def _normalize_linear(
        self, arr: np.ndarray, min_spec: float, max_spec: float
    ) -> np.ndarray:
        """Linear min–max normalization for continuous variables."""
        arr = np.clip(arr, min_spec, max_spec)
        return (arr - min_spec) / (max_spec - min_spec)

    def validate_bands(self, cube: xr.DataArray, expected_bands: list):
        band_names = list(cube.coords["bands"].values)
        logger.info(f"Input bands: {band_names}")
        logger.info(f"Expected bands from model: {expected_bands}")

        # Check for missing required bands
        missing_bands = [b for b in expected_bands if b not in band_names]
        if missing_bands:
            raise ValueError(
                f"Missing required bands: {missing_bands}. Got: {band_names}"
            )

        # Check order
        if band_names != expected_bands:
            logger.warning(
                f"Band order mismatch: reordering from {band_names} to {expected_bands}"
            )
            cube = cube.sel(bands=expected_bands)

        return cube

    def apply_normalization(self, cube: xr.DataArray) -> xr.DataArray:
        """
        Apply normalization to the input cube based on model specs.
        """

        t_coords = cube.coords.get("t", None)

        model_metadata = get_model_metadata_from_stac(self.model_id)
        expected_bands = model_metadata["input_bands"]

        logger.info(
            f"Using model: {self.model_id} with expected bands: {expected_bands}"
        )
        logger.info(f"Model expects {model_metadata['input_shape']} input bands")

        # Validate bands
        cube = self.validate_bands(cube, expected_bands)

        # Get normalization specs
        normalization_specs = self._get_normalization_specs(expected_bands)

        band_names = list(cube.coords["bands"].values)
        logger.info(f"Normalizing bands: {band_names}")

        img_values = cube.values
        normalized_bands = []

        for band in band_names:
            arr = img_values[band_names.index(band)]
            pre_stats = (arr.min(), arr.max(), arr.mean())

            # Find which group this band belongs to
            group = None
            for g, specs in normalization_specs.items():
                if band in specs:
                    group = g
                    min_spec, max_spec = specs[band]
                    norm_func = self.NORMALIZE_FUNCS[group]
                    normalized = norm_func(arr, min_spec, max_spec)
                    post_stats = (normalized.min(), normalized.max(), normalized.mean())
                    logger.info(
                        f"Band {band}: group={group}, "
                        f"min={pre_stats[0]:.3f}->{post_stats[0]:.3f}, "
                        f"max={pre_stats[1]:.3f}->{post_stats[1]:.3f}"
                    )
                    normalized_bands.append(normalized)
                    break

            if group is None:
                logger.warning(
                    f"Band {band}: no normalization defined, leaving unchanged."
                )
                normalized_bands.append(arr.astype(np.float32))

        # Stack back into DataArray
        result_array = np.stack(normalized_bands, axis=1)
        da = xr.DataArray(
            result_array,
            dims=("t", "bands", "y", "x"),
            coords={
                "bands": band_names,
                "x": cube.coords["x"],
                "y": cube.coords["y"],
                "t": t_coords,
            },
        )
        logger.info(f"Normalization complete for model {self.model_id}")
        return da


# =============================================================================
# ONNX Classifier
# =============================================================================
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


def apply_udf_data(udf_data: UdfData) -> UdfData:
    cube = udf_data.datacube_list[0]
    context = udf_data.user_context.copy()

    cube = cube.get_array().transpose("bands", "t", "y", "x")

    crs = udf_data.proj["EPSG"] if udf_data.proj else None
    if crs is None:
        raise ValueError("EPSG code not found in projection information")

    logger.info(f"EPSG code determined for feature extraction: {crs}")

    lonlat_calculator = LonLatCalculator()
    normalizer = Normalizer(context)
    classifier = ONNXClassifier(context)

    cube_with_lonlat = lonlat_calculator.append_lonlat(cube, crs)
    normalized_cube = normalizer.apply_normalization(cube_with_lonlat)
    cube = cube.transpose("y", "x", "bands", "t")
    output = classifier.apply_model(normalized_cube)

    udf_data.datacube_list = [XarrayDataCube(output)]

    return udf_data
