import sys
import functools
import numpy as np
import xarray as xr
import logging
from typing import Dict, Tuple
from scipy.special import expit


# Setup logger
def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

logger = _setup_logging()

# Add ONNX paths
sys.path.append("onnx_deps")
sys.path.append("onnx_models")
import onnxruntime as ort

# Constants for sanitization
_INF_REPLACEMENT = 1e6
_NEG_INF_REPLACEMENT = -1e6

@functools.lru_cache(maxsize=1)
def _load_ort_session(model_name: str) -> ort.InferenceSession:
    """Loads an ONNX model and returns a cached ONNX runtime session."""
    return ort.InferenceSession(f"onnx_models/{model_name}")

def _get_expected_band_count(session: ort.InferenceSession) -> int:
    """
    Detect the expected number of input bands from the ONNX model input shape.
    Handles both channels-last (1, H, W, C) and channels-first (1, C, H, W).
    """
    shape = session.get_inputs()[0].shape
    logger.info(f"Model input shape from ONNX: {shape}")

    # Extract numeric dimensions (ignore None or dynamic names)
    numeric_dims = [dim for dim in shape if isinstance(dim, int)]
    logger.info(f"Numeric input dimensions: {numeric_dims}")

    if not numeric_dims:
        raise ValueError(f"Cannot determine band count from shape: {shape}")

    # Assume the last numeric dimension is the band count
    band_count = numeric_dims[-1]
    logger.info(f"Model expects {band_count} input bands.")
    return band_count


def _drop_optional_bands_by_name(cube: xr.DataArray) -> xr.DataArray:
    """
    Drop NDRE and EVI bands by name, if present.
    """
    drop_bands = ["NDRE", "EVI"]
    band_names = cube.coords["bands"].values

    to_drop = [b for b in drop_bands if b in band_names]
    if not to_drop:
        logger.info("No optional bands found to drop.")
        return cube

    logger.info(f"Dropping optional bands by name: {to_drop}")
    return cube.sel(bands=[b for b in band_names if b not in to_drop])

def preprocess_image(cube: xr.DataArray) -> Tuple[np.ndarray, Dict[str, xr.Coordinate], np.ndarray]:
    """
    Prepare the input cube for inference:
      - Transpose to (y, x, bands)
      - Sanitize NaN/Inf
      - Return batch tensor, coords, and invalid-value mask
    """
    # Reorder dims
    reordered = cube.transpose("y", "x", "bands")
    values = reordered.values.astype(np.float32)

    # Mask invalid entries
    mask_invalid = ~np.isfinite(values)

    # Replace NaN with 0, inf with large sentinel
    sanitized = np.where(np.isnan(values), 0.0, values) #TODO validate if this is okay
    sanitized = np.where(np.isposinf(sanitized), _INF_REPLACEMENT, sanitized)
    sanitized = np.where(np.isneginf(sanitized), _NEG_INF_REPLACEMENT, sanitized)

    # Add batch dimension
    input_tensor = sanitized[None, ...]
    logger.info(f"Preprocessed tensor shape={input_tensor.shape}")
    return input_tensor, reordered.coords, mask_invalid


def run_inference(
    session: ort.InferenceSession,
    input_name: str,
    input_tensor: np.ndarray
) -> np.ndarray:
    """Run ONNX session and remove batch dimension from output."""
    outputs = session.run(None, {input_name: input_tensor})
    pred = np.squeeze(outputs[0], axis=0)
    logger.info(f"Inference output shape={pred.shape}")
    return pred

#TODO
def postprocess_output(
    pred: np.ndarray,  # Shape: [y, x, bands]
    coords: Dict[str, xr.Coordinate],
    mask_invalid: np.ndarray  # Shape: [y, x, bands]
) -> xr.DataArray:
    """
    Appends winning class index as new band to predictions:
      - Keeps original prediction values
      - Adds new band (-1 for invalid, 0..n-1 for winning class)
    """

    # Apply sigmoid
    #sigmoid_probs = expit(pred)  # shape [y, x, bands]

    # Optionally pick highest prob if needed
    #class_index = np.argmax(pred, axis=-1, keepdims=True)
    
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
        coords={
            "y": coords["y"],
            "x": coords["x"],
            "bands": new_band_coords
        },
        attrs={"description": "Original preds, sigmoid probs, class index"}
    )


def apply_model(
    cube: xr.DataArray,
    model_path: str
) -> xr.DataArray:
    """
    Full inference pipeline:
      - Read ONNX model input shape
      - If model expects 15 bands → drop NDRE/EVI by name
      - Preprocess → infer → postprocess
    """
    session = _load_ort_session(model_path)
    expected_bands = _get_expected_band_count(session)

    band_names = cube.coords["bands"].values
    logger.info(f"Cube has {len(band_names)} bands: {band_names}")

    if expected_bands == 15:
        cube = _drop_optional_bands_by_name(cube)
    elif expected_bands not in (15, 17):
        logger.warning(f"Unexpected band count {expected_bands}, expected 15 or 17.")

    input_tensor, coords, mask_invalid = preprocess_image(cube)
    input_name = session.get_inputs()[0].name
    raw_pred = run_inference(session, input_name, input_tensor)
    return postprocess_output(raw_pred, coords, mask_invalid)


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Apply ONNX model per timestep in the datacube.
    """
    logger.info(f"apply_datacube received shape={cube.shape}, dims={cube.dims}")

    model_path  = str(context.get("model_path" ))

    logger.info(f"Applying model: {model_path}")

    cube = cube.transpose('y', 'x', 'bands', 't')

    if 't' in cube.dims:
        logger.info("Applying model per timestep via groupby-map.")
        return cube.groupby('t').map(lambda da: apply_model(da,  model_path))
    else:
        logger.info("Single timestep: applying model once.")
        return apply_model(cube,  model_path)