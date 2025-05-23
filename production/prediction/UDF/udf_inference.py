import sys
import functools
import numpy as np
import xarray as xr
import logging
from typing import Dict

def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = _setup_logging()

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

def preprocess_image(cube: xr.DataArray) -> tuple[np.ndarray, Dict, np.ndarray]:
    """
    Reorder dimensions, extract values, and sanitize NaNs/Infs.

    Returns:
        input_tensor (np.ndarray): 4D input array for model (batch, y, x, bands).
        coords (dict): Coordinates for y, x, and bands dims.
        mask_invalid (np.ndarray): Boolean mask where original values were NaN or Inf.
    """
    # Reorder dimensions to (y, x, bands)
    reordered = cube.transpose("y", "x", "bands")
    values = reordered.values.astype(np.float32)

    # Identify invalid values
    mask_invalid = ~np.isfinite(values)

    # Sanitize for model input
    sanitized = np.where(np.isnan(values), 0.0, values)
    sanitized = np.where(np.isposinf(sanitized), _INF_REPLACEMENT, sanitized)
    sanitized = np.where(np.isneginf(sanitized), _NEG_INF_REPLACEMENT, sanitized)

    # Add batch dimension
    input_tensor = sanitized[None, ...]
    logger.info(f"Preprocessed input tensor shape={input_tensor.shape}")
    return input_tensor, reordered.coords, mask_invalid

def run_inference(
    session: ort.InferenceSession,
    input_name: str,
    input_tensor: np.ndarray
) -> np.ndarray:
    """
    Execute the ONNX model and return the raw prediction without batch dimension.
    """
    outputs = session.run(None, {input_name: input_tensor})
    raw_pred = outputs[0]
    pred = np.squeeze(raw_pred, axis=0)
    logger.info(f"Inference output shape={pred.shape}")
    return pred

def postprocess_output(
    pred: np.ndarray,
    coords: Dict,
    mask_invalid: np.ndarray
) -> xr.DataArray:
    """
    Convert raw prediction to a DataArray, restore invalid pixels to NaN.

    Args:
        pred (np.ndarray): Model output array (y, x, new_bands).
        coords (dict): Original coordinates for y and x dims.
        mask_invalid (np.ndarray): Mask of originally invalid pixels (y, x, bands).

    Returns:
        xr.DataArray: Postprocessed result with invalid pixels set to NaN.
    """
    # Drop background channel
    scores = pred[..., 1:]

    # Mask any residual Inf
    scores = np.where(np.isinf(scores), np.nan, scores)

    # Compute predicted classes
    classes = np.argmax(scores, axis=-1)

    # Combine class labels with scores
    output_array = np.concatenate([classes[..., None], scores], axis=-1)

    # Restore original NaN mask: for any pixel/location where any band was invalid,
    # set all output bands to NaN
    invalid_mask_any = np.any(mask_invalid, axis=-1)
    output_array[invalid_mask_any] = np.nan

    # Build DataArray
    y_coords = coords["y"]
    x_coords = coords["x"]
    band_coords = np.arange(output_array.shape[-1])

    return xr.DataArray(
        output_array,
        dims=("y", "x", "bands"),
        coords={"y": y_coords, "x": x_coords, "bands": band_coords}
    )

def apply_model(
    cube: xr.DataArray,
    model_path: str
) -> xr.DataArray:
    """
    Full pipeline: preprocess input, run inference, and postprocess output.
    """
    input_tensor, coords, mask_invalid = preprocess_image(cube)
    session = _load_ort_session(model_path)
    input_name = session.get_inputs()[0].name
    raw_pred = run_inference(session, input_name, input_tensor)
    result = postprocess_output(raw_pred, coords, mask_invalid)
    logger.info(f"apply_model result shape={result.shape}")
    return result


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Apply ONNX model to each timestep independently in the datacube.
    """
    logger.info(f"Inference, received data with shape: {cube.shape} and dims: {cube.dims}")
    cube = cube.transpose('y', 'x', 'bands', 't')

    if 't' in cube.dims:
        logger.info(f"Detected time dimension 't'. Applying model for each timestep.")
        result = cube.groupby('t').map(apply_model)
        return result
    else:
        logger.info(f"No time dimension detected. Applying model once.")
        return apply_model(cube)