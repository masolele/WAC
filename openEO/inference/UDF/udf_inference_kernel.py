import sys
import functools
import numpy as np
import xarray as xr
import logging
from typing import Dict, Tuple

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
    sanitized = np.where(np.isnan(values), 0.0, values)
    sanitized = np.where(np.isposinf(sanitized), _INF_REPLACEMENT, sanitized)
    sanitized = np.where(np.isneginf(sanitized), _NEG_INF_REPLACEMENT, sanitized)

    # Add batch dimension
    input_tensor = sanitized[None, ...]
    logger.info(f"Preprocessed tensor shape={input_tensor.shape}")
    return input_tensor, reordered.coords, mask_invalid



def postprocess_output(
    pred: np.ndarray,
    coords: Dict[str, xr.Coordinate],
    mask_invalid: np.ndarray
) -> xr.DataArray:
    """
    Combine class predictions and probabilities into DataArray,
    restoring NaNs for originally invalid pixels.
    """
    # Remove background (class 0), keep probabilities
    scores = pred[..., 1:]

    # Mask any Inf residuals
    scores = np.where(np.isinf(scores), np.nan, scores)

    # Compute class index
    classes = np.argmax(scores, axis=-1)

    # Stack class + scores
    output_arr = np.concatenate([classes[..., None], scores], axis=-1)

    # Restore any invalid pixel to NaN
    invalid_any = np.any(mask_invalid, axis=-1)
    output_arr[invalid_any] = np.nan

    # Build DataArray
    y_coords = coords["y"]
    x_coords = coords["x"]
    band_coords = np.arange(output_arr.shape[-1])

    return xr.DataArray(
        output_arr,
        dims=("y", "x", "bands"),
        coords={"y": y_coords, "x": x_coords, "bands": band_coords}
    )


def apply_model(
    cube: xr.DataArray,
    model_path: str,
    overlap: int = 32
) -> xr.DataArray:
    """
    Perform batched sliding window inference with a fixed overlap.
    Automatically infers window size from ONNX model input.
    Includes full preprocessing and postprocessing.
    """
    input_tensor, coords, mask_invalid = preprocess_image(cube)
    session = _load_ort_session(model_path)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # (batch, height, width, bands)

    window_height, window_width = input_shape[1], input_shape[2]
    stride_y = window_height - overlap
    stride_x = window_width - overlap

    H, W, C = input_tensor.shape[1:]  # (1, H, W, C)

    patches = []
    patch_coords = []

    for y in range(0, H - window_height + 1, stride_y):
        for x in range(0, W - window_width + 1, stride_x):
            patch = input_tensor[0, y:y + window_height, x:x + window_width, :]
            patches.append(patch)
            patch_coords.append((y, x))

    batch_tensor = np.stack(patches, axis=0)
    preds = session.run(None, {input_name: batch_tensor})[0]  # (N, h, w, classes)

    output = np.zeros((H, W, preds.shape[-1]), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)

    for i, (y, x) in enumerate(patch_coords):
        output[y:y + window_height, x:x + window_width, :] += preds[i]
        count[y:y + window_height, x:x + window_width, :] += 1

    output /= np.maximum(count, 1e-6)

    # Restore NaNs in invalid pixels and convert to labeled classes + scores
    result = postprocess_output(output, coords, mask_invalid)
    return result


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Apply ONNX model per timestep in the datacube.
    """
    logger.info(f"apply_datacube received shape={cube.shape}, dims={cube.dims}")
    cube = cube.transpose('y', 'x', 'bands', 't')

    if 't' in cube.dims:
        logger.info("Applying model per timestep via groupby-map.")
        return cube.groupby('t').map(lambda da: apply_model(da, "WAC_model_hansvrp.onnx"))
    else:
        logger.info("Single timestep: applying model once.")
        return apply_model(cube, "WAC_model_hansvrp.onnx")