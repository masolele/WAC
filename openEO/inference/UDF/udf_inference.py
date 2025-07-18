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


def postprocess_output(
    pred: np.ndarray,
    coords: Dict[str, xr.Coordinate],
    mask_invalid: np.ndarray
) -> xr.DataArray:
    """
    Combine class predictions and probabilities into DataArray,
    restoring NaNs for originally invalid pixels.
    """
    # Remove background class (class 0)
    scores = pred[..., 1:].astype(np.float32)

    # Normalize probabilities across class axis
    score_sums = np.sum(scores, axis=-1, keepdims=True)
    normalized_scores = np.divide(
        scores,
        score_sums,
        out=np.zeros_like(scores),
        where=score_sums != 0
    )

    normalized_scores *= 100.0

    # Restore invalid pixels as NaN
    invalid_any = np.any(mask_invalid, axis=-1)
    normalized_scores[invalid_any] = 1000 #TODO which value to set?

    # Build DataArray
    y_coords = coords["y"]
    x_coords = coords["x"]
    band_coords = np.arange(normalized_scores.shape[-1])

    return xr.DataArray(
        normalized_scores,
        dims=("y", "x", "bands"),
        coords={"y": y_coords, "x": x_coords, "bands": band_coords}
    )


def apply_model(
    cube: xr.DataArray,
    model_path: str
) -> xr.DataArray:
    """
    Full inference pipeline: preprocess, infer, postprocess.
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