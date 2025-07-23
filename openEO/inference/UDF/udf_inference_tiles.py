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

@functools.lru_cache(maxsize=1)
def build_gaussian_mask(height: int, width: int, sigma: float = 0.5) -> np.ndarray:
    """
    Builds a 2D Gaussian mask for CNN-style patch blending.
    
    sigma: relative spread (e.g., 0.125 → ~1/8th of patch size)
    """
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)
    gaussian = np.exp(-0.5 * ((xx**2 + yy**2) / sigma**2))
    return gaussian.astype(np.float32)



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

    gausian_mask = build_gaussian_mask(normalized_scores.shape[0], normalized_scores.shape[1])
    # Apply to each class score (broadcast along last dimension)
    weighted_scores = normalized_scores * gausian_mask[..., None]

    # Restore invalid pixels as NaN
    invalid_any = np.any(mask_invalid, axis=-1)
    weighted_scores[invalid_any] = 101 #TODO which value to set?

    output_arr = np.concatenate([gausian_mask[..., None], weighted_scores], axis=-1)

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

def _pad_to_64(arr: np.ndarray, pad_value: float = np.nan) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Given an array `arr` of shape (y_sub, x_sub, bands), where y_sub<=64 and x_sub<=64,
    pad it to exactly (64, 64, bands) by adding `pad_value` on bottom/right.
    Returns:
      - padded array of shape (64,64,bands)
      - (y_pad_before, y_pad_after)  → e.g. (0, 64-y_sub)
      - (x_pad_before, x_pad_after)  → e.g. (0, 64-x_sub)
    """
    y_sub, x_sub, bands = arr.shape
    pad_y = 64 - y_sub
    pad_x = 64 - x_sub
    pad_width = [(0, pad_y), (0, pad_x), (0, 0)]  # pad in bottom/right only
    padded = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
    return padded, (0, pad_y), (0, pad_x)

def _unpad_from_64(
    result_patch: np.ndarray,
    y_pad: Tuple[int, int],
    x_pad: Tuple[int, int]
) -> np.ndarray:
    """
    Given `result_patch` of shape (64,64,bands_out) and the paddings
    y_pad=(y_before,y_after), x_pad=(x_before,x_after), slice away the padded
    rows/columns and return an array of shape (64 - (y_before+y_after),
    64 - (x_before+x_after), bands_out).
    """
    y_before, y_after = y_pad
    x_before, x_after = x_pad
    y_stop = 64 - y_after
    x_stop = 64 - x_after
    return result_patch[y_before:y_stop, x_before:x_stop, :]


def apply_model_on_tiles(
    cube: xr.DataArray,
    model_path: str,
    tile_size: int = 64
) -> xr.DataArray:
    """
    Given a single‐timestep cube of shape (y_full, x_full, bands),
    break it into non‐overlapping tile_size×tile_size patches (padding edges),
    call apply_model() on each 64×64 patch, then unpad & stitch back.

    Returns: one DataArray of shape (y_full, x_full, bands_out).
    """
    y_full = cube.sizes["y"]
    x_full = cube.sizes["x"]

    # How many tiles along y and x?
    ny = int(np.ceil(y_full / tile_size))
    nx = int(np.ceil(x_full / tile_size))

    # Slice y=[0:64], x=[0:64] (clamped if smaller)
    top = cube.isel(
        y=slice(0, min(tile_size, y_full)),
        x=slice(0, min(tile_size, x_full))
    )
    arr0 = top.transpose("y", "x", "bands").values.astype(np.float32)
    padded0, y_pad0, x_pad0 = _pad_to_64(arr0, pad_value=np.nan)

    # Wrap into a DataArray so apply_model can run
    da0 = xr.DataArray(
        padded0,
        dims=("y", "x", "bands"),
        coords={
            "y": np.arange(tile_size),
            "x": np.arange(tile_size),
            "bands": top.coords["bands"]
        }
    )
    out0 = apply_model(da0, model_path)
    bands_out = out0.sizes["bands"]

    # Prepare a big container for the stitched result
    stitched = np.full((y_full, x_full, bands_out), np.nan, dtype=np.float32)


    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * tile_size
            x0 = ix * tile_size
            y1 = min(y0 + tile_size, y_full)
            x1 = min(x0 + tile_size, x_full)

            # Extract that sub‐window
            sub = cube.isel(
                y=slice(y0, y1),
                x=slice(x0, x1)
            )
            arr_sub = sub.transpose("y", "x", "bands").values.astype(np.float32)

            # Pad if needed (only on bottom/right)
            padded, y_pad, x_pad = _pad_to_64(arr_sub, pad_value=np.nan)

            # Wrap padded into DataArray for apply_model
            da_pad = xr.DataArray(
                padded,
                dims=("y", "x", "bands"),
                coords={
                    "y": np.arange(tile_size),
                    "x": np.arange(tile_size),
                    "bands": sub.coords["bands"]
                }
            )

            # Run the full preprocess → inference → postprocess for this tile
            out_patch = apply_model(da_pad, model_path)  
            # out_patch is (64,64,bands_out)

            # Unpad: remove any rows/cols that we added
            arr_out_patch = out_patch.values  # (64,64,bands_out)
            unpadded = _unpad_from_64(arr_out_patch, y_pad, x_pad)
            # Now unpadded.shape == (y1-y0, x1-x0, bands_out)

            # Write into the stitched array
            stitched[y0:y1, x0:x1, :] = unpadded

    return xr.DataArray(
        stitched,
        dims=("y", "x", "bands"),
        coords={
            "y": cube.coords["y"].values,
            "x": cube.coords["x"].values,
            "bands": np.arange(bands_out)
        }
    )


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Apply ONNX model per timestep in the datacube.
    """
    logger.info(f"apply_datacube received shape={cube.shape}, dims={cube.dims}")
    cube = cube.transpose('y', 'x', 'bands', 't')

    if 't' in cube.dims:
        logger.info("Applying model per timestep via groupby-map.")
        return cube.groupby('t').map(lambda da: apply_model_on_tiles(da, "WAC_model_hansvrp.onnx"))
    else:
        logger.info("Single timestep: applying model once.")
        return apply_model_on_tiles(cube, "WAC_model_hansvrp.onnx")