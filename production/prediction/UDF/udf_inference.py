import sys
import functools
import numpy as np
import xarray as xr
import logging

def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = _setup_logging()

sys.path.append("onnx_deps")
sys.path.append("onnx_models") 

import onnxruntime as ort
# logger.info(message)

# Set environment variable to allow multiple OpenMP libraries to be loaded
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@functools.lru_cache(maxsize=1)
def _load_ort_session(model_name: str) -> ort.InferenceSession:
    """Loads an ONNX model and returns a cached ONNX runtime session."""
    return ort.InferenceSession(f"onnx_models/{model_name}")

def _preprocess(cube: xr.DataArray) -> np.ndarray:
    """Reorder dimensions, extract values, and sanitize NaNs/Infs."""
    reordered = cube.transpose("y", "x", "bands")
    img = reordered.values.astype(np.float32)[None, ...]  # → (1, y, x, bands)
    # Replace NaN/Inf so the model sees only finite numbers
    img = np.nan_to_num(img, nan=0.0, posinf=1e6, neginf=-1e6) #TODO how was model trained for nans?
    logger.info(f"Preprocessed input shape = {img.shape}")
    return img, reordered.coords

def _infer(session: ort.InferenceSession, input_name: str, img: np.ndarray) -> np.ndarray:
    """Run ONNX inference and drop the batch dimension."""
    raw = session.run(None, {input_name: img})[0]  # (1, y, x, new_bands)
    pred = np.squeeze(raw, axis=0)                # → (y, x, new_bands)
    logger.info(f"Raw ONNX output shape = {pred.shape}")
    return pred

def _postprocess(pred: np.ndarray, coords) -> xr.DataArray:
    """Slice off unwanted bands, compute argmax, and rebuild DataArray."""
    # Drop background band (assumes it's the first channel)
    scores = pred[..., 1:]                        # → (y, x, bands-1)
    # Mask any residual ±Inf (should be none) and NaNs
    scores = np.where(np.isinf(scores), np.nan, scores)
    classes = np.argmax(scores, axis=-1)          # → (y, x)
    # Stack class + scores along 'bands'
    out = np.concatenate([classes[..., None], scores], axis=-1)
    y, x = coords["y"], coords["x"]
    bands = np.arange(out.shape[-1])
    return xr.DataArray(out, dims=("x", "y", "bands"),
                        coords={"x": y, "u": x, "bands": bands})

def apply_ml(cube: xr.DataArray, model_path: str = "WAC_model_hansvrp.onnx") -> xr.DataArray:
    """
    Process the tile provided by openEO using the ONNX model.
    Splits work into preprocessing, inference, and postprocessing.
    """
    # Preprocess
    img, coords = _preprocess(cube)

    # Inference
    session = _load_ort_session(model_path)
    input_name = session.get_inputs()[0].name
    pred = _infer(session, input_name, img)

    # Postprocess
    output = _postprocess(pred, coords)
    logger.info(f"apply_ml returning DataArray with shape={output.shape}")
    return output


def apply_datacube(cube: xr.DataArray, context) -> xr.DataArray:
    """
    Apply ONNX model to each timestep independently in the datacube.
    """
    logger.info(f"Inference, received data with shape: {cube.shape} and dims: {cube.dims}")
    cube = cube.transpose('y', 'x', 'bands', 't')

    if 't' in cube.dims:
        logger.info(f"Detected time dimension 't'. Applying model for each timestep.")
        result = cube.groupby('t').map(apply_ml)
        return result
    else:
        logger.info(f"No time dimension detected. Applying model once.")
        return apply_ml(cube)