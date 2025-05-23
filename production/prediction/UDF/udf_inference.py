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


@functools.lru_cache(maxsize=2)
def _load_ort_session(model_name: str) -> ort.InferenceSession:
    """Loads an ONNX model and returns a cached ONNX runtime session."""
    return ort.InferenceSession(f"onnx_models/{model_name}")

def apply_ml(cube: xr.DataArray) -> xr.DataArray:
    """Process the neighborhood/tile provided by openEO using the ONNX model."""

    cube = cube.transpose('y', 'x', 'bands')
    logger.info(f"Inference Input data shape: {cube.shape}, dimensions: {cube.dims}")

    # TODO; remove?
    band_names = cube.coords['bands'].values
    for i, band_name in enumerate(band_names):
        band_data = cube.values[:,:,i]
        logger.info(f"Inference UDF  Band {i} ({band_name}): min={band_data.min():.6f}, max={band_data.max():.6f}, mean={band_data.mean():.6f}")

    #TODO do not hardcode model name
    session = _load_ort_session("WAC_model_hansvrp.onnx")

    input_name = session.get_inputs()[0].name
    image_input = np.expand_dims(cube.values, axis=0).astype(np.float32) #TODO is the conversion to float needed
    logger.info(f"Inference Model input shape: {image_input.shape}")

    pred = session.run(None, {input_name: image_input})[0]
    logger.info(f"Inference Model output shape: {pred.shape}")
    
    # take the prediction probabilities and set infitnity to nan
    raw_pred = np.where(np.isinf(pred[0]), np.nan, pred[0])
    pred_class = np.argmax(raw_pred, axis=-1) #TODO needed? doesn't the onnx model already return the class?

    # Create output DataArray with the same coordinates
    da = xr.DataArray(
        pred_class,
        dims=['y', 'x'],
        coords={'y': cube.coords['y'], 'x': cube.coords['x']}
    )

    return da


def apply_datacube(cube: xr.DataArray, context) -> xr.DataArray:
    """
    Function that is called for each chunk of data that is processed.
    The function name and arguments are defined by the UDF API.
    
    """
    logger.info(f"Apply Datacube Inference input data shape: {cube.shape}, dimensions: {cube.dims}")

    # Apply the model for each timestep in the chunk
    output_data = cube.groupby("t").apply(apply_ml)

    return output_data