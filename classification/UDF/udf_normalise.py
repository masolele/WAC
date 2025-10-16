import numpy as np
import xarray as xr
import logging
import requests
import functools
from typing import List
from openeo.metadata import CollectionMetadata


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def get_model_metadata_from_stac(model_id: str, stac_api_url: str = "https://stac.openeo.vito.be") -> dict:
    """Fetch model metadata from STAC API"""
    try:
        # Get collection and item
        collection_id = "world-agri-commodities-models"
        url = f"{stac_api_url}/collections/{collection_id}/items/{model_id}"
        
        response = requests.get(url)
        response.raise_for_status()
        
        item = response.json()
        properties = item.get('properties', {})
        
        logger.info(f"Retrieved model metadata for {model_id}")
        return {
            'input_bands': properties.get('input_channels', []),
            'input_shape': properties.get('input_shape', 0),
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch model metadata: {e}")
        raise

def get_normalization_specs(input_bands: List[str]) -> dict:
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
            "lat": (-60, 60)
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

def _normalize_optical(arr: np.ndarray, min_spec: float, max_spec: float) -> np.ndarray:
    """Log-based normalization for optical bands."""
    arr = np.log(arr * 0.005 + 1)
    arr = (arr - min_spec) / (max_spec)
    arr = np.exp(arr * 5 - 1)
    return arr / (arr + 1)

def _normalize_linear(arr: np.ndarray, min_spec: float, max_spec: float) -> np.ndarray:
    """Linear min–max normalization for continuous variables."""
    arr = np.clip(arr, min_spec, max_spec)
    return (arr - min_spec) / (max_spec - min_spec)

NORMALIZE_FUNCS = {
    "optical": _normalize_optical,
    "linear": _normalize_linear,
}

def validate_bands(cube: xr.DataArray, expected_bands: list):
    band_names = list(cube.coords["bands"].values)
    logger.info(f"Input bands: {band_names}")
    logger.info(f"Expected bands from model: {expected_bands}")

    # Check for missing required bands
    missing_bands = [b for b in expected_bands if b not in band_names]
    if missing_bands:
        raise ValueError(f"Missing required bands: {missing_bands}. Got: {band_names}")

    # Check order
    if band_names != expected_bands:
        logger.warning(f"Band order mismatch: reordering from {band_names} to {expected_bands}")
        cube = cube.sel(bands=expected_bands)
    
    return cube

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Normalize bands based on model metadata from STAC API.
    """
    logger.info(f"Received data with shape: {cube.shape}, dims: {cube.dims}")
    
    # Get model ID from context
    model_id = context.get("model_id")
    if not model_id:
        raise ValueError("model_id must be provided in context")
    
    # Fetch model metadata from STAC
    model_metadata = get_model_metadata_from_stac(model_id)
    expected_bands = model_metadata['input_bands']
    
    logger.info(f"Using model: {model_id} with expected bands: {expected_bands}")
    logger.info(f"Model expects {model_metadata['input_shape']} input bands")
    
    # Validate and reorder bands
    cube = validate_bands(cube, expected_bands)
    
    # Get normalization specs for this specific model
    normalization_specs = get_normalization_specs(expected_bands)
    
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
                norm_func = NORMALIZE_FUNCS[group]
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
            logger.warning(f"Band {band}: no normalization defined, leaving unchanged.")
            normalized_bands.append(arr.astype(np.float32))

    # Stack back into DataArray
    result_array = np.stack(normalized_bands, axis=0)
    da = xr.DataArray(
        result_array,
        dims=("bands", "y", "x"),
        coords={
            "bands": band_names,
            "x": cube.coords["x"],
            "y": cube.coords["y"],
        },
    )
    logger.info(f"Normalization complete for model {model_id}")
    return da


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:

    model_id = context.get("model_id")
  
    # Fetch model metadata from STAC
    model_metadata = get_model_metadata_from_stac(model_id)
    input_bands = model_metadata.get('input_bands', []) 

    logger.info(f"Applying metadata with input bands: {input_bands}")
    return metadata.rename_labels(
        dimension = "bands",
        target = input_bands
    )

