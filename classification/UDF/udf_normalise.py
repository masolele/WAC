import numpy as np
import xarray as xr
import logging
from typing import Dict, Tuple, List


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Normalization specifications
NORMALIZATION_SPECS = {
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
        "VV": (-25, 0),
        "VH": (-30, -5),
        "DEM": (-400, 8000),
        "lon": (-180, 180),
        "lat": (-60, 60),
        "NDVI": (-1, 1),
        "NDRE": (-1, 1),
        "EVI": (-1, 1)
    },
}

def _normalize_optical(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Log-based normalization for optical bands."""
    arr = np.log(arr * 0.005 + 1)
    arr = (arr - min_val) / (max_val)
    arr = np.exp(arr * 5 - 1)
    return arr / (arr + 1)


def _normalize_linear(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Linear min–max normalization for continuous variables."""
    arr = np.clip(arr, min_val, max_val)
    return (arr - min_val) / (max_val - min_val)

NORMALIZE_FUNCS = {
    "optical": _normalize_optical,
    "linear": _normalize_linear,
}

def get_expected_bands() -> List[str]:
    """
    Derive expected band order directly from NORMALIZATION_SPECS.
    Preserves the order in which groups and bands were defined.
    """
    expected = []
    for group_bands in NORMALIZATION_SPECS.values():
        expected.extend(group_bands.keys())
    return expected

def validate_bands(cube: xr.DataArray, expected_bands: list):
    """
    Validate presence and order of required bands in a data cube.
    
    Ensures that:
      1. All required bands are present.
      2. Bands are in the correct order.

    Args:
        cube (xr.DataArray):
            Input data cube with a 'bands' coordinate.
        expected_bands (list):
            Ordered list of band names required for processing.

    Returns:
        xr.DataArray:
            Data cube with bands in the correct order.

    Raises:
        ValueError: If any required bands are missing.
    """
    band_names = list(cube.coords["bands"].values)
    logger.info(f"Input bands: {band_names}")

    # Check for missing bands
    missing_bands = [b for b in expected_bands if b not in band_names]
    if missing_bands:
        raise ValueError(f"Missing required bands: {missing_bands}. Got: {band_names}")

    # Reorder if needed
    if band_names != expected_bands:
        raise ValueError(f"Band order mismatch: {band_names} vs {expected_bands}")



def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:

    """
    Normalize all bands in an input data cube according to predefined specifications.

    Steps:
      1. Derive expected band order from NORMALIZATION_SPECS.
      2. Validate band presence and order.
      3. Apply normalization function per band based on its group.

    Args:
        cube (xr.DataArray):
            Input data cube with dimensions ("bands", "y", "x").

    Returns:
        xr.DataArray:
            Normalized data cube with same shape, dimensions, and band names.

    Raises:
        ValueError: If required bands are missing or in the wrong order.
    """
     
    logger.info(f"Received data with shape: {cube.shape}, dims: {cube.dims}")

    # --- Validate & reorder bands in one call ---
    #expected_bands = get_expected_bands()
    #validate_bands(cube, expected_bands)

    # --- Normalization logic stays unchanged ---
    band_names = list(cube.coords["bands"].values)
    logger.info(f"Normalizing bands: {band_names}")

    img_values = cube.values
    normalized_bands = []
    output_band_names = []
    for band in band_names:
        arr = img_values[band_names.index(band)]
        pre_stats = (arr.min(), arr.max(), arr.mean())
        # Find which group this band belongs to
        group = None
        for g, specs in NORMALIZATION_SPECS.items():
            if band in specs:
                group = g
                min_val, max_val = specs[band]
                norm_func = NORMALIZE_FUNCS[group]
                normalized = norm_func(arr, min_val, max_val)
                post_stats = (normalized.min(), normalized.max(), normalized.mean())
                logger.info(
                    f"Band {band}: group={group}, "
                    f"min={pre_stats[0]:.3f}->{post_stats[0]:.3f}, "
                    f"max={pre_stats[1]:.3f}->{post_stats[1]:.3f}, "
                    f"mean={pre_stats[2]:.3f}->{post_stats[2]:.3f}"
                )
                normalized_bands.append(normalized)
                output_band_names.append(band)
                break

        if group is None:
            logger.warning(f"Band {band}: no normalization defined, leaving unchanged.")
            post_stats = pre_stats
            logger.info(
                f"Band {band}: kept as-is, "
                f"min={pre_stats[0]:.3f}, max={pre_stats[1]:.3f}, mean={pre_stats[2]:.3f}"
            )
            normalized_bands.append(arr.astype(np.float32))
            output_band_names.append(band)

    # Stack back into DataArray
    result_array = np.stack(normalized_bands, axis=0)
    da = xr.DataArray(
        result_array,
        dims=("bands", "y", "x"),
        coords={
            "bands": output_band_names,
            "x": cube.coords["x"],
            "y": cube.coords["y"],
        },
    )
    logger.info(f"Normalization complete. Output bands: {output_band_names}")
    return da
