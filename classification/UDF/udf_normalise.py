import numpy as np
import xarray as xr
import logging
from typing import Dict, Tuple


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
        "NDVI": (-1, 1),
        "NDRE": (-1, 1),
        "EVI": (-1, 1),
        "DEM": (-400, 8000),
        "lon": (-180, 180),
        "lat": (-60, 60),
    },
}

def _normalize_optical(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    arr = np.log(arr * 0.005 + 1)
    arr = (arr - min_val) / (max_val)
    arr = np.exp(arr * 5 - 1)
    return (arr / (arr + 1))


def _normalize_linear(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    arr = np.clip(arr, min_val, max_val)
    return (arr - min_val) / (max_val - min_val)

NORMALIZE_FUNCS = {
    "optical": _normalize_optical,
    "linear": _normalize_linear,
}


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Flexible normalization UDF with detailed logging.
    - Normalizes whichever bands are present in the input cube
    - Uses NORMALIZATION_SPECS to decide method/range
    - Unknown bands are left unchanged (with a warning)
    """
    logger.info(f"Received data with shape: {cube.shape}, dims: {cube.dims}")
    band_names = list(cube.coords["bands"].values)
    logger.info(f"Input bands: {band_names}")

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
