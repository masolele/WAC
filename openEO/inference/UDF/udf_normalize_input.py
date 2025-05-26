import numpy as np
import xarray as xr
import logging
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Show only the message
)
logger = logging.getLogger(__name__)

# Constants
NORM_PERCENTILES = np.array([
    [1.7417268007636313, 2.023298706048351],
    [1.7261204997060209, 2.038905204308012],
    [1.6798346251414997, 2.179592821212937],
    [2.3828939530384052, 2.7578332604178284],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351]
], dtype=np.float32)

EXPECTED_BANDS = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12",
    "NDVI", "VV", "VH", "DEM", "lon", "lat"
]

# --- Normalization helpers ---

def normalise_vv(raster: np.ndarray) -> np.ndarray:
    return np.clip((raster + 25) / 25, 0, 1).astype(np.float32)

def normalise_vh(raster: np.ndarray) -> np.ndarray:
    return np.clip((raster + 30) / 25, 0, 1).astype(np.float32)

def normalise_longitude(raster: np.ndarray) -> np.ndarray:
    return np.clip((raster + 180) / 360, 0, 1).astype(np.float32)

def normalise_latitude(raster: np.ndarray) -> np.ndarray:
    return np.clip((raster + 60) / 120, 0, 1).astype(np.float32)

def normalise_altitude(raster: np.ndarray) -> np.ndarray:
    return np.clip((raster + 400) / 8400, 0, 1).astype(np.float32)

def normalise_ndvi(raster: np.ndarray) -> np.ndarray:
    return np.clip((raster + 1) / 2, 0, 1).astype(np.float32)

def norm_optical(image: np.ndarray) -> np.ndarray:
    min_values = NORM_PERCENTILES[:, 0].reshape(9, 1, 1)
    scale_values = NORM_PERCENTILES[:, 1].reshape(9, 1, 1)

    image = np.log(image * 0.005 + 1)
    image = (image - min_values) / scale_values
    image = np.exp(image * 5 - 1)
    return (image / (image + 1)).astype(np.float32)

# --- Core logic ---

def validate_and_reorder_bands(
    data: xr.DataArray
) -> Tuple[xr.DataArray, Dict[str, int]]:
    """
    Ensure the input DataArray has all EXPECTED_BANDS in the correct order.

    Returns:
        Tuple of (reordered data, band name to index mapping).
        Raises ValueError if bands are missing.
    """
    current = list(data.coords["bands"].values)

    if current != EXPECTED_BANDS:
        try:
            data = data.sel(bands=EXPECTED_BANDS)
            logger.info(f"Reordered bands from {current} to {EXPECTED_BANDS}")
        except KeyError:
            missing = sorted(set(EXPECTED_BANDS) - set(current))
            logger.error(f"Missing required band(s): {missing}. Available: {current}")
            raise ValueError(f"Missing required band(s): {missing}")

    band_names = list(data.coords["bands"].values)
    try:
        band_indices = {b: band_names.index(b) for b in EXPECTED_BANDS}
        logger.info(f"Band indices mapped: {band_indices}")
    except ValueError:
        logger.error(f"Failed to map indices. Bands present: {band_names}")
        raise ValueError(f"Band order mismatch. Expected: {EXPECTED_BANDS}, Got: {band_names}")

    return data, band_indices

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Normalize input datacube for inference:
    - Optical bands (first 9): via log-transform then sigmoid-like scale
    - Others (VV, VH, NDVI, DEM, lon, lat): via fixed min/max scalings

    Args:
        cube (xr.DataArray): Input cube with dims (bands, y, x, t)
        context (dict): Unused, placeholder for UDF interface

    Returns:
        Normalized xr.DataArray with shape (15, y, x, t)
    """
    cube = cube.transpose("bands", "y", "x", "t")
    logger.info(f"Received data with shape: {cube.shape} and dims: {cube.dims}")

    reordered, band_idx = validate_and_reorder_bands(cube)
    vals = reordered.values

    # 1) Optical bands normalization
    optical = vals[:9, ...]
    mins = NORM_PERCENTILES[:, 0].reshape(9, 1, 1, 1)
    scales = NORM_PERCENTILES[:, 1].reshape(9, 1, 1, 1)

    normed_opt = np.log(optical * 0.005 + 1)
    normed_opt = (normed_opt - mins) / scales
    normed_opt = np.exp(normed_opt * 5 - 1)
    normed_opt = (normed_opt / (normed_opt + 1)).astype(np.float32)

    # 2) Scalar band normalization
    ndvi = normalise_ndvi(vals[band_idx["NDVI"], ...])
    vv   = normalise_vv(vals[band_idx["VV"], ...])
    vh   = normalise_vh(vals[band_idx["VH"], ...])
    dem  = normalise_altitude(vals[band_idx["DEM"], ...])
    lon  = normalise_longitude(vals[band_idx["lon"], ...])
    lat  = normalise_latitude(vals[band_idx["lat"], ...])

    # 3) Concatenate all normalized bands
    output = np.concatenate(
        [normed_opt, ndvi[None], vv[None], vh[None], dem[None], lon[None], lat[None]],
        axis=0
    )

    return xr.DataArray(
        output,
        dims=("bands", "y", "x", "t"),
        coords={**reordered.coords, "bands": EXPECTED_BANDS}
    )


    

    

