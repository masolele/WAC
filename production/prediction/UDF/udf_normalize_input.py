
import numpy as np
import xarray as xr
import logging
from typing import Tuple, Dict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format showing only the message
)
logger = logging.getLogger(__name__)

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

EXPECTED_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B11","B12", "NDVI",
            "VV","VH","DEM","lon","lat"]


#TODO move all of this out of a UDF to base openEO code
def normalise_vv(raster):
    raster = np.clip(raster, -25, 0)
    return (raster + 25) / 25

def normalise_vh(raster):
    raster = np.clip(raster, -30, -5)
    return (raster + 30) / 25

def normalise_longitude(raster):
    raster = np.clip(raster, -180, 180)
    return (raster + 180) / 360

def normalise_latitude(raster):
    raster = np.clip(raster, -60, 60)
    return (raster + 60) / 120

def normalise_altitude(raster):
    raster = np.clip(raster, -400, 8000)
    return (raster + 400) / 8400

def normalise_ndvi(raster):
    raster = np.clip(raster, -1, 1)
    return (raster + 1) / 2

def norm_optical(image):
    
    min_values = NORM_PERCENTILES[:, 0].reshape(9, 1, 1)
    scale_values = NORM_PERCENTILES[:, 1].reshape(9, 1, 1)
  
    image = np.log(image * 0.005 + 1)
    image = (image - min_values) / scale_values
    image = np.exp(image * 5 - 1)
    return (image / (image + 1)).astype(np.float32)


def validate_and_reorder_bands(
    data: xr.DataArray,
) -> Tuple[xr.DataArray, Dict[str, int]]:
    """
    Ensure that `data` has exactly the expected bands, in the expected order.

    - If bands are out of order, reorders via .sel and logs the change.
    - If any expected band is missing, logs an error and raises ValueError.
    - Returns (reordered_data, band_indices), where band_indices maps
      each band name to its position in the final DataArray.

    """

    current = list(data.coords["bands"].values)

    # reorder if needed (will KeyError if a band is missing)
    if current != EXPECTED_BANDS:
        try:
            data = data.sel(bands=EXPECTED_BANDS)
            logger.info(f"Reordered bands from {current} to {EXPECTED_BANDS}")
        except KeyError:
            missing = sorted(set(EXPECTED_BANDS) - set(current))
            logger.error(f"Missing required band(s): {missing}. Available: {current}")
            raise ValueError(f"Missing required band(s): {missing}")

    # now guaranteed: exactly the expected bands in order
    band_names = list(data.coords["bands"].values)

    # map indices
    try:
        band_indices = {b: band_names.index(b) for b in EXPECTED_BANDS}
        logger.info(f"Band indices mapped: {band_indices}")
    except ValueError:
        # should never happen
        logger.error(f"Failed to map indices. Bands present: {band_names}")
        raise ValueError(f"Band order mismatch. Expected: {EXPECTED_BANDS}, Got: {band_names}")

    return data, band_indices



def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    '''Normalise the input data cube for model inference.
    - Normalises the first 9 bands using the norm function
    - Normalises VV, VH, DEM, lon, lat separately
    - Outputs a 15-band cube
    '''
    cube = cube.transpose("bands", "y", "x", "t")
    logger.info(f"Received data with shape: {cube.shape} and dims: {cube.dims}")
    
    #TODO check if this occurs; if data is missing instead of throwing a value error we could also pad with nans?
    reordered, band_idx = validate_and_reorder_bands(cube)

    # Convert to numpy for operations
    vals = reordered.values
    
     # 1) Optical stack: first 9 bands → (9, y, x, t)
    optical = vals[:9, ...]
    # reshape percentiles to (9,1,1,1) so they broadcast over y,x,t
    mins = NORM_PERCENTILES[:, 0].reshape(9, 1, 1, 1)
    scales = NORM_PERCENTILES[:, 1].reshape(9, 1, 1, 1)
    # reuse your norm_optical logic, but on 4D
    normed_opt = np.log(optical * 0.005 + 1)
    normed_opt = (normed_opt - mins) / scales
    normed_opt = np.exp(normed_opt * 5 - 1)
    normed_opt = (normed_opt / (normed_opt + 1)).astype(np.float32)

    # 2) Per‐band scalar normalisations also just broadcast over t:
    ndvi = normalise_ndvi(vals[band_idx["NDVI"], ...])
    vv   = normalise_vv(vals[band_idx["VV"], ...])
    vh   = normalise_vh(vals[band_idx["VH"], ...])
    dem  = normalise_altitude(vals[band_idx["DEM"], ...])
    lon  = normalise_longitude(vals[band_idx["lon"], ...])
    lat  = normalise_latitude(vals[band_idx["lat"], ...])

    # 3) Stack back into (15, y, x, t)
    out = np.concatenate(
        [normed_opt, ndvi[None], vv[None], vh[None], dem[None], lon[None], lat[None]],
        axis=0
    )

    return xr.DataArray(
        out,
        dims=("bands", "y", "x", "t"),
        coords={ **reordered.coords, "bands": EXPECTED_BANDS }
    )


    

    

