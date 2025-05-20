
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
    logger.info(f"Received data with shape: {{original_data.shape}} and dims: {{original_data.dims}}")
    
    #TODO check if this occurs; if data is missing instead of throwing a value error we could also pad with nans?
    reordered_data, band_indices = validate_and_reorder_bands(cube)

    # Convert to numpy for operations
    img_values = reordered_data.values
    
    # Normalize the first 9 bands (optical) as a single operation
    optical_bands = [img_values[band_indices[b]] for b in EXPECTED_BANDS[:9]]
    optical_bands_stack = np.stack(optical_bands, axis=0)
    normalised_optical = norm_optical(optical_bands_stack)
    
    # Normalise individual bands
    #TODO use standard openEO process for normalisation
    ndvi_normalised = normalise_ndvi(img_values[band_indices["NDVI"]])
    vv_normalised = normalise_vv(img_values[band_indices["VV"]])
    vh_normalised = normalise_vh(img_values[band_indices["VH"]])
    dem_normalised = normalise_altitude(img_values[band_indices["DEM"]])
    lon_normalised = normalise_longitude(img_values[band_indices["lon"]])
    lat_normalised = normalise_latitude(img_values[band_indices["lat"]])
    

    # TODO; remove?
    band_names = cube.coords['bands'].values
    for i, band_name in enumerate(band_names):
        band_data = cube.values[:,:,i]
        logger.info(f"  Band {i} ({band_name}): min={band_data.min():.6f}, max={band_data.max():.6f}, mean={band_data.mean():.6f}")
    
    # Stack all bands together: 9 optical + NDVI + VV + VH + DEM + Lon + Lat = 15 bands
    result_array = np.stack(
        [
            *normalised_optical,  # Unpack the 9 normalised optical bands
            ndvi_normalised,
            vv_normalised,
            vh_normalised,
            dem_normalised,
            lon_normalised,
            lat_normalised
        ],
        axis=0
    )
    

    # build DataArray with the same dims
    da = xr.DataArray(
        result_array,
        dims=("bands", "y", "x"),
        coords={
            "bands": EXPECTED_BANDS,
            # Here we *overwrite* the x/y coords to be the new geographic grid
            "x": reordered_data.coords['x'],
            "y": reordered_data.coords['y']
        },
    )


    return da
    

    

