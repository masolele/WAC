import numpy as np
import xarray as xr
import logging
from openeo.udf import XarrayDataCube

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def lnp(message):
    logger.info(message)

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

def normalise_ndre(raster):
    raster = np.clip(raster, -1, 1)
    return (raster + 1) / 2

def normalise_evi(raster):
    raster = np.clip(raster, -1, 1)
    return (raster + 1) / 2

def norm(image):
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
    
    if image.ndim == 3 and image.shape[0] == 9:
        min_values = NORM_PERCENTILES[:, 0].reshape(9, 1, 1)
        scale_values = NORM_PERCENTILES[:, 1].reshape(9, 1, 1)
    else:
        min_values = NORM_PERCENTILES[:, 0]
        scale_values = NORM_PERCENTILES[:, 1]
    
    image = np.log(image * 0.005 + 1)
    image = (image - min_values) / scale_values
    image = np.exp(image * 5 - 1)
    return (image / (image + 1)).astype(np.float32)

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    '''
    Normalise the input data cube for model inference.
    - Normalises the first 9 bands using the norm function
    - NDVI (clipped [-1,1], left as-is)
    - NDRE, EVI normalised to [0,1]
    - Normalises VV, VH, DEM, lon, lat
    - Outputs a 17-band cube
    '''
    original_data = cube.get_array()
    lnp(f"Received data with shape: {original_data.shape} and dims: {original_data.dims}")
    
    expected_order = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12",
                      "NDVI", "NDRE", "EVI", "VV", "VH", "DEM", "lon", "lat"]
    
    if list(original_data.coords['bands'].values) != expected_order:
        lnp(f"Reordering bands from {original_data.coords['bands'].values} to {expected_order}")
        x_img = original_data.sel(bands=expected_order)
    else:
        x_img = original_data
    
    band_names = x_img.coords['bands'].values
    
    for band in expected_order:
        if band not in band_names:
            raise ValueError(f"Missing required band: {band}. Available bands: {band_names}")
    
    if list(band_names) != expected_order:
        raise ValueError(f"Band order mismatch. Expected: {expected_order}, Got: {list(band_names)}")
    
    band_indices = {band: list(band_names).index(band) for band in expected_order}
    
    img_values = x_img.values
    
    # Normalise optical
    optical_bands = [img_values[band_indices[b]] for b in expected_order[:9]]
    optical_bands_stack = np.stack(optical_bands, axis=0)
    normalised_optical = norm(optical_bands_stack)
    
    # Vegetation indices
    ndvi = np.clip(img_values[band_indices["NDVI"]], -1, 1)  # unchanged
    ndre_normalised = normalise_ndre(img_values[band_indices["NDRE"]])
    evi_normalised = normalise_evi(img_values[band_indices["EVI"]])
    
    # Other bands
    vv_normalised = normalise_vv(img_values[band_indices["VV"]])
    vh_normalised = normalise_vh(img_values[band_indices["VH"]])
    dem_normalised = normalise_altitude(img_values[band_indices["DEM"]])
    lon_normalised = normalise_longitude(img_values[band_indices["lon"]])
    lat_normalised = normalise_latitude(img_values[band_indices["lat"]])
    
    # Stack all 17
    result_array = np.stack(
        [
            *normalised_optical,
            ndvi,
            ndre_normalised,
            evi_normalised,
            vv_normalised,
            vh_normalised,
            dem_normalised,
            lon_normalised,
            lat_normalised
        ],
        axis=0
    )
    
    new_bands = expected_order
    
    if result_array.shape[0] != len(new_bands):
        raise ValueError(f"Band count mismatch: data shape {result_array.shape[0]} != label count {len(new_bands)}")    
    
    output_data = xr.DataArray(
        result_array,
        dims=['bands', 'y', 'x'],
        coords={'bands': new_bands, 'y': x_img.coords['y'], 'x': x_img.coords['x']}
    )
    
    lnp(f"Final output dimensions: {output_data.dims} with shape {output_data.shape}") 
    lnp(f"Final bands: {output_data.coords['bands'].values}") 
    
    return XarrayDataCube(output_data)