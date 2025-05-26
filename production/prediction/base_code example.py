##% extractor

import openeo

RESOLUTION = 10
CRS = 'EPSG:3035'
SPATIAL_EXTENT = {
    'east': 4811500,
    'south': 2808500,
    'west': 4801500,
    'north': 2818500,
    'crs': CRS
}
TEMPORAL_EXTENT = ["2020-01-01", "2020-12-31"]

#%% utility

def filter_and_resample(cube, spatial_filter, resolution, crs, method='bilinear'):
    """
    Apply spatial filter and resample data cube.
    """
    return cube.filter_spatial(spatial_filter)





#%%

def extract_s2(connection: openeo.Connection, temporal_extent, max_cloud_cover=75):
    """
    Load and process Sentinel-2 data, applying cloud masking and temporal aggregation.
    """
    # Load and create cloud mask
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover=max_cloud_cover
    )

    mask = scl.process("to_scl_dilation_mask", data=scl)

    # Load and mask Sentinel-2 data
    cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        max_cloud_cover=max_cloud_cover
    ).mask(mask)

    # obtain a mean value per month
    cube = cube.aggregate_temporal_period(period="month", reducer="mean")
    cube = cube.apply_dimension(dimension="t", process="array_interpolate_linear")

    return cube


def extract_dem(connection: openeo.Connection):
    """
    Load and process DEM data.
    """
    cube =  connection.load_collection(
        collection_id="COPERNICUS_30",
        bands=["DEM"]
    ).max_time()

    return cube


def extract_s1(connection: openeo.Connection, temporal_extent):
    """
    Load and process Sentinel-1 SAR data.
    """
    cube = connection.load_collection(
        "SENTINEL1_GLOBAL_MOSAICS",
        temporal_extent=temporal_extent,
        bands=["VV", "VH"]
    )

    cube = cube.aggregate_temporal_period(period="month", reducer="mean")
    cube = cube.apply_dimension(dimension="t", process="array_interpolate_linear")
    cube = cube.apply(lambda x: 10 * x.log(base=10)) #TODO: check if this is correct


    return cube





#%% RAW input data #TODO checkpoint for validation input/output

connection = openeo.connect("https://openeo-staging.dataspace.copernicus.eu/")
connection.authenticate_oidc()


# Load and process Sentinel-2
s2_cube = extract_s2(connection, TEMPORAL_EXTENT).resample_spatial(projection=CRS,
                            resolution=RESOLUTION,
                            method="near").filter_bbox(SPATIAL_EXTENT)

# Load and process Sentinel-1
s1_cube = extract_s1(connection, TEMPORAL_EXTENT).resample_spatial(projection=CRS,
                            resolution=RESOLUTION,
                            method="near").filter_bbox(SPATIAL_EXTENT)

# Combine Sentinel-2 and Sentinel-1
result_cube = s2_cube.merge_cubes(s1_cube)

# Add DEM
dem_cube = extract_dem(connection).resample_spatial(projection=CRS,
                            resolution=RESOLUTION,
                            method="bilinear").filter_bbox(SPATIAL_EXTENT)

result_cube = result_cube.merge_cubes(dem_cube)





#%% preprocessing



