from openeo import UDF
from pathlib import Path


# Determine script directory
BASE_DIR = Path().parent.resolve()
UDF_DIR = BASE_DIR / 'UDF'



def load_sentinel2(conn, spatial_extent, temporal_extent, max_cloud_cover, resolution, crs):
    """
    Load, mask, and temporal-reduce Sentinel-2 L2A bands.
    """
    s2 = (
        conn.load_collection(
            'SENTINEL2_L2A',
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands=["B02","B03","B04","B05","B06","B07","B08","B11","B12"],
            max_cloud_cover=max_cloud_cover
        )
        .resample_spatial(resolution=resolution, projection=crs)
    )

    scl = (
        conn.load_collection(
            'SENTINEL2_L2A',
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands=["SCL"],
            max_cloud_cover=max_cloud_cover
        )
        .resample_spatial(resolution=resolution, projection=crs)
    )

    mask = scl.process('to_scl_dilation_mask', data=scl)
    s2 = s2.mask(mask)
    return s2.reduce_dimension(dimension='t', reducer='mean')


def compute_ndvi(cube):
    """
    Compute NDVI and temporal-reduce if needed.
    """
    ndvi = cube.ndvi(red='B04', nir='B08').add_dimension('bands', 'NDVI', 'bands')
    if ndvi.metadata.has_temporal_dimension():
        ndvi = ndvi.reduce_dimension(dimension='t', reducer='mean')
    return ndvi


def load_sentinel1(conn, spatial_extent, temporal_extent, resolution, crs):
    """
    Load and process Sentinel-1 VV/VH global mosaics.
    """
    s1 = (
        conn.load_collection(
            'SENTINEL1_GLOBAL_MOSAICS',
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            bands=['VV','VH']
        )
        .resample_spatial(resolution=resolution, projection=crs)
    )
    s1 = s1.apply(lambda x: 10 * x.log(base=10))
    return s1.reduce_dimension(dimension='t', reducer='mean')


def compute_latlon(sentinel2, spatial_extent, resolution, crs):
    """
    Apply lat/lon UDF to Sentinel-2 cube and temporal-reduce if needed.
    """
    # Inline UDF loading
    context = {
        'west': spatial_extent['west'],
        'south': spatial_extent['south'],
        'east': spatial_extent['east'],
        'north': spatial_extent['north'],
        'crs': spatial_extent['crs']
    }
    udf_latlon = UDF.from_file(UDF_DIR / 'udf_lat_lon.py', context=context)

    latlon = (
        sentinel2
        .apply(process=udf_latlon)
        .resample_spatial(resolution=resolution, projection=crs)
        .rename_labels('bands', ['lon','lat'])
    )
    if latlon.metadata.has_temporal_dimension():
        latlon = latlon.reduce_dimension(dimension='t', reducer='mean')
    return latlon


def load_dem(conn, spatial_extent, resolution, crs):
    """
    Load DEM and temporal-reduce if needed.
    """
    dem = (
        conn.load_collection('COPERNICUS_30', spatial_extent=spatial_extent)
        .resample_spatial(resolution=resolution, projection=crs, method='bilinear')
    )
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension='t', reducer='mean')
    return dem


def load_input_WAC(conn, spatial_extent, temporal_extent, max_cloud_cover = 85, resolution = 10, crs = "EPSG:3035"):

    """
    Main extractor: loads data cubes, merges them,
    applies normalization UDF, and returns the processed cube.
    """
    # Load all sources
    s2 = load_sentinel2(conn, spatial_extent, temporal_extent, max_cloud_cover, resolution, crs)
    ndvi = compute_ndvi(s2)
    s1 = load_sentinel1(conn, spatial_extent, temporal_extent, resolution, crs)
    latlon = compute_latlon(s2, spatial_extent, resolution,crs)
    dem = load_dem(conn, spatial_extent, resolution, crs)

    # Merge cubes
    input_cube = s2.merge_cubes(ndvi).merge_cubes(s1).merge_cubes(dem).merge_cubes(latlon)

    # Normalize
    udf_norm = UDF.from_file(UDF_DIR / 'udf_normalize_input.py')
    norm_cube = input_cube.apply(process=udf_norm)

    return norm_cube

    

