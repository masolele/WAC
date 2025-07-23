#TODO do normalisation in extractor instead of UDF

from openeo import UDF
from pathlib import Path
from datetime import datetime, timedelta
from normalization import normalize_cube

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
    return s2.aggregate_temporal_period(period = 'month', reducer = 'mean')


def compute_vegetation_indices(cube):
    """Compute vegetation indices and return as a merged cube"""
    ndvi = cube.ndvi(red='B04', nir='B08').add_dimension('bands', 'NDVI', 'bands')
    
    ndre = (cube.band('B08') - cube.band('B05')) / \
           (cube.band('B08') + cube.band('B05')) \
           .add_dimension('bands', 'NDRE', 'bands')
    
    output = ndvi.merge_cubes(ndre)
    
    evi_numerator = 2.5 * (cube.band('B08') - cube.band('B04'))
    evi_denominator = cube.band('B08') + 6 * cube.band('B04') - 7.5 * cube.band('B02') + 1
    evi = (evi_numerator / evi_denominator).add_dimension('bands', 'EVI', 'bands')

    output = output.merge_cubes(evi)

    return output

def compute_vegetation_indices(cube):
    """
    Compute all required vegetation indices
    """
    # NDVI
    ndvi = cube.ndvi(red='B04', nir='B08')
    ndvi = ndvi.add_dimension('bands', 'NDVI', 'bands')
    
    # NDRE: (NIR - RedEdge1)/(NIR + RedEdge1)
    ndre = (cube.band('B08') - cube.band('B05')) / (cube.band('B08') + cube.band('B05'))
    ndre = ndre.add_dimension('bands', 'NDRE', 'bands')
    
    # EVI: 2.5*(NIR - Red)/(NIR + 6*Red - 7.5*Blue + 1)
    numerator = 2.5 * (cube.band('B08') - cube.band('B04'))
    denominator = (
        cube.band('B08') + 
        6 * cube.band('B04') - 
        7.5 * cube.band('B02') + 
        1
    )
    evi = numerator / denominator
    evi = evi.add_dimension('bands', 'EVI', 'bands')
    output = ndvi
    #output = ndvi.merge_cubes(ndre)
    #output = output.merge_cubes(evi)
    
    # Add dimension labels
    return output


def load_sentinel1(conn, spatial_extent, temporal_extent, resolution, crs):
    """
    Load and process Sentinel-1 VV/VH global mosaics.

    There is an issue with the sentinel1 Global mosaic due to wich the first observation is always 0

    """
    orig_start = datetime.strptime(temporal_extent[0], "%Y-%m-%d")
    extended_start = (orig_start - timedelta(days=1)).strftime("%Y-%m-%d")
    extended_temporal = [extended_start, temporal_extent[1]]


    s1 = (
        conn.load_collection(
            'SENTINEL1_GLOBAL_MOSAICS',
            temporal_extent=extended_temporal,
            spatial_extent=spatial_extent,
            bands=['VV','VH']
        )
        .resample_spatial(resolution=resolution, projection=crs)
    )
    s1 = s1.apply(lambda x: 10 * x.log(base=10))

    return s1

#TODO looks wrong, should contain unique values
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

    
def load_input_cube(conn, spatial_extent, temporal_extent, max_cloud_cover = 85, resolution = 10, crs = "EPSG:3035"):

    """
    Main extractor: loads data cubes, merges them,
    applies normalization UDF, and returns the processed cube.
    """
    # Load all sources
    s2 = load_sentinel2(conn, spatial_extent, temporal_extent, max_cloud_cover, resolution, crs)
    s1 = load_sentinel1(conn, spatial_extent, temporal_extent, resolution, crs)
    veg_indices = compute_vegetation_indices(s2)
    latlon = compute_latlon(s2, spatial_extent, resolution, crs)
    dem = load_dem(conn, spatial_extent, resolution, crs)
    
    # Normalize cubes
    normalized_cubes = [normalize_cube(cube) for cube in [s2, veg_indices, s1, dem, latlon]] #TODO validate the order of bands
    
    # Merge all processed cubes
    output = normalized_cubes[0]
    for cube in normalized_cubes[1:]:
        output = output.merge_cubes(cube)
    return output

    

