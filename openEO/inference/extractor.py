#TODO do normalisation in extractor instead of UDF

from openeo import UDF
from pathlib import Path
from datetime import datetime, timedelta
from normalization import normalize_cube, normalize_band

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
    """
    Compute all required vegetation indices
    """
    # NDVI
    ndvi = cube.ndvi(red='B04', nir='B08')
    
    # NDRE: (NIR - RedEdge1)/(NIR + RedEdge1)
    ndre = (cube.band('B08') - cube.band('B05')) / (cube.band('B08') + cube.band('B05'))
    
    # EVI: 2.5*(NIR - Red)/(NIR + 6*Red - 7.5*Blue + 1)
    numerator = 2.5 * (cube.band('B08') - cube.band('B04'))
    denominator = (
        cube.band('B08') + 
        6 * cube.band('B04') - 
        7.5 * cube.band('B02') + 
        1
    )
    evi = numerator / denominator
    
    # Add dimension labels
    return ndvi.add_dimension('bands', 'NDVI', 'bands'), \
           ndre.add_dimension('bands', 'NDRE', 'bands'), \
           evi.add_dimension('bands', 'EVI', 'bands')


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

    
def load_input_WAC(conn, spatial_extent, temporal_extent, max_cloud_cover = 85, resolution = 10, crs = "EPSG:3035"):

    """
    Main extractor: loads data cubes, merges them,
    applies normalization UDF, and returns the processed cube.
    """
    # Load all sources
    s2 = load_sentinel2(conn, spatial_extent, temporal_extent, max_cloud_cover, resolution, crs)
    s2_norm = normalize_cube(s2, band_order=["B02", "B03", "B04"])

    ndvi, ndre, evi = compute_vegetation_indices(s2)
    ndvi_norm = normalize_band(ndvi, "NDVI")
    ndre_norm = normalize_band(ndre, "NDRE")
    evi_norm = normalize_band(evi, "EVI")


    s1 = load_sentinel1(conn, spatial_extent, temporal_extent, resolution, crs)
    s1_norm = normalize_cube(s1, band_order=["VV", "VH"])

    latlon = compute_latlon(s2, spatial_extent, resolution,crs)
    latlon_norm = normalize_cube(latlon, band_order=["lon", "lat"])

    dem = load_dem(conn, spatial_extent, resolution, crs)
    dem_norm = normalize_band(dem, "DEM")

    # Merge cubes
    # Merge all components
    input_cube = (
        s2_norm
        .merge_cubes(s1_norm)
        .merge_cubes(ndvi_norm)
        .merge_cubes(ndre_norm)
        .merge_cubes(evi_norm)
        .merge_cubes(dem_norm)
        .merge_cubes(latlon_norm)
    )

    return input_cube

    

