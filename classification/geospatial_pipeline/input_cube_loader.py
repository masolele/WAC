#TODO do normalisation in extractor instead of UDF

from openeo import UDF, DataCube, Connection
from pathlib import Path
from datetime import datetime, timedelta
from geospatial_pipeline.band_normalization import normalize_cube
from typing import Dict, List, Union


# Determine script directory
BASE_DIR = Path().parent.resolve()
UDF_DIR = BASE_DIR / 'UDF'


def load_sentinel2(
    conn: Connection,
    spatial_extent: Dict[str, Union[float, str]],
    temporal_extent: List[str],
    max_cloud_cover: int = 70,
    resolution: int = 10,
    crs: str = 'EPSG:3035'
) -> DataCube:
    """
    Load Sentinel-2 L2A data, apply cloud masking, and compute monthly temporal mean.

    Args:
        conn: OpenEO connection object.
        spatial_extent: Spatial bounds of the data.
        temporal_extent: Date range in ['YYYY-MM-DD', 'YYYY-MM-DD'] format.
        max_cloud_cover: Maximum allowed cloud cover percentage.
        resolution: Spatial resolution in meters.
        crs: Coordinate Reference System (e.g., 'EPSG:3035').

    Returns:
        DataCube: Monthly averaged and masked Sentinel-2 image cube.
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


def compute_vegetation_indices(cube: DataCube) -> DataCube:
    """
    Compute NDVI, NDRE, and EVI vegetation indices.

    Args:
        cube: Input Sentinel-2 data cube.

    Returns:
        DataCube: Cube with vegetation indices as new bands.
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

    #TODO need clarity on final input design for ML models
    #output = ndvi.merge_cubes(ndre)
    #output = output.merge_cubes(evi)
    
    # Add dimension labels
    return output


def load_sentinel1(
    conn: Connection,
    spatial_extent: Dict[str, Union[float, str]],
    temporal_extent: List[str],
    resolution: int,
    crs: str
) -> DataCube:
    """
    Load Sentinel-1 VV/VH mosaics and apply log transformation.

    Note:
        An issue with the dataset causes the first observation to be always 0.
        This is avoided by extending the start date by 1 day.

    Args:
        conn: OpenEO connection object.
        spatial_extent: Spatial bounds of the data.
        temporal_extent: Date range in ['YYYY-MM-DD', 'YYYY-MM-DD'] format.
        resolution: Spatial resolution in meters.
        crs: Coordinate Reference System (e.g., 'EPSG:3035').

    Returns:
        DataCube: Sentinel-1 log-transformed image cube.
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

#TODO validate output
def compute_latlon(
    sentinel2: DataCube,
    spatial_extent: Dict[str, Union[float, str]],
    resolution: int,
    crs: str
) -> DataCube:
    """
    Calculates lat/lon values for the given spatial extent.

    Args:
        sentinel2: Input Sentinel-2 data cube.
        spatial_extent: Spatial bounds of the data.
        resolution: Spatial resolution in meters.
        crs: Coordinate Reference System (e.g., 'EPSG:3035').

    Returns:
        DataCube: Cube with latitude and longitude as bands.
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


def load_dem(
    conn: Connection,
    spatial_extent: Dict[str, Union[float, str]],
    resolution: int,
    crs: str
) -> DataCube:
    """
    Load Digital Elevation Model (DEM) data and temporally reduce it if needed.

    Args:
        conn: OpenEO connection object.
        spatial_extent: Spatial bounds of the data.
        resolution: Spatial resolution in meters.
        crs: Coordinate Reference System (e.g., 'EPSG:3035').

    Returns:
        DataCube: DEM image cube.
    """
    dem = (
        conn.load_collection('COPERNICUS_30', spatial_extent=spatial_extent)
        .resample_spatial(resolution=resolution, projection=crs, method='bilinear')
    )
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension='t', reducer='mean')
    return dem

    
def load_input_cube(
    conn: Connection,
    spatial_extent: Dict[str, Union[float, str]],
    temporal_extent: List[str],
    max_cloud_cover: int = 85,
    resolution: int = 10,
    crs: str = "EPSG:3035"
) -> DataCube:
    """
    Main extractor function that loads and processes all required input data cubes,
    normalizes them, and returns a merged cube.

    Args:
        conn: OpenEO connection object.
        spatial_extent: Spatial bounds of the data.
        temporal_extent: Date range in ['YYYY-MM-DD', 'YYYY-MM-DD'] format.
        max_cloud_cover: Maximum allowed cloud cover percentage.
        resolution: Spatial resolution in meters.
        crs: Coordinate Reference System (default: 'EPSG:3035').

    Returns:
        DataCube: Final merged and normalized data cube.
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

    

