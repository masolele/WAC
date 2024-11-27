import geojson
import openeo
import openeo.processes as eop

import geopandas as gpd
import pandas as pd
from helper.eo_utils import compute_percentiles
from helper.BAP_utils import *
from helper.s3proxy_utils import upload_geoparquet_file

# --- Utility Functions ---
def divide_vv_vh_ratio(bands):
    """
    Add VH/VV ratio as a new band.
    """
    vv, vh = bands[0], bands[1]
    return eop.array_append(bands, eop.divide(vh, vv))

# --- Extractor Functions ---
def extract_agera5(connection: openeo.Connection, temporal_extent):
    """
    Load AGERA5 climate data for the specified temporal extent.
    """
    cube = connection.load_stac(
        "https://s3.waw3-1.cloudferro.com/swift/v1/agera/stac/collection.json",
        temporal_extent=temporal_extent,
        bands=["precipitation-flux", "temperature-mean"],
    )
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

    return compute_percentiles(cube)

#TODO multi resolution
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

    # Aggregate monthly and interpolate missing values
    cube = cube.aggregate_temporal_period(period="month", reducer="mean")
    cube = cube.apply_dimension(dimension="t", process="array_interpolate_linear")

    return compute_percentiles(cube)


#TODO; integrate
def extract_s2_BAP(connection: openeo.Connection, temporal_extent, area, resolution=20, max_cloud_cover=75):

    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover=max_cloud_cover
    ).resample_spatial(resolution)

    scl = scl.apply(lambda x: eop.if_(eop.is_nan(x), 0, x))
    cloud_mask =  calculate_cloud_mask(scl)

    ## Calculate the individual scores which will be combined in a weighted sum

    #calculate the coverage score
    coverage_score = calculate_cloud_coverage_score(cloud_mask, area, scl)

    #datescore
    date_score = calculate_date_score(scl)

    #distance to cloud score
    dtc_score = calculate_distance_to_cloud_score(cloud_mask, resolution)

    ##Aggregate the scores
    score = aggregate_BAP_scores(dtc_score, date_score, coverage_score)
    score = score.mask(scl.band("SCL") == 0)

    ##Create a mask
    rank_mask = create_rank_mask(score)

    ##Create a composite
    sentinel2 = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent = temporal_extent,
        bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        max_cloud_cover=max_cloud_cover
    )

    #select the first day per month
    composite = sentinel2.mask(rank_mask).mask(cloud_mask).aggregate_temporal_period("month","first")

    return(compute_percentiles(composite))

# --- Job Submission Function ---
def wac_extraction_job(row: pd.Series, connection: openeo.Connection, **kwargs) -> openeo.BatchJob:
    """
    Create and submit a processing job for the given GeoDataFrame row.
    """
    
    temporal_extent = row["temporal_extent"]
    geometry = geojson.loads(row["geometry"])
    crs = row.crs

    # Upload geometry as Parquet file
    features = gpd.GeoDataFrame.from_features(geometry).set_crs(crs)
    spatial_filter_url = upload_geoparquet_file(features, connection)

    # Extract S2
    s2_cube = extract_s2(connection, temporal_extent).filter_spatial(
        connection.load_url(spatial_filter_url, format="Parquet")
    )

    # add S1
    s1_cube = extract_s1(connection, temporal_extent).filter_spatial(
        connection.load_url(spatial_filter_url, format="Parquet")
    )
    
    result_cube = s2_cube.merge_cubes(s1_cube)

    # resample and add DEM
    dem_cube = extract_dem(connection).filter_spatial(
        connection.load_url(spatial_filter_url, format="Parquet")
    )
    dem_cube = dem_cube.resample_spatial(resolution = 10, projection = crs, method =  'bilineair')

    result_cube = result_cube.merge_cubes(dem_cube)

    # resample and add AGERA5
    agera_cube = extract_agera5(connection, temporal_extent).filter_spatial(
        connection.load_url(spatial_filter_url, format="Parquet")
    )
    agera_cube = agera_cube.resample_spatial(resolution = 10, projection = crs, method = 'bilineair')
 
    result_cube = result_cube.merge_cubes(agera_cube)


    job = result_cube.create_job(
        format="NetCDF",
        sample_by_feature = True,
    )

    return job

