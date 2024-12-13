import geojson
import openeo
import ast

import geopandas as gpd
import pandas as pd
from helper.eo_utils import compute_percentiles, assign_max_band_label
from helper.s3proxy_utils import upload_geoparquet_file

# --- Utility Functions ---
def filter_and_resample(cube, spatial_filter, resolution, crs, method='bilinear'):
    """
    Apply spatial filter and resample data cube.
    """
    return cube.filter_spatial(spatial_filter).resample_spatial(resolution=resolution, projection=crs, method=method)

def prepare_geometry(row: gpd.GeoSeries, connection: openeo.Connection):
    """
    Upload geometry and return its spatial filter URL.
    """
    geometry = geojson.loads(row["geometry"])
    features = gpd.GeoDataFrame.from_features(geometry).set_crs(row.crs)
    return upload_geoparquet_file(features, connection)

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

def extract_biome(connection: openeo.Connection):
    """
    Load the biome information.
    """
    cube = connection.load_stac(
        "https://stac.openeo.vito.be/collections/biomes",
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

    return cube

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

    return cube



# --- Job Submission Function ---
def wac_extraction_job(row: pd.Series, connection: openeo.Connection, **kwargs) -> openeo.BatchJob:
    """
    Create and submit a processing job for the given GeoDataFrame row.
    """

    if isinstance(row["temporal_extent"], str):
        temporal_extent = ast.literal_eval(temporal_extent)
    else:
        temporal_extent = row["temporal_extent"]
    spatial_filter_url = prepare_geometry(row, connection)

    # Load and process Sentinel-2
    s2_cube = extract_s2(connection, temporal_extent).filter_spatial(
        connection.load_url(spatial_filter_url, format="Parquet")
    )

    # Load and process Sentinel-1
    s1_cube = extract_s1(connection, temporal_extent).filter_spatial(
        connection.load_url(spatial_filter_url, format="Parquet")
    )
    
    # Combine Sentinel-2 and Sentinel-1
    result_cube = s2_cube.merge_cubes(s1_cube)

    # Add DEM
    dem_cube = filter_and_resample(
        extract_dem(connection),
        connection.load_url(spatial_filter_url, format="Parquet"),
        resolution=int(row.resolution),
        crs=row.crs,
    )
    result_cube = result_cube.merge_cubes(dem_cube)

    # Add AGERA5
    agera_cube = filter_and_resample(
        extract_agera5(connection, temporal_extent),
        connection.load_url(spatial_filter_url, format="Parquet"),
        resolution=int(row.resolution),
        crs=row.crs,
    )
    result_cube = result_cube.merge_cubes(agera_cube)


    # Add biome
    biome = filter_and_resample(
        extract_biome(connection),
        connection.load_url(spatial_filter_url, format="Parquet"),
        resolution=int(row.resolution),
        crs=row.crs,
    )
    result_cube = result_cube.merge_cubes(biome)

    # Submit job
    return result_cube.create_job(
        out_format="NetCDF",
        sample_by_feature=True,
        feature_id_property="id",
        filename_prefix="WAC_Extraction"
    )

