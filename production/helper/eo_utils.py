import rasterio
import numpy as np
import pandas as pd
import openeo
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from pyproj import Transformer
from typing import List, Dict, Tuple, Union
from openeo.extra.spectral_indices.spectral_indices import compute_and_rescale_indices
from openeo_gfmap.manager.job_splitters import split_job_s2grid, append_h3_index

# Constants
DEFAULT_PATCH_SIZE = 64
DEFAULT_RESOLUTION = 20.0
PERCENTILE_STATS = ["P10", "P25", "P50", "P75", "P90"]


def compute_temporal_extent(start_date: str, duration_months: int) -> List[str]:
    """
    Calculate temporal extent given a start date and duration in months.
    """
    start = pd.to_datetime(start_date)
    end = start + pd.DateOffset(months=duration_months)
    return [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]


def compute_percentiles(base_features: openeo.DataCube) -> openeo.DataCube:
    """
    Compute percentiles (P10, P25, P50, P75, P90) for a given time dimension.
    """
    def compute_stats(input_timeseries):
        return input_timeseries.quantiles(probabilities=[0.1, 0.25, 0.50, 0.75, 0.9])

    stats = base_features.apply_dimension(
        dimension="t", target_dimension="bands", process=compute_stats
    )
    all_bands = [
        f"{band}_{stat}"
        for band in base_features.metadata.band_names
        for stat in PERCENTILE_STATS
    ]
    return stats.rename_labels("bands", all_bands)

def compute_statistics(base_features):
    """
    Computes  MEAN, STDDEV, MIN, P25, MEDIAN, P75, MAX over a datacube.
    """
    def computeStats(input_timeseries):
        result = openeo.processes.array_concat(
            input_timeseries.mean(),
            input_timeseries.sd()
        )
        result = openeo.processes.array_concat(result, input_timeseries.min())
        result = openeo.processes.array_concat(result, input_timeseries.quantiles(probabilities=[0.25]))
        result = openeo.processes.array_concat(result, input_timeseries.median())
        result = openeo.processes.array_concat(result, input_timeseries.quantiles(probabilities=[0.75]))
        result = openeo.processes.array_concat(result, input_timeseries.max())
        return result
    
    stats = base_features.apply_dimension(dimension='t', target_dimension='bands', process=computeStats)
    all_bands = [band + "_" + stat for band in base_features.metadata.band_names for stat in ["mean", "stddev", "min", "p25", "median", "p75", "max"]]
    return stats.rename_labels('bands', all_bands)


# Patch Creation Functions
def create_aligned_patches(
    polygon: Polygon,
    start_date: str,
    duration_months: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    resolution: float = DEFAULT_RESOLUTION
) -> gpd.GeoDataFrame:
    """
    Generate aligned, non-overlapping patches within a polygon.
    """
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Input polygon must be valid and non-empty.")

    polygon_series = gpd.GeoSeries([polygon], crs="EPSG:4326")
    utm_crs = polygon_series.estimate_utm_crs()
    polygon_series = polygon_series.to_crs(utm_crs)
    distance_m = resolution * patch_size

    # Align polygon bounds to resolution grid
    minx, miny, maxx, maxy = polygon_series.total_bounds
    minx, miny, maxx, maxy = [
        round(coord / resolution) * resolution for coord in (minx, miny, maxx, maxy)
    ]

    # Create grid patches
    x_coords = np.arange(minx, maxx, distance_m)
    y_coords = np.arange(miny, maxy, distance_m)

    patches, centroids_latlon = [], []
    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    for x in x_coords:
        for y in y_coords:
            patch = box(x, y, x + distance_m, y + distance_m)
            if patch.intersects(polygon_series.iloc[0]):
                patches.append(patch)
                centroid = transformer.transform(*patch.centroid.coords[0])
                centroids_latlon.append(centroid)

    # Construct GeoDataFrame
    patches_gdf = gpd.GeoDataFrame(
        {
            "geometry": patches,
            "centroid_lat": [lat for lat, _ in centroids_latlon],
            "centroid_lon": [lon for _, lon in centroids_latlon],
            "temporal_extent": [compute_temporal_extent(start_date, duration_months)] * len(patches),
            "resolution": [resolution] * len(patches),
            "id": [
            f"patch_lat_{str(lat).replace('.', '_')}_lon_{str(lon).replace('.', '_')}" 
            for lat, lon in centroids_latlon
        ],
        },
        crs=utm_crs
    )
    return patches_gdf


def generate_patches_by_crs(
    base_gdf: gpd.GeoDataFrame,
    start_date: str,
    duration_months: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    resolution: float = DEFAULT_RESOLUTION
) -> List[gpd.GeoDataFrame]:
    """
    Create aligned patches for each unique UTM CRS in the base GeoDataFrame.
    """
    patches_by_crs = {}

    for _, row in base_gdf.iterrows():
        patches = create_aligned_patches(
            row.geometry, start_date, duration_months, patch_size, resolution
        )
        crs = patches.crs
        if crs not in patches_by_crs:
            patches_by_crs[crs] = []
        patches_by_crs[crs].append(patches)

    return [
        gpd.GeoDataFrame(pd.concat(patches_list, ignore_index=True), crs=crs)
        for crs, patches_list in patches_by_crs.items()
    ]


# Job Management Functions
def split_jobs_by_s2(
    base_gdf: gpd.GeoDataFrame,
    max_points: int,
    grid_resolution: int = 3
) -> List[gpd.GeoDataFrame]:
    """
    Append H3 indices and split the GeoDataFrame into smaller jobs.
    """
    original_crs = base_gdf.crs
    base_gdf = append_h3_index(base_gdf, grid_resolution=grid_resolution)
    base_gdf = base_gdf.to_crs(original_crs)
    return split_job_s2grid(base_gdf, max_points=max_points)


def create_job_dataframe(split_jobs: List[gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Generate a summary DataFrame for split jobs, including feature counts.
    """
    job_data = []

    for job in split_jobs:
        job_data.append({
            "temporal_extent": job.temporal_extent.iloc[0],
            "geometry": job.to_json(),
            "s2_tile": job.tile.iloc[0] if "tile" in job.columns else None,
            "h3index": job.h3index.iloc[0] if "h3index" in job.columns else None,
            "crs": job.crs.to_string(),
            "resolution": job.resolution.iloc[0],
            "feature_count": len(job),
        })

    return pd.DataFrame(job_data)


def process_split_jobs(
    geodataframes: List[gpd.GeoDataFrame],
    max_points: int,
    grid_resolution: int = 3
) -> List[gpd.GeoDataFrame]:
    """
    Process a list of GeoDataFrames by applying H3 indexing and splitting them.
    """
    all_splits = []
    for gdf in geodataframes:
        all_splits.extend(split_jobs_by_s2(gdf, max_points, grid_resolution))
    return all_splits




