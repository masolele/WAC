import os
import requests
import pandas as pd
import datetime
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tempfile import NamedTemporaryFile
from typing import List
from openeo_gfmap.manager.job_splitters import split_job_hex


def build_job_options(row) -> dict:
    """Build job options from a DataFrame row."""
    job_options = {}
    
    # Helper function to add options if they exist and are not null
    def add_option(option_name: str, value):
        if pd.notna(value):
            job_options[option_name] = value

    # Check for memory options in the row
    add_option("executor-memory", getattr(row, 'executor_memory', None))
    add_option("executor-memoryOverhead", getattr(row, 'executor_memoryOverhead', None))
    add_option("python-memory", getattr(row, 'python_memory', None))
    
    return job_options


def calculate_month_difference(start_date_str: str, end_date_str: str) -> int:
    """Calculate the number of months between two dates."""
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

    # Total number of months
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

def create_geographic_patch(center: Point, patch_size: float) -> Polygon:
    """Create a square patch centered at a specified point."""
    half_size = patch_size / 2.0
    return Polygon([
        (center.x - half_size, center.y - half_size),
        (center.x + half_size, center.y - half_size),
        (center.x + half_size, center.y + half_size),
        (center.x - half_size, center.y + half_size),
    ])

def buffer_geometry(geometry: gpd.GeoDataFrame, buffer_distance: int, rounding_resolution: int = 20) -> gpd.GeoDataFrame:
    """Buffer the input geometry by a specified distance and round to align with S2 tile boundaries."""
    
    # Ensure the CRS is set, default to EPSG:4326 if not set
    if geometry.crs is None:
        geometry.set_crs(epsg=4326, inplace=True)

    # Reproject to UTM for accurate distance calculations
    utm_crs = geometry.estimate_utm_crs()
    geometry = geometry.to_crs(utm_crs)

    # Buffer the entire geometry (polygon) instead of centroids
    geometry['geometry'] = geometry['geometry'].buffer(distance=buffer_distance, cap_style=3)

    # Optionally, round the coordinates of the resulting buffered polygon
    geometry['geometry'] = geometry['geometry'].apply(
        lambda geom: geom.simplify(rounding_resolution, preserve_topology=True)
    )

    return geometry


def create_non_overlapping_patches_with_buffer(job_geometry: gpd.GeoSeries, patch_size: float) -> pd.DataFrame:
    """Create non-overlapping patches from the given job geometry with a buffer."""
    
    # Buffer the geometry by half the patch size to ensure patches align
    buffered_geom = buffer_geometry(gpd.GeoDataFrame(geometry=job_geometry), buffer_distance=patch_size / 2).geometry.values[0]

    # Get the bounds of the polygon after buffering
    minx, miny, maxx, maxy = buffered_geom.bounds
    
    patches_data = []

    # Create patches that cover the entire polygon, not just the bounding box
    x_steps = int((maxx - minx) // patch_size) + 1  # Added +1 to ensure full coverage
    y_steps = int((maxy - miny) // patch_size) + 1  # Added +1 to ensure full coverage

    for i in range(x_steps):
        for j in range(y_steps):
            lower_left_x = minx + (i * patch_size)
            lower_left_y = miny + (j * patch_size)
            patch_center = Point(lower_left_x + (patch_size / 2.0), lower_left_y + (patch_size / 2.0))
            patch = create_geographic_patch(patch_center, patch_size)

            # Add patch if it intersects the buffered geometry (polygon)
            if patch.intersects(buffered_geom):
                patches_data.append({
                    'geometry': patch,
                    'center': patch_center,
                    'bounds': patch.bounds,  # Optional: Add bounds or other properties as needed
                    'crs': job_geometry.crs
                })

    # Create a DataFrame from the list of patch data
    return pd.DataFrame(patches_data)



def create_job_dataframe(input_geoms: gpd.GeoDataFrame, config: dict) -> pd.DataFrame:
    """Create a DataFrame from the split jobs with necessary job information."""

    all_patches_data = []
    for _, job in input_geoms.iterrows():
        job_geometry = gpd.GeoSeries([job["geometry"]], crs=input_geoms.crs)  # Create a GeoSeries from the geometry
        patch_size = config.get('patch_size', 64) * config.get('pixel_size', 10)  # Convert patch size to meters

        patches_df = create_non_overlapping_patches_with_buffer(
            job_geometry,
            patch_size,  # Convert to meters
        )

        # Add necessary job information to each patch row
        for _, patch_row in patches_df.iterrows():
            row_data = {
                'geometry': patch_row['geometry'],
                'patch_centers': patch_row['center'],  # Center of the patch
                'start_date': config['start_date'],
                'end_date': config['end_date'],
                'west': patch_row['bounds'][0],
                'east': patch_row['bounds'][2],
                'north': patch_row['bounds'][3],
                'south': patch_row['bounds'][1],
                'crs': patch_row['crs'],
                'executor_memory': config['executor_memory'],
                'executor_memoryOverhead': config['executor_memoryOverhead'],
                'python_memory': config['python_memory'],
                'export_workspace': config['export_workspace'],
                'asset_per_band': config['asset_per_band']
            }
            all_patches_data.append(row_data)

    # Convert list of dictionaries to DataFrame
    return pd.DataFrame(all_patches_data)



def split_job_s2grid(polygons: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
    split_datasets = []
    for _, sub_gdf in polygons.groupby("tile"):
        split_datasets.append(sub_gdf.reset_index(drop=True))
    return split_datasets


def upload_geoparquet_artifactory(gdf: gpd.GeoDataFrame, row_id: int) -> str:
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile(delete=False, suffix=".parquet")
    temporary_file.close()
    gdf.to_parquet(temporary_file.name)
    temporary_file.delete = True

    ARTIFACTORY_USERNAME = os.getenv("ARTIFACTORY_USERNAME")
    ARTIFACTORY_PASSWORD = os.getenv("ARTIFACTORY_PASSWORD")
    if not ARTIFACTORY_USERNAME or not ARTIFACTORY_PASSWORD:
        raise ValueError(
            "Please set the ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD environment variables"
        )

    headers = {"Content-Type": "application/octet-stream"}

    upload_url = f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/WAC/patch_geom_{row_id}.parquet"

    with open(temporary_file.name, "rb") as f:
        response = requests.put(
            upload_url,
            headers=headers,
            data=f,
            auth=(ARTIFACTORY_USERNAME, ARTIFACTORY_PASSWORD),
        )

    assert (
        response.status_code == 201
    ), f"Error uploading the dataframe to artifactory: {response.text}"
    return upload_url


