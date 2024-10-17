import pandas as pd
import datetime
import geopandas as gpd
from shapely.geometry import Point
import geojson
from typing import List

# Function to conditionally build the job_options dictionary
def build_job_options(row):
    job_options = {}
    
    # Check for 'executor_memory' in the row and add to job_options if present and not null
    if hasattr(row, 'executor_memory') and pd.notna(row.executor_memory):
        job_options["executor-memory"] = row.executor_memory

    # Check for 'executor_memoryOverhead' in the row and add to job_options if present and not null
    if hasattr(row, 'executor_memoryOverhead') and pd.notna(row.executor_memoryOverhead):
        job_options["executor-memoryOverhead"] = row.executor_memoryOverhead

    # Conditionally add 'python_memory' if the field exists and has a valid value
    if hasattr(row, 'python_memory') and pd.notna(row.python_memory):
        job_options["python-memory"] = row.python_memory
    
    return job_options

def calculate_month_difference(start_date_str, end_date_str):
    """
    Calculate the number of months between two dates.
    
    Args:
    start_date_str (str): The start date as a string in "YYYY-MM-DD" format.
    end_date_str (str): The end date as a string in "YYYY-MM-DD" format.

    Returns:
    int: The number of months between the two dates.
    """
    # Convert the date strings to datetime objects
    startdate = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    enddate = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Calculate the year and month difference
    year_diff = enddate.year - startdate.year
    month_diff = enddate.month - startdate.month
    
    # Total number of months
    total_months = year_diff * 12 + month_diff
    
    return total_months

def create_job_dataframe(split_jobs: List[gpd.GeoDataFrame], config: dict) -> pd.DataFrame:
    """Create a dataframe from the split jobs, using config to set job parameters."""
    columns = [
        'location_id', 'west', 'south', 'east', 'north', 'epsg',
        'startdate', 'enddate', 'executor_memory', 'executor_memoryOverhead',
        'python_memory', 'export_workspace', 'asset_per_band'
    ]
    
    rows = []
    for job in split_jobs:
        # 1. Set location_id (using h3index or similar identifier)
        location_id = job.h3index.iloc[0]

        patch = buffer_geometry(job.geometry, int(config['buffer']))        
        # 2. Calculate bounding box for spatial extent (west, south, east, north)
        bounds = patch.total_bounds
        west, south, east, north = bounds[0], bounds[1], bounds[2], bounds[3]

        # Use values from the config
        epsg = patch.crs.to_string()
    
        startdate = config['startdate']
        enddate = config['enddate']
        executor_memory = config['executor_memory']
        executor_memoryOverhead = config['executor_memoryOverhead']
        python_memory = config['python_memory']
        export_workspace = config['export_workspace']
        asset_per_band = config['asset_per_band']

        # Append the row
        rows.append(
            pd.Series(
                dict(zip(columns, [
                    location_id, west, south, east, north, epsg, startdate, enddate,
                    executor_memory, executor_memoryOverhead, python_memory, export_workspace, asset_per_band
                ]))
            )
        )

    return pd.DataFrame(rows)


def buffer_geometry(geometry: geojson.FeatureCollection, buffer: int) -> gpd.GeoDataFrame:
    """
    Buffer the input geometry by a specified distance and round to the nearest 20m on the S2 grid.

    :param geometry: GeoJSON feature collection.
    :param buffer: Buffer distance in meters.
    :return: GeoDataFrame with buffered geometries in UTM CRS.
    """
    gdf = gpd.GeoDataFrame.from_features(geometry).set_crs(epsg=4326)
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    # Round to nearest 20m and apply buffering
    gdf['geometry'] = gdf.centroid.apply(
        lambda point: Point(round(point.x / 20.0) * 20.0, round(point.y / 20.0) * 20.0)
    ).buffer(distance=buffer, cap_style=3)  # cap_style=3 for square-shaped buffer

    return gdf