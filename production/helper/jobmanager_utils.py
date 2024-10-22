import pandas as pd
import datetime
import geopandas as gpd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from openeo_gfmap.manager.job_splitters import split_job_hex

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


def calculate_end_date(start_date: str) -> str:
    """Calculate the end date exactly 12 months after the start date.
    
    Parameters
    ----------
    start_date : str
        The start date in the format 'YYYY-MM-DD'.
    
    Returns
    -------
    end_date : str
        The end date, exactly 12 months after the start date, in the format 'YYYY-MM-DD'.
    """
    try:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date must be in the format YYYY-MM-DD")

    # Calculate end date (12 months after start date)
    end_date_dt = start_date_dt + relativedelta(months=12)
    
    # Return end date as string in 'YYYY-MM-DD' format
    return end_date_dt.strftime("%Y-%m-%d")


def create_job_dataframe(
    input_geoms: gpd.GeoDataFrame,
    config: dict,
    max_points: int = 1
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containing all the necessary information to run the job."""
    
    # Define the dataframe columns
    columns = [
        'geometry', 'start_date', 'end_date', 'west', 'east', 'north', 'south', 'crs',
        'executor_memory', 'executor_memoryOverhead', 'python_memory'
    ]

    # Split jobs based on H3 hexagons (adjust as needed for specific grid)
    split_jobs = split_job_hex(input_geoms, max_points=max_points)
    
    
    # Get start and end date for full year
    start_date = config['start_date']
    
    if not start_date:
        raise ValueError("Config must contain a 'start_date' field in the format YYYY-MM-DD")

    # Use the new function to calculate the end date
    end_date = calculate_end_date(start_date)


    #get addiitional values
    executor_memory = config['executor_memory']
    executor_memoryOverhead = config['executor_memoryOverhead']
    python_memory = config['python_memory']
    crs = input_geoms.crs

    # append the information row wise
    rows = []
    for job in split_jobs:
        # Calculate bounding box
        west, south, east, north = job.iloc[0]["geometry"].bounds
        
        # Snap the points to a grid and buffer them
        geometry = job["geometry"] 
        
        # Append job data as a new row
        row_data = [geometry, start_date, end_date, west, east, north, south, crs, executor_memory, 
                    executor_memoryOverhead, python_memory
                   ]
        
        # Create a Series with actual values
        rows.append(pd.Series(dict(zip(columns, row_data))))
    
    # Create the final DataFrame
    return pd.DataFrame(rows)










