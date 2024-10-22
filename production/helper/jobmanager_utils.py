import pandas as pd
import datetime
import geopandas as gpd


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
    
    
    # Use values from the config for time and memory settings
    start_date = config['start_date']
    end_date = config['end_date']
    executor_memory = config['executor_memory']
    executor_memoryOverhead = config['executor_memoryOverhead']
    python_memory = config['python_memory']
    crs = input_geoms.crs

    # create patches for each geometry and append them to the dataframe
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







