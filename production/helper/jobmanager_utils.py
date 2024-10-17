import pandas as pd

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