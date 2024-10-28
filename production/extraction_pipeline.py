#%%
import openeo
import geopandas as gpd
from openeo.extra.job_management import MultiBackendJobManager
from helper.jobmanager_utils import create_job_dataframe

# Define the file path
file_path = r"C:\Git_projects\WAC\production\resources\Land_use_Roads_tile.shp"

# Example of how to use it with a config
config = {
    'start_date': '2020-01-01',
    'executor_memory': '2GB',
    'executor_memoryOverhead': '1GB',
    'python_memory': '1GB'
}

base_df = gpd.read_file(file_path) 
job_df = create_job_dataframe(base_df, config)
job_df = job_df.iloc[0:1]
job_df


#%%
from AGERA5.pipeline import start_job

# Generate a unique name for the tracker
job_tracker = 'example_job_tracker.csv'

# Initiate MultiBackendJobManager 
manager = MultiBackendJobManager()  
connection = openeo.connect(url="openeo.dataspace.copernicus.eu").authenticate_oidc()
manager.add_backend("cdse", connection=connection, parallel_jobs=10)

# Run the jobs
manager.run_jobs(df=job_df, start_job=start_job, job_db=job_tracker)
