#%%

import geopandas as gpd
from openeo_gfmap.manager.job_splitters import split_job_hex
from helper.jobmanager_utils import create_job_dataframe
import geopandas as gpd

# Define the file path
file_path = r"C:\Git_projects\WAC\production\resources\32736-random-points.geoparquet"

job_config = {
    "start_date": "2020-01-01",
    "end_date": "2020-02-01",
    "executor_memory": "4G",
    "executor_memoryOverhead": "1G",
    "python_memory": "2G",
    "export_workspace": "False",
    "asset_per_band": "False",
    "buffer": "320"
}

#%%
# Read the GeoParquet file
base_df = gpd.read_parquet(file_path)
if base_df.crs is None:
    base_df.set_crs(epsg=32736, inplace=True)


#Split the jobs per h3 hex #TODO push fix to standard gfmap
split_jobs = split_job_hex(
    base_df, max_points=1
)

# Example usage with split_jobs and optional custom config
job_df = create_job_dataframe(split_jobs, job_config)

job_df

#%%
import openeo
from openeo.extra.job_management import MultiBackendJobManager

from s2.pipeline import start_job

# Generate a unique name for the tracker
job_tracker = 'community_example_job_tracker.parquet'

# Initiate MultiBackendJobManager 
manager = MultiBackendJobManager()  
connection = openeo.connect(url="openeo.dataspace.copernicus.eu").authenticate_oidc()
manager.add_backend("cdse", connection=connection, parallel_jobs=10)

# Run the jobs
manager.run_jobs(df=job_df, start_job=start_job, job_db=job_tracker)

