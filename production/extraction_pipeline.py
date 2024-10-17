#%%

import geopandas as gpd
from openeo_gfmap.manager.job_splitters import split_job_hex
from helper.jobmanager_utils import create_job_dataframe
import geopandas as gpd

# Define the file path
file_path = r"C:\Git_projects\WAC\production\resources\32736-random-points.geoparquet"

job_config = {
    "startdate": "2020-01-01",
    "enddate": "2021-01-01",
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
    # Assuming the data is in EPSG:4326 (WGS 84), assign it
    base_df.set_crs(epsg=32736, inplace=True)


#split the jobs per h3 hex #TODO push fix to standard gfmap
split_jobs = split_job_hex(
    base_df, max_points=1
)

# Example usage with split_jobs and optional custom config
job_df = create_job_dataframe(split_jobs, job_config)

job_df

