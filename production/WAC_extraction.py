#%%
import openeo
import geopandas as gpd
from pathlib import Path

# Import functions from the cleaned-up module
from helper.eo_utils import (
    generate_patches_by_crs,
    process_split_jobs,
    create_job_dataframe
)

# Filepath to your input shapefile
file_path = Path(r"C:\Git_projects\WAC\production\resources\Land_use_Roads_tile.shp")

# Parameters for processing
patch_size = 64          # Size of patches in pixels
resolution = 10.0         # Alignment resolution in meters
start_date = "2023-01-01" # Temporal extent start date
nb_months = 3             # Number of months for the temporal extent
max_points = 10           # Maximum points per job for splitting
grid_resolution = 3       # H3 index resolution

# Load input data (GeoDataFrame)
base_df = gpd.read_file(file_path)

# Step 1: Generate aligned patches by UTM CRS
dataframes_by_crs = generate_patches_by_crs(
    base_gdf=base_df,
    start_date=start_date,
    duration_months=nb_months,
    patch_size=patch_size,
    resolution=resolution
)

# Step 2: Process the patches into split jobs with H3 indexing
split_jobs = process_split_jobs(
    geodataframes=dataframes_by_crs,
    max_points=max_points,
    grid_resolution=grid_resolution
)

# Step 3: Create a summary DataFrame for the split jobs
job_dataframe = create_job_dataframe(split_jobs)
job_dataframe



#%%

import openeo
from openeo.extra.job_management import MultiBackendJobManager, CsvJobDatabase

# Authenticate and add the backend
connection = openeo.connect(url="openeo.dataspace.copernicus.eu").authenticate_oidc()

# initialize the job manager
manager = MultiBackendJobManager()
manager.add_backend("cdse", connection=connection, parallel_jobs=2)

job_tracker = 'job_tracker.csv'
job_db = CsvJobDatabase(path=job_tracker)
if not job_db.exists():
    df = manager._normalize_df(job_dataframe)
    job_db.persist(df)

#%%
from eo_extractors.extractor import wac_extraction_job

# Run the jobs
manager.run_jobs(start_job=wac_extraction_job, job_db=job_db)


#%%

import xarray as xr

test = xr.open_dataset('C:\Git_projects\WAC\production\job_j-241128b7868942158beedba998e07004\WAC_Extraction_patch_lat_-58_4559412373374_lon_-12_808285009665795.nc')
test