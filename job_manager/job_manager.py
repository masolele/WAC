import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import openeo
from openeo.extra.job_management import MultiBackendJobManager, ParquetJobDatabase
from world_agrocommodities import map_commodities
import world_agrocommodities.classification.config as config
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("job_manager.log"),
    ],
)
logger = logging.getLogger(__name__)


# Create a .env file with the following content:
# OPENEO_AUTH_METHOD="client_credentials"
# OPENEO_AUTH_PROVIDER_ID="CDSE"
# OPENEO_AUTH_CLIENT_ID="openeo-worldcereal-service-account"
# OPENEO_AUTH_CLIENT_SECRET=<your_client_secret_here>

load_dotenv('.env')

### Load and/or create job database
logger.info("Loading job dataframe from parquet...")
job_dataframe = gpd.read_parquet("./job_database_backup.parquet")
logger.info("Job dataframe loaded: %d rows", len(job_dataframe))
job_database = ParquetJobDatabase("./job_database.parquet")
job_database = job_database.initialize_from_df(job_dataframe, on_exists="skip")  # Don't overwrite existing job database!
logger.info("Job database initialized.")

### Create openeo connection
logger.info("Connecting to OpenEO backend...")
connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
# connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc_client_credentials()  # Will take credentials from .env file
logger.info("Authenticated successfully.")

### Define the start_job
def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj

def start_job(row: pd.Series, connection: openeo.Connection, **kwargs) -> openeo.BatchJob:
    logger.info("Starting job for id: %s", row.get("id", "unknown"))
    params = {col: to_python(row[col]) for col in row.index}

    pg = map_commodities(
        connection=connection,
        spatial_extent=params["spatial_extent"],
        temporal_extent=params["temporal_extent"],
        model_id=params["model_id"],
        crs=params["crs"],
    )

    result = pg.save_result(
        format="GTiff",
        filename_prefix=f"wac_terrasphereAOI_{params['id']}_{params['year']}"
    )

    return result.create_job(
        title = f"WAC Terrasphere Job {params['id']} - {params['year']}",
        job_options = config.JOB_OPTIONS,
        auto_add_save_result=False,
    )

# Create the job manager
logger.info("Creating job manager...")
job_manager = MultiBackendJobManager(output_dir = './terrasphere_results/')  
job_manager.add_backend("cdse", connection=connection, parallel_jobs=2)

# Run the job manager
logger.info("Running jobs...")
job_manager.run_jobs(start_job=start_job, job_db=job_database)
logger.info("Job manager finished.")