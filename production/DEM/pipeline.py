import geopandas as gpd
import openeo
from helper.jobmanager_utils import build_job_options
import ast

"""
Extract DEM elevation on 30 m resolution. Load precalculated DEM aspect and DEM slope at 10m resolution. The data is stored in the
CloudFerro S3 stoage, allowing faster access and processing from the CDSE
backend.

Limitations:
    - DEM elevation will need to be resampled for merging to 10m resolution.
"""

def start_job(
    row: gpd.GeoDataFrame, connection: openeo.Connection, *args: list, **kwargs: dict
):
    """
    Create a job for the given row.

    :param row: The row containing the job paramters. it needs the following columns:
        'geometry', 'start_date', 'end_date', 'west', 'east', 'north', 'south', 'crs',
        'executor_memory', 'executor_memoryOverhead', 'python_memory'

    """
    spatial_extent = {'west': float(row.west),
                      'east': float(row.east),
                      'north': float(row.north),
                      'south': float(row.south),
                     }
    
    # 1. DEM elevation
    #TODO precompute 10m resolution?
    elevation = connection.load_collection(
        collection_id="COPERNICUS_30",
        spatial_extent=spatial_extent,
        bands=["DEM"]
    )
    
    elevation = elevation.max_time() #eliminate time dimension

    # build the job options from the dataframe
    job_options = build_job_options(row)

    save_result_options = {
        "filename_prefix": f"WAC-DEM",
    }

    result_datacube = elevation.save_result(
        format="GTiff",
        options=save_result_options,
    )

    # Create the job
    job = result_datacube.create_job(
        title=f"WAC DEM",
        description=str(row),
        job_options=job_options
    )
    print(
        f"Starting Job: {job.job_id} for \nspatial extent: \n{spatial_extent} \nwith options \n{job_options}"
    )
    return job
