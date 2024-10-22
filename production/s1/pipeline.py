import ast
import geopandas as gpd
import openeo
import openeo.processes as eop
from helper.eo_utils import compute_percentiles
from helper.jobmanager_utils import build_job_options

"""
Extract precalculated monthly S1 composites.

Limitations:
    - only available for 2019-2021.
"""


def start_job(
    row: gpd.GeoDataFrame, connection: openeo.Connection, *args: list, **kwargs: dict
):
    """
    Create a job for the given row.

    :param row: The row containing the job paramters. it needs the following columns:
        - geometry
        - temporal_extent
        - original_extent
        - executor_memory
        - executor_memoryOverhead
        - python_memory
    """

    # Get the spatial extent
    spatial_extent = {'west': float(row.west),
                      'east': float(row.east),
                      'north': float(row.north),
                      'south': float(row.south),
                     }
    
    
    temporal_extent = [str(row.start_date), str(row.end_date)]

    # build the job options from the dataframe
    job_options = build_job_options(row)

    s1 = connection.load_collection(
        "SENTINEL1_GLOBAL_MOSAICS",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["VV", "VH"],
    )

    s1_with_ratio = s1.apply_dimension(
        process=divide_bands, dimension="bands"
    ).rename_labels(dimension="bands", target=["VV", "VH", "VH/VV"])

    result_datacube = compute_percentiles(s1_with_ratio)


    save_result_options = {
        # TODO change the filename_prefix to the correct format, extra variables can be added in the job_db and used here
        "filename_prefix": f"WAC_S1",
    }

    save_datacube = result_datacube.save_result(
        format="netCDF",
        options=save_result_options,
    )
    
    # Create the job
    job = save_datacube.create_job(
        title=f"LCFM S1 - {row.id}",
        description=str(row),
        job_options=job_options
    )

    print(
        f"Starting Job: {job.job_id} for \nspatial extent: \n{spatial_extent} \ntemporal extent: \n{temporal_extent} \nwith options \n{job_options}"
    )


def divide_bands(bands):
    vv = bands[0]
    vh = bands[1]
    return eop.array_append(bands, eop.divide(vh, vv))