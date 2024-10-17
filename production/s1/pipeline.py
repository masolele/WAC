"""
Extract precalculated monthly S1 composites.

Limitations:
    - only available for 2019-2021.
"""


import geopandas as gpd
import openeo
import openeo.processes as eop
from helper.eo_utils import compute_percentiles
from helper.jobmanager_utils import build_job_options


def start_job(
    row: gpd.GeoDataFrame, connection: openeo.Connection, *args: list, **kwargs: dict
):
    """
    Create a job for the given row.

    :param row: The row containing the job paramters. it needs the following columns:
        - location_id
        - west
        - south
        - east
        - north
        - epsg
        - startdate
        - enddate
        - executor_memory
        - executor_memoryOverhead
        - python_memory
        - export_workspace #TODO not applicable just yet: require to set up WAC STR storage
        - asset_per_band 
    """
    print(f"Starting job for \n{row}")

    # Get the spatial extent
    spatial_extent = {
        "west": int(row.west),
        "south": int(row.south),
        "east": int(row.east),
        "north": int(row.north),
        "crs": "EPSG:" + str(row.epsg),
    }

    temporal_extent = [str(row.startdate), str(row.enddate)]

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

    stats = compute_percentiles(s1_with_ratio)

    save_result_options = {
        "filename_prefix": f"wac-s1-{row.location_id}",
    }
    if "asset_per_band" in row and row.asset_per_band:
        save_result_options["separate_asset_per_band"] = True

    result_datacube = stats.save_result(
        format="GTiff",
        options=save_result_options,
    )

    # Create the job
    job = result_datacube.create_job(
        title=f"WAC S1 - {row.location_id}",
        description=str(row),
        job_options=job_options
    )
    print(
        f"Starting Job: {job.job_id} for \nspatial extent: \n{spatial_extent} \ntemporal extent: \n{temporal_extent} \nwith options \n{job_options}"
    )
    return job


def divide_bands(bands):
    vv = bands[0]
    vh = bands[1]
    return eop.array_append(bands, eop.divide(vh, vv))