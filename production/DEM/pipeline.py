import geopandas as gpd
import openeo
from helper.jobmanager_utils import build_job_options

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

    # 1. DEM elevation
    #TODO precompute 10m resolution?
    elevation = connection.load_collection(
        collection_id="COPERNICUS_30",
        spatial_extent=spatial_extent,
        bands=["DEM"]
    )
    
    elevation = elevation.max_time() #eliminate time dimension

    # 2. DEM Slope
    slope = connection.load_stac(
        url="https://stac.openeo.vito.be/collections/DEM_slope_10m",
        spatial_extent=spatial_extent,
        bands=["SLP10"],
    )
    slope.metadata = slope.metadata.add_dimension("t", label=None, type="temporal")
    slope = slope.rename_labels("bands", ["SLOPE"])
    slope = slope.max_time()

    # 3. DEM Aspect
    aspect = connection.load_stac(
        url="https://stac.openeo.vito.be/collections/DEM_aspec_10m",
        spatial_extent=spatial_extent,
        bands=["ASP10"],
    )
    aspect.metadata = aspect.metadata.add_dimension("t", label=None, type="temporal")
    aspect = aspect.rename_labels("bands", ["ASPECT"])
    aspect = aspect.max_time()

    DEM_cube = slope.merge_cubes(aspect)
    DEM_cube = DEM_cube.merge_cubes(elevation)
    
    # build the job options from the dataframe
    job_options = build_job_options(row)

    save_result_options = {
        "filename_prefix": f"wac-DEM-{row.location_id}",
    }
    if "asset_per_band" in row and row.asset_per_band:
        save_result_options["separate_asset_per_band"] = True

    result_datacube = DEM_cube.save_result(
        format="GTiff",
        options=save_result_options,
    )

    # Create the job
    job = result_datacube.create_job(
        title=f"WAC DEM - {row.location_id}",
        description=str(row),
        job_options=job_options
    )
    print(
        f"Starting Job: {job.job_id} for \nspatial extent: \n{spatial_extent} \nwith options \n{job_options}"
    )
    return job
