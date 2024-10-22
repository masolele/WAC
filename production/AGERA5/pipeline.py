import ast
import geopandas as gpd
import openeo
from helper.jobmanager_utils import build_job_options

"""
Extract the precipitation and temperature AGERA5 data from a
pre-composited and pre-processed collection. The data is stored in the
CloudFerro S3 stoage, allowing faster access and processing from the CDSE
backend.

Limitations:
    - Only monthly composited data is available.
    - Only two bands are available: precipitation-flux and temperature-mean.
    - This function do not support fetching points or polygons, but only
      tiles. #TODO how so?



"""

def start_job(
    row: gpd.GeoDataFrame, connection: openeo.Connection, *args: list, **kwargs: dict
):
    """
    Create a job for the given row.

    :param row: The row containing the job paramters. it needs the following columns:
        - location_id
        - original job bounds
        - original job crs
        - spatial_extent
        - temporal_extent
        - executor_memory
        - executor_memoryOverhead
        - python_memory
        - export_workspace #TODO not applicable just yet: require to set up WAC STR storage
        - asset_per_band 
    """
    print(f"Starting job for \n{row}")

    # Get the spatial extent
    spatial_extent = ast.literal_eval(row.spatial_extent)
    temporal_extent = ast.literal_eval(row.temporal_extent)

    # Monthly composited METEO data
    cube = connection.load_stac(
        "https://s3.waw3-1.cloudferro.com/swift/v1/agera/stac/collection.json",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["precipitation-flux", "temperature-mean"],
    )

    # cube.result_node().update_arguments(featureflags={"tilesize": 1})
    cube = cube.rename_labels(
        dimension="bands", target=["AGERA5-PRECIP", "AGERA5-TMEAN"]
    )

    
    # build the job options from the dataframe
    job_options = build_job_options(row)

    save_result_options = {
        "filename_prefix": f"wac-DEM-{row.location_id}",
    }
    if "asset_per_band" in row and row.asset_per_band:
        save_result_options["separate_asset_per_band"] = True

    result_datacube = cube.save_result(
        format="GTiff",
        options=save_result_options,
    )

    # Create the job
    job = result_datacube.create_job(
        title=f"WAC AGERA6 - {row.location_id}",
        description=str(row),
        job_options=job_options
    )
    print(
        f"Starting Job: {job.job_id} for \nspatial extent: \n{spatial_extent} \nwith options \n{job_options}"
    )
    return job
