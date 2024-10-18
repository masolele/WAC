#TODO include BAP?

"""
Processes Sentinel-2 data by applying a cloud mask using SCL bands and computes yearly features 
and monthly composites for the given temporal and spatial extent.

"""

import geopandas as gpd
import openeo
from helper.eo_utils import compute_yearly_s2features_and_monthly_s2composites
from helper.jobmanager_utils import build_job_options
from helper.scl_preprocessing import compute_scl_aux
from helper.jobmanager_utils import calculate_month_difference

MAX_CLOUD_COVER = 90

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
    spatial_extent = row.spatial_extent
    temporal_extent = row.temporal_extent

    # build the job options from the dataframe
    job_options = build_job_options(row)

    s2 = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["B02", "B03", "B04", "B08", "B11", "B12"],
        max_cloud_cover=MAX_CLOUD_COVER,
    )
    
    # Step 2: Load the SCL collection for cloud masking
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover=MAX_CLOUD_COVER,
    )
    
    # Step 3: Compute the cloud mask and auxiliary data from SCL
    scl_mask, aux_data = compute_scl_aux(
        scl_datacube=scl,
        erode_r=3,
        dilate_r=13,
        snow_dilate_r=3,
        max_invalid_ratio=0.9,
    )
    
    # Step 4: Apply the cloud mask to the Sentinel-2 data
    s2_masked = s2.mask(scl_mask)
    
    # Step 5: Calculate the number of months between the start and end dates
    nb_of_months = calculate_month_difference(temporal_extent[0], temporal_extent[1])
    
    # Step 6: Compute yearly features and monthly composites
    merged_features = compute_yearly_s2features_and_monthly_s2composites(
        s2_datacube=s2_masked,
        nb_of_months=nb_of_months,
    )
    
    # Step 7: Ensure the output is in int16 range
    result_datacube = merged_features.linear_scale_range(
        -32_766, 32_766, -32_766, 32_766
    )

    # Create the job
    job = result_datacube.create_job(
        title=f"LCFM S2 - {row.location_id}",
        description=str(row),
        job_options=job_options
    )
    print(
        f"Starting Job: {job.job_id} for \nspatial extent: \n{spatial_extent} \ntemporal extent: \n{temporal_extent} \nwith options \n{job_options}"
    )
    return job

