import openeo
from typing import Dict, List
from eo_utils import compute_percentiles, divide_bands, compute_yearly_features_and_monthly_composites
from scl_preprocessing import compute_scl_aux
from utils import calculate_month_difference


def process_sentinel1_mosaic(connection: openeo.Connection, 
                             spatial_extent: Dict[str, float], 
                             temporal_extent: List[str]) -> openeo.rest.datacube.DataCube:
    """
    Processes Sentinel-1 mosaic data by applying a band ratio (VH/VV) and computing percentiles.
    
    Args:
    connection (openeo.Connection): An authenticated OpenEO connection.
    spatial_extent (dict): The spatial extent as a dictionary (bounding box).
    temporal_extent (list): The temporal extent as a list (start and end date as strings).
    
    Returns:
    DataCube: An OpenEO DataCube with the computed percentiles.
    """
    # Step 1: Load the Sentinel-1 Global Mosaics collection
    s1 = connection.load_collection(
        "SENTINEL1_GLOBAL_MOSAICS",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["VV", "VH"]
    )

    # Step 2: Apply band ratio (VH/VV)
    s1_with_ratio = s1.apply_dimension(
        process=divide_bands, dimension="bands"
    ).rename_labels(dimension="bands", target=["VV", "VH", "VH/VV"])

    # Step 3: Compute percentiles
    output = compute_percentiles(s1_with_ratio)
    
    # Return the processed DataCube
    return output



def process_sentinel2_data(connection: openeo.Connection, 
                           spatial_extent: Dict, 
                           temporal_extent: List[str], 
                           max_cloud_cover: float) -> openeo.rest.datacube.DataCube:
    """
    Processes Sentinel-2 data by applying a cloud mask using SCL bands and computes yearly features 
    and monthly composites for the given temporal and spatial extent.
    
    Args:
    connection (openeo.Connection): An authenticated OpenEO connection.
    spatial_extent (Dict): The spatial extent as a dictionary (GeoJSON or bounding box).
    temporal_extent (List[str]): A list containing the start and end date as strings in 'YYYY-MM-DD' format.
    max_cloud_cover (float): The maximum allowed cloud cover percentage for filtering the Sentinel-2 data.
    
    Returns:
    openeo.rest.datacube.DataCube: A DataCube with yearly features and monthly composites.
    """
    # Step 1: Load the Sentinel-2 collection (optical bands)
    s2 = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["B02", "B03", "B04", "B08", "B11", "B12"],
        max_cloud_cover=max_cloud_cover,
    )
    
    # Step 2: Load the SCL collection for cloud masking
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover=max_cloud_cover,
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
    merged_features = compute_yearly_features_and_monthly_composites(
        s2_datacube=s2_masked,
        nb_of_months=nb_of_months,
    )
    
    # Step 7: Ensure the output is in int16 range
    merged_features = merged_features.linear_scale_range(
        -32_766, 32_766, -32_766, 32_766
    )
    
    return merged_features