"""
Module that combines the entire processing pipeline for geospatial data classification using OpenEO.
"""

import world_agrocommodities.classification.config as config
import openeo

from openeo.api.process import Parameter

from typing import Dict, Union, List

from world_agrocommodities.classification import (
    add_tree_cover_density,
    normalize_cube,
    load_input_cube,
    run_inference,
)


def map_commodities(
    connection: openeo.Connection,
    spatial_extent: Union[Dict[str, Union[float, str]], Parameter],
    temporal_extent: Union[List[str], Parameter],
    model_id: Union[
        str, Parameter
    ],  # TODO: currently this is the ID. Should we change this to the model URL instead?
    crs: Union[int, str, Parameter],
    max_cloud_cover: int = config.MAX_CLOUD_COVER,
    quantile: float = config.QUANTILE,
    resolution: float = config.RESOLUTION,
    patch_size: int = config.PATCH_SIZE,
    overlap: int = config.OVERLAP_SIZE,
) -> openeo.DataCube:
    """Main function to create an openEO proces graph for the full inference pipeline.

    Parameters
    ----------
    connection : openeo.Connection
        connection to openEO backend
    spatial_extent : Union[Dict[str, Union[float, str]], Parameter]
        Spatial extent over which to run the inference
    temporal_extent : Union[List[str], Parameter]
        Temporal extent over which to run the inference
    model_id : Union[str, Parameter]
        ID of the inference ONNX model
    crs : Union[int, str, Parameter]
        EPSG code of the projection you want in the output
    max_cloud_cover : int, optional
        Max cloud cover percentage in sentinel_2 loading, by default config.MAX_CLOUD_COVER
    quantile : float, optional
        Quantile used in temporal aggregation, by default config.QUANTILE
    resolution : float, optional
        Resolution of the output cube, by default config.RESOLUTION
    patch_size : int, optional
        Size of the patch over which to run the inference, by default config.PATCH_SIZE
    overlap : int, optional
        Overlap of the patches over which inference is ran, by default config.OVERLAP_SIZE

    Returns
    -------
    openeo.DataCube
        _description_
    """

    # Load and preprocess data
    cube = load_input_cube(
        conn=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        max_cloud_cover=max_cloud_cover,
        quantile=quantile,
        resolution=resolution,
        crs=crs,
    )

    # Run inference
    inference_cube = run_inference(
        cube,
        model_id=model_id,
        patch_size=patch_size,
        overlap=overlap,
    )

    # Add tree cover density
    output_cube = add_tree_cover_density(
        connection=connection,
        cube=inference_cube,
        spatial_extent=spatial_extent,
        crs=crs,
    )

    return output_cube
