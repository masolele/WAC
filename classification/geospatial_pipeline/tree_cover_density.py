"""
Module to add LCFM tree cover density to a datacube
"""

import openeo
from typing import Dict, Union, List


def add_tree_cover_density(
    connection: openeo.Connection,
    cube: openeo.DataCube,
    spatial_extent: Dict[str, Union[float, str]],
) -> openeo.DataCube:
    tree_cover_density = connection.load_stac(
        url="https://www.stac.lcfm.dataspace.copernicus.eu/collections/LCFM_TCD-10_CDSE_v100",
        spatial_extent=spatial_extent,
        bands=["MAP"],
    ).min_time()  # Reduce time dimension since there is only one time step

    tree_cover_density = tree_cover_density.rename_labels(
        dimension="bands", target=["Tree_cover_density_2020"]
    )

    output = cube.merge_cubes(
        tree_cover_density
    )  # Resamples tree_cover_density to match cube's grid and resolution, using nearest neighbor by default

    return output
