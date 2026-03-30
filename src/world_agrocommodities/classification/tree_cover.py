"""
Module to add LCFM tree cover to a datacube
"""

import openeo
from typing import Dict, Union, List


def add_tree_cover(
    connection: openeo.Connection,
    cube: openeo.DataCube,
    spatial_extent: Dict[str, Union[float, str]],
) -> openeo.DataCube:
    tree_cover = connection.load_stac(
        url="https://stac.terrascope.be/collections/lcfm-lcm-10/",
        spatial_extent=spatial_extent,
        bands=["MAP"],
    ).min_time()  # Reduce time dimension since there is only one time step

    tree_cover = tree_cover.rename_labels(dimension="bands", target=["Tree_cover_2020"])

    tree_cover = tree_cover.apply(
        lambda x: x == 10
    )  # class 10 corresponds to tree cover in the LCFM classification

    output = cube.merge_cubes(
        tree_cover
    )  # Resamples tree_cover to match cube's grid and resolution, using nearest neighbor by default

    return output
