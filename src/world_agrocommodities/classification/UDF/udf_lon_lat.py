import logging

import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
from pyproj import Transformer


# Setup logging
def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(__name__)


from openeo.udf.udf_data import UdfData

logger = _setup_logging()


def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    cube = udf_data.datacube_list[0]
    arr = cube.get_array().transpose("bands", "y", "x")

    crs = udf_data.proj
    if crs is not None:
        crs = crs["EPSG"]

    logger.info(f"EPSG code determined for feature extraction: {crs}")

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    longitudes, latitudes = transformer.transform(arr.x, arr.y)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    logger.info(f"Transformed longitudes range: {longitudes.min()}, {longitudes.max()}")
    logger.info(f"Transformed latitudes range: {latitudes.min()}, {latitudes.max()}")

    combined = xr.DataArray(
        data=np.stack([lon_grid, lat_grid], axis=0),  # shape: (2, y, x)
        dims=("bands", "y", "x"),
        coords={"bands": ["lon", "lat"], "x": arr.coords["x"], "y": arr.coords["y"]},
    )

    cube_output = XarrayDataCube(combined)
    udf_data.datacube_list = [cube_output]

    return udf_data
