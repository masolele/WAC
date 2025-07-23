import numpy as np
import xarray as xr
import logging
from pyproj import Transformer
from typing import Dict

# Setup logging
def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(__name__)

logger = _setup_logging()

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Constructs a lon/lat grid as a new DataArray based on the cube's spatial resolution
    and the geographic extent provided in `context`.

    Args:
        cube (xr.DataArray): Input data cube with 'x' and 'y' dimensions.
        context (dict): Dictionary containing ''crs'.

    Returns:
        xr.DataArray: A new DataArray of shape (2, y, x) with bands ['lon', 'lat'].
    """


    crs   = context["crs"]
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    longitudes, latitudes = transformer.transform(cube.x, cube.y)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    logger.info(f"Cube x range: {cube.x.min().values}, {cube.x.max().values}")
    logger.info(f"Cube y range: {cube.y.min().values}, {cube.y.max().values}")

    logger.info(f"Transformed longitudes range: {longitudes.min()}, {longitudes.max()}")
    logger.info(f"Transformed latitudes range: {latitudes.min()}, {latitudes.max()}")

    
    # Build output DataArray
    return xr.DataArray(
        data=np.stack([lon_grid, lat_grid], axis=0),  # shape: (2, y, x)
        dims=("bands", "y", "x"),
        coords={
            "bands": ["lon", "lat"],
            "x": cube.coords["x"],
            "y": cube.coords["y"]
        }
    )