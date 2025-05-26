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
        context (dict): Dictionary containing 'west', 'south', 'east', 'north', and 'crs'.

    Returns:
        xr.DataArray: A new DataArray of shape (2, y, x) with bands ['lon', 'lat'].
    """

    # Parse extent and CRS
    try:
        west  = float(context["west"])
        south = float(context["south"])
        east  = float(context["east"])
        north = float(context["north"])
        crs   = context["crs"]
    except KeyError as e:
        raise ValueError(f"Missing required context key: {e}")

    logger.info(f"Original extent: {west}, {south} → {east}, {north} in {crs}")

    # Transform extent to EPSG:4326 if needed
    if crs != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(west, south)
        east, north = transformer.transform(east, north)
        logger.info(f"Transformed extent to EPSG:4326: {west}, {south} → {east}, {north}")

    # Get cube dimensions
    nx = cube.sizes["x"]
    ny = cube.sizes["y"]

    # Create lon/lat coordinate arrays
    lon = np.linspace(west, east, nx, dtype=np.float32)
    lat = np.linspace(north, south, ny, dtype=np.float32)  # north → south to match image orientation

    # Generate 2D meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    logger.info(f"Longitude range: {lon_grid.min()} to {lon_grid.max()}")
    logger.info(f"Latitude range: {lat_grid.min()} to {lat_grid.max()}")

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