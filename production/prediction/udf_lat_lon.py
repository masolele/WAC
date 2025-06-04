import numpy as np
import xarray as xr
import logging
from pyproj import Transformer

def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = _setup_logging()



def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:

    # Pull each piece of the extent out by name
    west  = float(context.get("west" ))
    south = float(context.get("south"))
    east  = float(context.get("east" ))
    north = float(context.get("north"))
    crs   = context.get("crs")
    if not crs:
        raise ValueError("Missing 'crs' in context")

    logger.info(f"Original extent: {west}, {south} → {east}, {north} in {crs}")

    # Convert extent to EPSG:4326 if needed
    if crs != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(west, south)
        east, north = transformer.transform(east, north)
        logger.info(f"Transformed extent to EPSG:4326: {west}, {south} → {east}, {north}")


 
    # Get pixel counts
    nx = cube.coords["x"].size
    ny = cube.coords["y"].size

    # Build evenly spaced lon/lat along each axis
    lon_array = np.linspace(west, east, nx)
    # Note: lat goes from north down to south so the orientation matches
    lat_array = np.linspace(north, south, ny)

    # Produce 2D grids
    lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)

    logger.info(f"Longitude range: {lon_grid.min()} to {lon_grid.max()}")
    logger.info(f"Latitude range: {lat_grid.min()} to {lat_grid.max()}")

    # build DataArray with the same dims
    da = xr.DataArray(
        np.stack([lon_grid, lat_grid], axis=0),
        dims=("bands", "y", "x"),
        coords={
            "bands": ["lon", "lat"],
            # Here we *overwrite* the x/y coords to be the new geographic grid
            "x": cube.coords['x'],
            "y": cube.coords['y']
        },
    )


    return da