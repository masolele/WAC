
from typing import Dict, Any
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer
import shapely
import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any


# spatial
def create_spatial_extent(offset_x: int, offset_y: int, base_extent: Dict[str, Any], spatial_window_size: int, spatial_window_gap: int) -> Dict[str, Any]:
    """Create a spatial extent dictionary based on an offset for 10 km by 10 km windows."""
    return {
        "west": base_extent["west"] + offset_x * (spatial_window_size + spatial_window_gap),
        "south": base_extent["south"] + offset_y * (spatial_window_size + spatial_window_gap),
        "east": base_extent["west"] + (offset_x + 1) * spatial_window_size + offset_x * spatial_window_gap,
        "north": base_extent["south"] + (offset_y + 1) * spatial_window_size + offset_y * spatial_window_gap,
        "crs": base_extent["crs"],
        "srs": base_extent["srs"]
    }


def create_polygon(extent: Dict[str, Any]) -> Dict[str, Any]:
    """Create a polygon from the spatial extent returned by create_spatial_extent."""
    
    # Extract coordinates from extent dictionary
    west = extent["west"]
    south = extent["south"]
    east = extent["east"]
    north = extent["north"]

    # Create a polygon from the corner coordinates
    polygon = Polygon([
        (west, south),
        (west, north),
        (east, north),
        (east, south),
        (west, south)
    ])


    # Convert to EPSG:4326 (WGS 84) if necessary
    if extent["crs"] != "EPSG:4326":
        transformer = Transformer.from_crs(extent["crs"], "EPSG:4326", always_xy=True).transform
        polygon = transform(transformer, polygon)

    # Return the geojson representation of the polygon and its areas
    return shapely.to_geojson(polygon)


#temporal

def create_temporal_extent(start_date_str: str, nb_months:int):
    """
    Create a temporal extent by adding months to the start date and adjusting for invalid dates.
    
    Args:
    start_date_str (str): The start date as a string in "YYYY-MM-DD" format.
    nb_months (int): The number of months to add.

    Returns:
    list: A list with the start date and end date as strings in "YYYY-MM-DD" format.
    """
    # Convert the start date string to a datetime object
    startdate = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    
    # Add the number of months using relativedelta
    enddate = startdate + relativedelta(months=nb_months)
    
    # Convert the datetime objects back to strings
    return [startdate.strftime("%Y-%m-%d"), enddate.strftime("%Y-%m-%d")]

def calculate_month_difference(start_date_str, end_date_str):
    """
    Calculate the number of months between two dates.
    
    Args:
    start_date_str (str): The start date as a string in "YYYY-MM-DD" format.
    end_date_str (str): The end date as a string in "YYYY-MM-DD" format.

    Returns:
    int: The number of months between the two dates.
    """
    # Convert the date strings to datetime objects
    startdate = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    enddate = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Calculate the year and month difference
    year_diff = enddate.year - startdate.year
    month_diff = enddate.month - startdate.month
    
    # Total number of months
    total_months = year_diff * 12 + month_diff
    
    return total_months