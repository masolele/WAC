"""
Utility functions to interact with the STAC API for retrieving
world-agrocommodities models based on spatial and temporal extents.
"""

import requests
from pyproj import Transformer


def get_wgs84_bbox(bbox):
    """Convert bounding box to WGS84 if needed."""
    if "crs" not in bbox or bbox["crs"].upper() == "EPSG:4326":
        return [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]

    transformer = Transformer.from_crs(bbox["crs"], "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox["west"], bbox["south"])
    maxx, maxy = transformer.transform(bbox["east"], bbox["north"])
    return [minx, miny, maxx, maxy]


def format_datetime_range(dates):
    """Format temporal extent for STAC API."""
    start = f"{dates[0]}T00:00:00Z"
    end = f"{dates[1]}T23:59:59Z"
    return f"{start}/{end}"


def get_model_id(spatial_extent, temporal_extent):
    """
    Query STAC API for model matching a given spatial and temporal extent.
    Returns the first model ID found.
    """
    bbox = get_wgs84_bbox(spatial_extent)
    datetime_range = format_datetime_range(temporal_extent)

    query = {
        "collections": ["world-agri-commodities-models"],
        "bbox": bbox,
        "datetime": datetime_range,
    }

    url = "https://stac.openeo.vito.be/search"
    response = requests.post(url, json=query)

    if not response.ok:
        print("Error:", response.status_code, response.text)
        return None

    data = response.json()
    features = data.get("features", [])
    if not features:
        print("No models found for given extent/time.")
        return None

    stac_id = features[0].get("id")
    print("STAC Item found:")
    print(f"  ID: {stac_id}")
    return stac_id


def get_model_metadata(model_id):
    """Retrieve model metadata from STAC API."""
    url = f"https://stac.openeo.vito.be/collections/world-agri-commodities-models/items/{model_id}"
    response = requests.get(url)

    if not response.ok:
        print("Error:", response.status_code, response.text)
        return None

    item = response.json()
    props = item.get("properties", {})

    metadata = {
        "ModelID": props.get("model_id"),
        "Name": props.get("title"),
        "Region": props.get("region"),
        "Countries Covered": props.get("countries"),
        "Framework": props.get("framework"),
        "Input Shape": props.get("input_shape"),
        "Output Shape": props.get("output_shape"),
        "Input Channels": props.get("input_channels"),
        "Output Classes": props.get("output_classes"),
        "Time of Data begins": props.get("start_datetime"),
        "Time of Data ends": props.get("end_datetime"),
    }

    print("Model metadata retrieved.")
    return metadata
