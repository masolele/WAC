"""
utility.py
-----------
Reusable utilities for:
  1. Interactive selection of bounding box and year.
  2. Interaction with the VITO STAC API to retrieve model IDs and metadata.
"""

import datetime
from pathlib import Path

import ipyleaflet
import ipywidgets as widgets
import matplotlib.pyplot as plt
import requests
import xarray as xr
from pyproj import Transformer
import math


# -----------------------------------------------------------------------------
# Interactive configuration setup
# -----------------------------------------------------------------------------
def interactive_configurator(
    max_size_km: int = 20, default_center=(50.0, 4.0), default_zoom=2
):
    """
    Creates an interactive map with simple coverage visualization.
    """

    result = {"spatial_extent": None, "crs": None, "temporal_extent": None}

    # Map widget
    m = ipyleaflet.Map(center=default_center, zoom=default_zoom)
    m.add_layer(ipyleaflet.basemap_to_tiles(ipyleaflet.basemaps.OpenStreetMap.Mapnik))

    # Store coverage layers
    model_layers = []
    tcd_layers = []

    # Year selection
    current_year = datetime.datetime.now().year
    year_picker = widgets.IntSlider(
        value=current_year,
        min=2020,
        max=current_year,
        step=1,
        description="Year:",
        continuous_update=False,
    )

    # Output panel
    info_out = widgets.Output()
    status_indicator = widgets.HTML(value="<div>🌍 Loading coverage data...</div>")

    # Draw control
    draw_control = ipyleaflet.DrawControl(
        rectangle={"shapeOptions": {"color": "#FF0000", "weight": 3}},
        polygon={},
        circle={},
        polyline={},
        circlemarker={},
        marker={},
    )

    def load_coverage():
        """Load and display coverage areas."""
        nonlocal model_layers, tcd_layers

        # Clear existing
        for layer in model_layers + tcd_layers:
            if layer in m.layers:
                m.remove_layer(layer)

        model_layers = []
        tcd_layers = []

        # Load model coverage
        status_indicator.value = "<div> Loading model coverage...</div>"
        model_features = get_model_coverage()
        if model_features:
            layer = ipyleaflet.GeoJSON(
                data={"type": "FeatureCollection", "features": model_features},
                style={
                    "color": "green",
                    "fillColor": "green",
                    "fillOpacity": 0.3,
                    "weight": 1,
                },
                name="Model Coverage",
            )
            m.add_layer(layer)
            model_layers.append(layer)
            print(f"Loaded {len(model_features)} model coverage areas")

        # Load TCD coverage
        status_indicator.value = "<div> Loading TCD coverage...</div>"
        tcd_features = get_tcd_coverage()
        if tcd_features:
            layer = ipyleaflet.GeoJSON(
                data={"type": "FeatureCollection", "features": tcd_features},
                style={
                    "color": "orange",
                    "fillColor": "orange",
                    "fillOpacity": 0.3,
                    "weight": 1,
                },
                name="TCD Coverage",
            )
            m.add_layer(layer)
            tcd_layers.append(layer)
            print(f" Loaded {len(tcd_features)} TCD coverage areas")

        m.add_control(draw_control)
        status_indicator.value = "<div style='color: green'>✅ Ready - Draw bounding box where coverage overlaps</div>"

    def snap_to_max_size(west, south, east, north, center_lon, max_size_km):
        """Snap the bounding box to maximum allowed size while preserving center."""
        # Simple approach: calculate current center and create new bbox
        center_lat = (south + north) / 2

        # Convert km to degrees (approximate)
        size_deg = max_size_km / 111.0  # 1 degree ≈ 111 km

        half_size_deg = size_deg / 2

        new_west = center_lon - half_size_deg
        new_east = center_lon + half_size_deg
        new_south = center_lat - half_size_deg
        new_north = center_lat + half_size_deg

        return new_west, new_south, new_east, new_north

    def handle_draw(target, action, geo_json):
        info_out.clear_output()
        with info_out:
            if not geo_json:
                return

            coords = geo_json["geometry"]["coordinates"][0]
            west, south = coords[0]
            east, north = coords[2]

            # Check size and snap if needed
            width_km = (
                (east - west) * 111.0 * math.cos(math.radians((south + north) / 2))
            )
            height_km = (north - south) * 111.0
            max_km = max(width_km, height_km)

            was_snapped = False
            if max_km > max_size_km:
                print(f"Bounding box snapped from {max_km:.1f} km to {max_size_km} km")
                center_lon = (west + east) / 2
                west, south, east, north = snap_to_max_size(
                    west, south, east, north, center_lon, max_size_km
                )
                was_snapped = True

            # Get UTM CRS
            center_lon = (west + east) / 2
            utm_zone = int((center_lon + 180) / 6) + 1
            crs_code = f"EPSG:{32600 + utm_zone}"

            # Transform to UTM
            transformer = Transformer.from_crs("EPSG:4326", crs_code, always_xy=True)
            west_m, south_m = transformer.transform(west, south)
            east_m, north_m = transformer.transform(east, north)

            # Set temporal extent
            year = year_picker.value
            temporal_extent = [f"{year}-01-01", f"{year}-12-31"]

            result.update(
                {
                    "spatial_extent": {
                        "west": west_m,
                        "south": south_m,
                        "east": east_m,
                        "north": north_m,
                        "crs": crs_code,
                    },
                    "crs": crs_code,
                    "temporal_extent": temporal_extent,
                }
            )

            print("Selection complete:")
            print(f"  CRS: {crs_code}")
            print(f"  Bounding box: {west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}")
            print(f"  Size: {max_size_km if was_snapped else max_km:.1f} km")
            print(f"  Year: {year}")

            # Check coverage
            check_area_coverage([west, south, east, north], temporal_extent)
            status_indicator.value = (
                "<div style='color: green'>✅ Selection complete</div>"
            )

    def check_area_coverage(bbox_wgs84, temporal_extent):
        """Check coverage in selected area."""
        print("\n Coverage check:")

        model_count = count_stac_items(bbox_wgs84, temporal_extent, "models")
        tcd_count = count_stac_items(bbox_wgs84, temporal_extent, "tcd")

        print(f"  Models: {model_count}")
        print(f"  TCD: {tcd_count}")

        if model_count > 0 and tcd_count > 0:
            print("   Optimal: Both available. Defaulting to TCD 2020")
        elif model_count > 0:
            print("    Only models available")
        elif tcd_count > 0:
            print("    Only TCD available. Defaulting to TCD 2020")
        else:
            print("   No data - try different area")

    draw_control.on_draw(handle_draw)
    load_coverage()

    ui = widgets.VBox(
        [
            widgets.HTML(
                f"<h3> Draw Bounding Box (max {max_size_km}×{max_size_km} km)</h3>"
            ),
            status_indicator,
            widgets.HTML(
                "<div><span style='color: green'>■</span> Models | <span style='color: orange'>■</span> TCD</div>"
            ),
            m,
            year_picker,
            info_out,
        ]
    )

    return ui, result


# -----------------------------------------------------------------------------
# Simple STAC functions
# -----------------------------------------------------------------------------
def get_model_coverage():
    """Get model coverage areas - show actual item geometries."""
    try:
        response = requests.post(
            "https://stac.openeo.vito.be/search",
            json={
                "collections": ["world-agri-commodities-models"],
                "bbox": [-180, -90, 180, 90],
                "datetime": "2020-01-01T00:00:00Z/2024-12-31T23:59:59Z",
                "limit": 100,
            },
            timeout=30,
        )
        if response.ok:
            data = response.json()
            return data.get("features", [])
    except Exception as e:
        print(f"Error loading models: {e}")
    return []


def get_tcd_coverage():
    """Get TCD coverage areas - show actual item geometries."""
    try:
        response = requests.post(
            "https://www.stac.lcfm.dataspace.copernicus.eu/search",
            json={
                "collections": ["LCFM_TCD-10_CDSE_v100"],
                "bbox": [-180, -60, 180, 90],  # Focus on populated areas
                "datetime": "2020-01-01T00:00:00Z/2024-12-31T23:59:59Z",
                "limit": 200,
            },
            timeout=30,
        )
        if response.ok:
            data = response.json()
            return data.get("features", [])
    except Exception as e:
        print(f"Error loading TCD: {e}")
    return []


def count_stac_items(bbox_wgs84, temporal_extent, item_type):
    """Count items in area - matching the working approach from original code."""
    if item_type == "models":
        url = "https://stac.openeo.vito.be/search"
        collections = ["world-agri-commodities-models"]
        datetime_range = (
            f"{temporal_extent[0]}T00:00:00Z/{temporal_extent[1]}T23:59:59Z"
        )
    else:
        url = "https://www.stac.lcfm.dataspace.copernicus.eu/search"
        collections = ["LCFM_TCD-10_CDSE_v100"]
        datetime_range = f"2020-01-01T00:00:00Z/2020-12-31T23:59:59Z"

    # Match the exact query format from your working code
    query = {
        "collections": collections,
        "bbox": bbox_wgs84,
        "datetime": datetime_range,
        # No limit parameter - same as working version
    }

    try:
        response = requests.post(url, json=query, timeout=10)
        if response.ok:
            data = response.json()
            features = data.get("features", [])
            count = len(features)
            print(f"    Response: {count} features found")

            # Also show the first feature ID if available
            if features:
                print(f"    First feature ID: {features[0].get('id')}")

            return count
        else:
            print(f"    Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"    Exception: {e}")

    return 0


# Keep essential existing functions
def get_wgs84_bbox(bbox):
    """Convert bounding box to WGS84 if needed."""
    if "crs" not in bbox or bbox["crs"].upper() == "EPSG:4326":
        return [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]

    transformer = Transformer.from_crs(bbox["crs"], "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox["west"], bbox["south"])
    maxx, maxy = transformer.transform(bbox["east"], bbox["north"])
    return [minx, miny, maxx, maxy]


def get_model_id(spatial_extent, temporal_extent):
    """Get model ID for selected area."""
    bbox = get_wgs84_bbox(spatial_extent)
    datetime_range = f"{temporal_extent[0]}T00:00:00Z/{temporal_extent[1]}T23:59:59Z"

    response = requests.post(
        "https://stac.openeo.vito.be/search",
        json={
            "collections": ["world-agri-commodities-models"],
            "bbox": bbox,
            "datetime": datetime_range,
            "limit": 1,
        },
    )

    if response.ok:
        data = response.json()
        features = data.get("features", [])
        if features:
            return features[0].get("id")

    return None


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


# =============================================================================
# Plotting OpenEO Job Results
# =============================================================================
def plot_job_results(
    job,
    output_dir="results",
    time_step=0,
    ncols=4,
    mode="input",  # must be "input" or "output"
    save_path=None,
):
    """
    Download and visualize OpenEO job results.

    Supports two modes:
      - "input": Continuous data (e.g., spectral bands) scaled 0–1.
      - "output": Classification-style results with discrete color map.

    Args:
        job: OpenEO job object (must have .job_id and .get_results()).
        output_dir (str | Path): Directory where results will be downloaded.
        time_step (int): Time index to visualize.
        ncols (int): Number of columns in subplot grid.
        mode (str): Either "input" or "output".
        save_path (str | Path | None): Optional path to save the figure.
    """
    if mode not in {"input", "output"}:
        raise ValueError("mode must be either 'input' or 'output'")

    output_dir = Path(output_dir).expanduser().resolve() / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading job results to: {output_dir}")
    result_paths = job.get_results().download_files(str(output_dir))
    if not result_paths:
        print("No files downloaded from job.")
        return

    ds = xr.open_dataset(result_paths[0])

    variables_to_plot = [
        var for var in ds.data_vars if {"t", "y", "x"}.issubset(ds[var].dims)
    ]
    if not variables_to_plot:
        print("No (t, y, x) variables found in dataset.")
        return

    n = len(variables_to_plot)
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    print(f"Plotting mode: {mode}")

    if mode == "input":
        for i, var in enumerate(variables_to_plot):
            data = ds[var].isel(t=time_step)
            ax = axes[i]
            im = data.plot(
                ax=ax,
                cmap="viridis",
                add_colorbar=True,
                add_labels=False,
            )
            ax.set_title(var)
            ax.axis("off")

    elif mode == "output":
        classification_var = "ARGMAX"
        tree_cover_var = "Tree_cover_density_2020"

        for i, var in enumerate(variables_to_plot):
            if var not in ds:
                continue

            data = ds[var].isel(t=time_step)
            ax = axes[i]

            if var == classification_var:
                # Discrete classification map
                im = data.plot(
                    ax=ax,
                    cmap="tab20",
                    add_colorbar=True,
                    add_labels=False,
                )
                cbar = im.colorbar
                cbar.set_label("Class ID")
            elif var == tree_cover_var:
                # Continuous 0–100 map
                im = data.plot(
                    ax=ax,
                    cmap="YlGn",
                    vmin=0,
                    vmax=100,
                    add_colorbar=True,
                    add_labels=False,
                )
                cbar = im.colorbar
                cbar.set_label("Tree Cover Density (%)")
            else:
                # Probability or continuous confidence-like layers (0–1)
                im = data.plot(
                    ax=ax,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    add_colorbar=True,
                    add_labels=False,
                )

            ax.set_title(var)
            ax.axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
