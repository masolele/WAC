"""
utility.py
-----------
Reusable utilities for:
  1. Interactive selection of bounding box and year.
  2. Interaction with the VITO STAC API to retrieve model IDs and metadata.
"""

import ipyleaflet
import ipywidgets as widgets
from pyproj import CRS, Transformer
import datetime
import requests
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path



# -----------------------------------------------------------------------------
# Interactive configuration setup
# -----------------------------------------------------------------------------
def interactive_configurator(max_size_km: int = 20, default_center=(50.0, 4.0), default_zoom=5):
    """
    Creates an interactive map and year selector to select:
      - spatial_extent
      - crs
      - temporal_extent

    Returns a widget that, when the user draws a box and picks a year,
    displays the selected values and stores them in a return dict.

    Args:
        max_size_km (int): Maximum allowed bounding box size (default: 20 km)
        default_center (tuple): Default map center (lat, lon)
        default_zoom (int): Initial map zoom level

    Returns:
        dict: {"spatial_extent": dict, "crs": str, "temporal_extent": list}
    """

    result = {"spatial_extent": None, "crs": None, "temporal_extent": None}

    # Map widget
    m = ipyleaflet.Map(center=default_center, zoom=default_zoom)
    draw_control = ipyleaflet.DrawControl(
        rectangle={"shapeOptions": {"color": "#0000FF"}},
        polygon={}, circle={}, polyline={}, circlemarker={}, marker={}
    )
    m.add_control(draw_control)

    # Year selection widget
    current_year = datetime.datetime.now().year
    year_picker = widgets.IntSlider(
        value=current_year,
        min=2020,
        max=current_year,
        step=1,
        description="Year:",
        continuous_update=False
    )

    # Output panel
    info_out = widgets.Output()

    # Handle drawing
    def handle_draw(target, action, geo_json):
        info_out.clear_output()
        with info_out:
            if not geo_json:
                print("No geometry drawn.")
                return

            # Coordinates from the drawn rectangle in WGS84 (lon/lat)
            coords = geo_json["geometry"]["coordinates"][0]
            west, south = coords[0]
            east, north = coords[2]

            # Determine correct UTM zone from center longitude
            center_lon = (west + east) / 2
            utm_zone = int((center_lon + 180) / 6) + 1
            crs_code = f"EPSG:{32600 + utm_zone}"

            # Transform WGS84 degrees → UTM meters
            transformer = Transformer.from_crs("EPSG:4326", crs_code, always_xy=True)
            west_m, south_m = transformer.transform(west, south)
            east_m, north_m = transformer.transform(east, north)

            # Compute bbox size in meters
            dx = abs(east_m - west_m)
            dy = abs(north_m - south_m)
            max_km = max(dx, dy) / 1000

            if max_km > max_size_km:
                print(f"Bounding box too large: {max_km:.1f} km (max {max_size_km} km)")
                result["spatial_extent"] = None
                return

            # Year → temporal extent
            year = year_picker.value
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            # Final spatial extent: in meters, consistent with UTM CRS
            spatial_extent = {
                "west": west_m,
                "south": south_m,
                "east": east_m,
                "north": north_m,
                "crs": crs_code
            }

            temporal_extent = [start_date, end_date]

            result.update({
                "spatial_extent": spatial_extent,
                "crs": crs_code,
                "temporal_extent": temporal_extent
            })

            print("Selection complete:")
            print(f"  CRS: {crs_code}")
            print(f"  Spatial extent (in meters): {spatial_extent}")
            print(f"  Temporal extent: {temporal_extent}")
            print(f"  Bounding box size: {max_km:.2f} km")

    draw_control.on_draw(handle_draw)

    ui = widgets.VBox([
        widgets.HTML(f"<h3>🗺️ Draw a Bounding Box (max {max_size_km}×{max_size_km} km)</h3>"),
        m,
        year_picker,
        info_out
    ])

    return ui, result


# -----------------------------------------------------------------------------
# STAC Model Interaction
# -----------------------------------------------------------------------------
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
        "datetime": datetime_range
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
                vmin=0,
                vmax=1,
                add_colorbar=True,
                add_labels=False,
            )
            ax.set_title(var)
            ax.axis("off")

    elif mode == "output":
        classification_var = variables_to_plot[-1]
        class_ids = list(range(len(variables_to_plot) - 1))

        for i, var in enumerate(variables_to_plot):
            data = ds[var].isel(t=time_step)
            ax = axes[i]

            if var == classification_var:
                im = data.plot(
                    ax=ax,
                    cmap="tab20",
                    add_colorbar=True,
                    add_labels=False,
                )
                cbar = im.colorbar
                cbar.set_ticks(class_ids)
                cbar.set_ticklabels(class_ids)
            else:
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

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()