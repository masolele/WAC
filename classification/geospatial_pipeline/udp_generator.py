#%% pipeline_builder.py
from pathlib import Path
from typing import Callable
import json

from openeo import Connection, DataCube
from openeo.api.process import Parameter 
from openeo.rest.udp import build_process_dict

import config as config
from geospatial_pipeline.input_cube_loader import load_input_cube
from geospatial_pipeline.onnx_inference import run_inference
from geospatial_pipeline.band_normalization import normalize_cube

SPATIAL_PARAM = Parameter.spatial_extent(name="spatial_extent", description= 'spatial extent including crs', default=config.SPATIAL_EXTENT)
TEMPORAL_PARAM = Parameter.temporal_interval(name="temporal_extent", default=config.TEMPORAL_EXTENT)
#%%
def create_classification_cube(conn: Connection) -> DataCube:
    """
    Load data, preprocess, and run inference in a single pipeline.

    Args:
        conn (Connection): OpenEO connection.

    Returns:
        DataCube: Inference results cube with renamed class probability bands.
    """

    # Load and preprocess data
    cube = load_input_cube(
        conn,
        spatial_extent=SPATIAL_PARAM,
        temporal_extent=TEMPORAL_PARAM,
        max_cloud_cover=config.MAX_CLOUD_COVER,
        quantile = config.QUANTILE        
    )

    #UDF based normalisation
    cube_normalised = normalize_cube(cube)

    # Run inference
    inference_cube = run_inference(
        cube_normalised,
        model_name=config.MODEL_NAME,
        patch_size=config.PATCH_SIZE,
        overlap=config.OVERLAP_SIZE
    )

    # Rename bands to probability labels
    class_labels = [config.CLASS_MAPPING[i] for i in sorted(config.CLASS_MAPPING)] + ["ARGMAX"]
    return  inference_cube.rename_labels(dimension='bands', target=class_labels)


def generate_udp(conn: Connection,
                build_cube_fn: Callable[[Connection], DataCube],
                process_id: str,
                summary: str,
                output_dir: Path,
    ) -> dict:
    """
    Generate an openEO UDP (User Defined Process) JSON dict using given pipeline.

    Args:
        conn: OpenEO connection object.
        build_pipeline_fn: Function accepting an openEO Connection, returning a DataCube.
        process_id: Unique identifier for the UDP.
        summary: Short summary of the process.
        output_dir: Directory where the JSON file will be saved.

    Returns:
        dict: The complete UDP process dictionary.
    """
    
    # Connect and build the data cube (could be just to get the process graph)
    datacube = build_cube_fn(conn)

    # Build the UDP process dictionary
    udp_dict = build_process_dict(
        process_graph=datacube,
        process_id=process_id,
        summary=summary,
        parameters=[SPATIAL_PARAM, TEMPORAL_PARAM],
        default_job_options=config.JOB_OPTIONS,
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the UDP JSON file
    output_path = output_dir / f"{process_id}.json"
    with open(output_path, "w") as f:
        json.dump(udp_dict, f, indent=2)

    print(f"Saved UDP JSON to {output_path}")
    return udp_dict

#%%
import openeo
conn = openeo.connect("https://openeo.dataspace.copernicus.eu/")
conn.authenticate_oidc()

cube = create_classification_cube(
        conn
    )

UDPdir = Path('C:/Git_projects/WAC/classification/UDP/')

generate_udp(conn=conn,
    build_cube_fn = create_classification_cube,  # <-- pass function, not result
    process_id =  'wac_inference_africa',
    summary = 'wac_inference_africa',
    output_dir = UDPdir)

#%%

import openeo
conn = openeo.connect("https://openeo.dataspace.copernicus.eu/")
conn.authenticate_oidc()

process_graph_file = "C:\\Git_projects\\WAC\\classification\\UDP\\wac_inference_africa.json"

with open(process_graph_file, 'r') as f:
    process_graph  = json.load(f)

# Define the spatial extent
spatial_extent = config.SPATIAL_EXTENT
temporal_extent = config.TEMPORAL_EXTENT

cube = conn.datacube_from_json(process_graph_file,
                                    parameters = {'spatial_extent': spatial_extent,
                                                  'temporal_extent': temporal_extent
                                                  })
job = cube.create_job(title="test", additional=config.JOB_OPTIONS)
job.start_and_wait()

#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import colorcet as cc
from matplotlib.colors import ListedColormap


output_dir = "C:\\Git_projects\\WAC\\classification\\test_qoutput\\" + job.job_id
path = job.get_results().download_files(f"{output_dir}")
ds = xr.open_dataset(path[0])

#%%

ds

#%%
da = ds['band_data']
n_bands = da.sizes['band']
long_names = da.attrs['long_name']   # tuple of 26 labels


# Set up subplot grid (for 26 bands, a 5x6 grid works well)
ncols = 5
nrows = (n_bands + ncols - 1) // ncols  # ceiling division

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
axes = axes.ravel()  # flatten to 1D array

for i in range(n_bands):
    da.isel(band=i).plot.imshow(
        ax=axes[i],
        add_colorbar=False,
        vmin=0,
        vmax =1

    )
    # Use the long_name for title
    axes[i].set_title(f"{long_names[i]}", fontsize=9)
    axes[i].axis("off")

# Hide unused axes
for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
