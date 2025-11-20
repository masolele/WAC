# %% pipeline_builder.py
import json
from pathlib import Path
from typing import Callable

import config as config
from geospatial_pipeline.band_normalization import normalize_cube
from geospatial_pipeline.input_cube_loader import load_input_cube
from geospatial_pipeline.onnx_inference import run_inference
from openeo import Connection, DataCube
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

SPATIAL_PARAM = Parameter.spatial_extent(
    name="spatial_extent",
    description="spatial extent in UTM coordinates",
    default=config.SPATIAL_EXTENT,
)
TEMPORAL_PARAM = Parameter.temporal_interval(
    name="temporal_extent", default=config.TEMPORAL_EXTENT
)
CRS_PARAM = Parameter.string(
    name="crs", description="CRS of the output in ", default=config.CRS
)


# %%
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
        quantile=config.QUANTILE,
        resolution=config.RESOLUTION,
        crs=CRS_PARAM,
    )

    # UDF based normalisation
    cube_normalised = normalize_cube(cube)

    # Run inference
    inference_cube = run_inference(
        cube_normalised,
        model_name=config.MODEL_NAME,
        patch_size=config.PATCH_SIZE,
        overlap=config.OVERLAP_SIZE,
    )

    # Rename bands to probability labels
    class_labels = [config.CLASS_MAPPING[i] for i in sorted(config.CLASS_MAPPING)] + [
        "ARGMAX"
    ]
    return inference_cube.rename_labels(dimension="bands", target=class_labels)


def generate_udp(
    conn: Connection,
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
        parameters=[SPATIAL_PARAM, TEMPORAL_PARAM, CRS_PARAM],
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


# %%
import openeo

conn = openeo.connect("https://openeo.dataspace.copernicus.eu/")
conn.authenticate_oidc()

cube = create_classification_cube(conn)

UDPdir = Path("C:/Git_projects/WAC/classification/UDP/")

generate_udp(
    conn=conn,
    build_cube_fn=create_classification_cube,  # <-- pass function, not result
    process_id="wac_inference_africa",
    summary="wac_inference_africa",
    output_dir=UDPdir,
)

import config as config

# %%
import openeo

conn = openeo.connect("https://openeo.dataspace.copernicus.eu/")
conn.authenticate_oidc()

cube = conn.datacube_from_json(
    "C:/Git_projects/WAC/classification/UDP/wac_inference_africa.json",
    parameters={
        "spatial_extent": config.SPATIAL_EXTENT,
        "temporal_extent": config.TEMPORAL_EXTENT,
        "crs": config.CRS,
    },
)
cube = cube.save_result(format="NetCDF")
