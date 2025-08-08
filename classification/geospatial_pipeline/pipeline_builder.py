# pipeline_builder.py
from pathlib import Path
import json
from openeo import Connection, DataCube
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

import config
from geospatial_pipeline.input_cube_loader import load_input_cube
from geospatial_pipeline.onnx_inference import run_inference


def build_pipeline(conn: Connection) -> DataCube:
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
        spatial_extent=config.SPATIAL_EXTENT,
        temporal_extent=config.TEMPORAL_EXTENT,
        max_cloud_cover=config.MAX_CLOUD_COVER,
        resolution=config.RESOLUTION,
        crs=config.CRS
    )

    # Run inference
    inference_cube = run_inference(
        cube,
        model_name=config.MODEL_NAME,
        patch_size=config.PATCH_SIZE,
        overlap=config.OVERLAP_SIZE
    )

    # Rename bands to probability labels
    class_labels = [f"prob_class_{c}" for c in range(config.N_CLASSES)] + ['classification']
    return inference_cube.rename_labels(dimension='bands', target=class_labels)


def generate_udp(conn: Connection,
                build_pipeline_fn: DataCube,
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

    # Define parameters with defaults from config
    spatial_extent = Parameter.spatial_extent(name="spatial_extent",default=config.SPATIAL_EXTENT)
    temporal_extent = Parameter.temporal_interval(name="temporal_extent",default=config.TEMPORAL_EXTENT)
    
    # Connect and build the data cube (could be just to get the process graph)
    datacube = build_pipeline_fn(conn)


    # Build the UDP process dictionary
    udp_dict = build_process_dict(
        process_graph=datacube,
        process_id=process_id,
        summary=summary,
        parameters=[spatial_extent, temporal_extent],
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