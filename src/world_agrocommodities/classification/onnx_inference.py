from pathlib import Path

from openeo import UDF, DataCube

# Path to inference UDF
UDF_DIR = Path(__file__).parent.resolve() / "UDF"


# TODO automate patch_size
def run_inference(
    input_cube: DataCube,
    model_id: str,
    patch_size: int = 64,
    overlap: int = 0,
    skip_inference: bool = False,
) -> DataCube:
    """
    Run model inference over an input data cube.

    Applies a neighborhood UDF with the specified patch size and overlap and cuts off the overlap. The UDF
    should load and apply the model defined by the `model_name`.

    Args:
        input_cube: The input DataCube on which inference will be applied.
        model_name: Name of the ONNX model file (without extension) located in the folder 'dynamic_models/' within the model zip.
        patch_size: Size (in pixels) of the square window used for neighborhood inference.
        overlap: Number of pixels to overlap between patches (sliding window). Increasing this value, will result is less artefacts at the edges of patches, but will also increase the computation time.

    Returns:
        DataCube: Output data cube after inference.
    """

    context = {"model_id": model_id}

    if skip_inference:
        context["skip_inference"] = True

    udf_inference = UDF.from_file(UDF_DIR / "udf_inference.py", context=context)

    output = input_cube.apply_neighborhood(
        process=udf_inference,
        size=[
            {"dimension": "x", "value": patch_size, "unit": "px"},
            {"dimension": "y", "value": patch_size, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": overlap, "unit": "px"},
            {"dimension": "y", "value": overlap, "unit": "px"},
        ],
    )
    return output
