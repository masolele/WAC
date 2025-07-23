from pathlib import Path
from openeo import UDF

# Path to inference UDF
UDF_DIR = Path(__file__).parent.resolve() / 'UDF'

#TODO automate patch_size
def run_inference(input_cube, model_name, patch_size = 64, overlap = 0):
    """
    Apply neighborhood inference UDF over input data cube using a sliding window.

    Returns:
        Processed result cube.
    """

    context = {
        'model_path': f"dynamic_models//{model_name}.onnx"
    }

    udf_inference = UDF.from_file(UDF_DIR / 'udf_inference.py', context=context)

    output =  input_cube.apply_neighborhood(
        process=udf_inference,
        size=[
            {'dimension': 'x', 'value': patch_size, 'unit': 'px'},
            {'dimension': 'y', 'value': patch_size, 'unit': 'px'},
        ],
        overlap=[
            {'dimension': 'x', 'value': overlap, 'unit': 'px'},
            {'dimension': 'y', 'value': overlap, 'unit': 'px'},
        ]
    )
    return output
