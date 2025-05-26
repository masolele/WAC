from pathlib import Path
from openeo import UDF

# Path to inference UDF
UDF_DIR = Path(__file__).parent.resolve() / 'UDF'
udf_inference = UDF.from_file(UDF_DIR / 'udf_inference_kernel.py')

#TODO automate patch_size
def inference_WAC(input_cube, patch_size = 64, overlap = 0):
    """
    Apply neighborhood inference UDF over input data cube using a sliding window.

    Returns:
        Processed result cube.
    """


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
    return output.rename_labels(dimension = 'bands', target = ['prediction'])