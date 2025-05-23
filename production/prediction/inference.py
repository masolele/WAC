from pathlib import Path
from openeo import UDF

# Path to inference UDF
UDF_DIR = Path(__file__).parent.resolve() / 'UDF'
udf_inference = UDF.from_file(UDF_DIR / 'udf_inference.py')


def inference_WAC(input_cube, patch_size = 56, overlap = 4):
    """
    Apply neighborhood inference UDF over input data cube using a sliding window.

    Returns:
        Processed result cube.
    """
    window_size = patch_size + 2 * overlap  # Should equal 64
    assert window_size == 64, f"Invalid window size: {window_size}. Must be 64 (patch + 2 * overlap)."

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
    output = output.rename_labels(dimension="bands",target=['prediction'])
    return output