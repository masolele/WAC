from pathlib import Path

from openeo import UDF, DataCube

# Path to inference UDF
UDF_DIR = Path(__file__).parent.parent.resolve() / "UDF"


# TODO automate patch_size
def normalize_cube(
    input_cube: DataCube,
    model_id: str,
) -> DataCube:
    """
    Normalize all bands in the input data cube using predefined normalization
    specifications for optical and linear data types.

    This function applies a user-defined function (UDF) that performs band-wise
    normalization according to `NORMALIZATION_SPECS`. For each band, the UDF:
      - Identifies the normalization group (`optical` or `linear`) based on the band name.
      - Applies the corresponding normalization function:
          * Optical bands: Log-based transformation followed by scaling.
          * Linear bands: Min–max clipping and scaling.
      - Leaves unknown bands unchanged while logging a warning.

    Detailed pre- and post-normalization statistics for each band are logged
    for traceability.

    Args:
        input_cube (DataCube):
            The input data cube whose bands will be normalized.
            Must contain a 'bands' dimension with band names matching those in
            `NORMALIZATION_SPECS`.

    Returns:
        DataCube:
            A new data cube with normalized band values, preserving the original
            spatial dimensions and band names.
    """

    context = {"model_id": model_id}

    udf_norm = UDF.from_file(UDF_DIR / "udf_normalise.py", context=context)

    output = input_cube.apply(process=udf_norm)
    return output
