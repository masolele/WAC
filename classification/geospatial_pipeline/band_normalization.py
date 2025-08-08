from openeo import DataCube
import logging

logger = logging.getLogger(__name__)

# ML model onstants for normalization ranges
NORMALIZATION_SPECS = {
    "optical": {
        "B02": (1.7417268007636313, 2.023298706048351),
        "B03": (1.7261204997060209, 2.038905204308012),
        "B04": (1.6798346251414997, 2.179592821212937),
        "B05": (2.3828939530384052, 2.7578332604178284),
        "B06": (1.7417268007636313, 2.023298706048351),
        "B07": (1.7417268007636313, 2.023298706048351),
        "B08": (1.7417268007636313, 2.023298706048351),
        "B11": (1.7417268007636313, 2.023298706048351),
        "B12": (1.7417268007636313, 2.023298706048351)
    },
    "linear": {
        "NDVI": (-1, 1),
        "NDRE": (-1, 1),
        "EVI": (-1, 1),
        "VV": (-25, 0),
        "VH": (-30, -5),
        "DEM": (-400, 8000),
        "lon": (-180, 180),
        "lat": (-60, 60)
    }
}

    
def _normalize_optical(band: DataCube, min_val: float, max_val: float) -> DataCube:
    """
    Apply non-linear normalization to optical bands.

    Args:
        band: Input optical band as a DataCube.
        min_val: Minimum expected log-transformed value.
        max_val: Maximum expected log-transformed value.

    Returns:
        DataCube: Normalized optical band.
    """
    scaled = band * 0.005
    transformed = scaled.apply(lambda x: 1 + x.ln())
    normalized = (transformed - min_val) / (max_val - min_val)
    t = normalized * 5.0 - 1.0
    exp_t = t.apply(lambda x: x.exp())
    return exp_t / (exp_t + 1.0)

def _normalize_linear(band: DataCube, min_val: float, max_val: float) -> DataCube:
    """
    Apply min-max normalization to a band with linear value range.

    Args:
        band: Input band as a DataCube.
        min_val: Minimum expected value.
        max_val: Maximum expected value.

    Returns:
        DataCube: Normalized band.
    """
    clipped = band.apply(lambda x: x.clip(min_val, max_val))
    return (clipped - min_val) / (max_val - min_val)

def normalize_cube(cube: DataCube) -> DataCube:
    """
    Normalize each band in the input cube based on predefined specs.

    Uses either optical or linear normalization depending on the band.
    Bands without a known specification will be returned unchanged.

    Args:
        cube: Input DataCube with a 'bands' dimension.

    Returns:
        DataCube: Cube with normalized bands.
    """
    # Skip normalization if cube lacks bands dimension
    if 'bands' not in cube.metadata.dimension_names():
        logger.warning("Input cube has no 'bands' dimension, skipping normalization.")
        return cube
    
    # Process each band
    normalized_bands = []
    for band_name in cube.metadata.band_names:
        band_cube = cube.band(band_name)
        
        if band_name in NORMALIZATION_SPECS["optical"]:
            min_val, max_val = NORMALIZATION_SPECS["optical"][band_name]
            norm_band = _normalize_optical(band_cube, min_val, max_val)
        elif band_name in NORMALIZATION_SPECS["linear"]:
            min_val, max_val = NORMALIZATION_SPECS["linear"][band_name]
            norm_band = _normalize_linear(band_cube, min_val, max_val)
        else:
            norm_band = band_cube  # Unspecified bands remain unchanged
        
        normalized_bands.append(norm_band.add_dimension('bands', band_name, 'bands'))
    
    # Merge processed bands
    output_cube = normalized_bands[0]
    for band in normalized_bands[1:]:
        output_cube = output_cube.merge_cubes(band)
    return output_cube