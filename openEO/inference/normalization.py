
from openeo.processes import clip, merge_cubes, ln, exp


# Define band order
BAND_ORDER = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12",
    "NDVI", "NDRE", "EVI",
    "VV", "VH",
    "DEM", "lon", "lat"
]

# Constants
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



def normalize_band(band, band_name):
    """Apply band-specific normalization"""
    if band_name in NORMALIZATION_SPECS["optical"]:
        min_val, max_val = NORMALIZATION_SPECS["optical"][band_name]
        return _normalize_optical(band, min_val, max_val)
    
    if band_name in NORMALIZATION_SPECS["linear"]:
        min_val, max_val = NORMALIZATION_SPECS["linear"][band_name]
        return _normalize_linear(band, min_val, max_val)
    
    return band  # Return unchanged if no spec

def _normalize_optical(band, min_val, max_val):
    """Optical band normalization using openEO processes"""
    scaled = band * 0.005
    transformed = scaled.apply(lambda x: 1 + x.ln())
    normalized = (transformed - min_val) / (max_val - min_val)
    t = normalized * 5.0 - 1.0
    exp_t = normalized.apply(lambda x: x.exp())
    return exp_t / (exp_t + 1.0)

def _normalize_linear(band, min_val, max_val):
    """Linear normalization using openEO processes"""
    clipped = band.apply(lambda x: x.clip(min_val, max_val))
    return (clipped - min_val) / (max_val - min_val)

def normalize_cube(cube):
    """Apply normalization to all bands and merge"""
    # Process bands
    normalized_bands = [
        normalize_band(cube.band(band_name), band_name)
        for band_name in BAND_ORDER
    ]
    
        # Merge all bands
    result = normalized_bands[0].add_dimension(name="bands", label=BAND_ORDER[0])

    for i in range(1, len(normalized_bands)):
        band_cube = normalized_bands[i]
        band_cube = band_cube.add_dimension(name="bands", label=BAND_ORDER[i])
        result = merge_cubes(result, band_cube)

    return result 