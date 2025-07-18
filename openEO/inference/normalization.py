from openeo.processes import clip, log, exp, add, divide, multiply, subtract

# Constants
NORMALIZATION_SPECS = {
    # Optical bands: (min, max) for percentile scaling
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
    
    # Linear normalizations: (min, max) for clip and scale
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
    """Generic band normalization router"""
    if band_name in NORMALIZATION_SPECS["optical"]:
        return _normalize_optical(band, *NORMALIZATION_SPECS["optical"][band_name])
    
    if band_name in NORMALIZATION_SPECS["linear"]:
        min_val, max_val = NORMALIZATION_SPECS["linear"][band_name]
        return _normalize_linear(band, min_val, max_val)
    
    # Return unmodified if no spec found
    return band

def _normalize_optical(band, min_val, max_val):
    """Complex optical band normalization"""
    scaled = multiply(band, 0.005)
    transformed = log(add(scaled, 1))
    normalized = divide(subtract(transformed, min_val), max_val - min_val)
    exp_part = exp(subtract(multiply(normalized, 5), 1))
    return divide(exp_part, add(exp_part, 1))

def _normalize_linear(band, min_val, max_val):
    """Simple linear min-max normalization"""
    clipped = clip(band, min_val, max_val)
    return divide(subtract(clipped, min_val), max_val - min_val)