from openeo.processes import clip, ln, exp, add, divide, multiply, subtract
from openeo import DataCube

# Define band order
BAND_ORDER = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12",
    "NDVI", "NDRE", "EVI",
    "VV", "VH",
    "DEM", "lon", "lat"
]

# Normalization specifications
NORMALIZATION_SPECS = {
    # Optical bands
    "B02": (1.7417268007636313, 2.023298706048351),
    "B03": (1.7261204997060209, 2.038905204308012),
    "B04": (1.6798346251414997, 2.179592821212937),
    "B05": (2.3828939530384052, 2.7578332604178284),
    "B06": (1.7417268007636313, 2.023298706048351),
    "B07": (1.7417268007636313, 2.023298706048351),
    "B08": (1.7417268007636313, 2.023298706048351),
    "B11": (1.7417268007636313, 2.023298706048351),
    "B12": (1.7417268007636313, 2.023298706048351),
    
    # Other bands
    "NDVI": (-1, 1),
    "NDRE": (-1, 1),
    "EVI": (-1, 1),
    "VV": (-25, 0),
    "VH": (-30, -5),
    "DEM": (-400, 8000),
    "lon": (-180, 180),
    "lat": (-60, 60)
}

def normalize_cube(cube: DataCube) -> DataCube:
    """Apply all normalizations and return concatenated DataCube"""
    normalized_bands = []
    
    for band in BAND_ORDER:
        min_val, max_val = NORMALIZATION_SPECS[band]
        band_cube = cube.band(band)
        
        if band in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]:
            # Optical normalization using nested expressions
            # Step 1: Log transformation and scaling
            step1 = band_cube.apply(
                lambda x: ((x * 0.005 + 1).ln() - min_val) / (max_val - min_val)
            )
            
            # Step 2: Sigmoid transformation
            normalized_band = step1.apply(
                lambda y: (y * 5 - 1).exp() / ((y * 5 - 1).exp() + 1)
            )
        else:
            # Linear normalization
            normalized_band = band_cube.apply(
                lambda x: (x.clip(min_val, max_val) - min_val) / (max_val - min_val)
            )
        
        normalized_bands.append(normalized_band)
    
    # Start with first band
    result = normalized_bands[0].add_dimension(name='bands', label=BAND_ORDER[0])
    
    # Sequentially concatenate other bands
    for i in range(len(normalized_bands[1:])):
        result = result.merge_cubes(normalized_bands[i + 1].add_dimension(name='bands', label=BAND_ORDER[i + 1]))

    return result