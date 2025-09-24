TEMPORAL_EXTENT = ['2023-01-01', '2024-01-01'] 

#coffee
CRS = "EPSG:32629" 

SPATIAL_EXTENT = {
    'west': 677736,
    'south': 624010,
    'east': 694576,
    'north': 638629,
    'crs': CRS  
}


CLASS_MAPPING = {
    0: "Background",
    1: "Other_Large_Scale_Cropland",
    2: "Pasture",
    3: "Mining",
    4: "Other_Small_Scale_Cropland",
    5: "Roads",
    6: "Forest",
    7: "Plantation_Forest",
    8: "Coffee",
    9: "Built_up",
    10: "Water",
    11: "Oil_Palm",
    12: "Rubber",
    13: "Cacao",
    14: "Avocado",
    15: "Soy",
    16: "Sugar",
    17: "Maize",
    18: "Banana",
    19: "Pineapple",
    20: "Rice",
    21: "Wood_Logging",
    22: "Cashew",
    23: "Tea",
    24: "Other",
}

#TODO discuss model management; versioning, storage, access, STAC?
MODEL_NAME = "best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sentAfrica6"

MAX_CLOUD_COVER = 75
QUANTILE = 0.8


RESOLUTION = 10  # important; the resolution is implicitely tied to the CRS; so we need to use UTM based CRS here

PATCH_SIZE = 128
OVERLAP_SIZE = 32


JOB_OPTIONS = {'driver-memory': '500m',
 'driver-memoryOverhead': '2000m',
 'executor-memory': '2000m',
 'executor-memoryOverhead': '1000m',
 'python-memory': '3000m',
 'max-executors': 10,
 "udf-dependency-archives": [
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_dependencies_1.16.3.zip#onnx_deps",
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/WorldAgriCommodities/dynamic_models.zip#onnx_models"
        ]
 }
 