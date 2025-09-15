CRS = "EPSG:32630" #cacao
CRS = "EPSG:32637" #coffee

RESOLUTION = 10  # important; the resolution is implicitely tied to the CRS; so we need to use UTM based CRS here

PATCH_SIZE = 128
OVERLAP_SIZE = 32

#cacao
SPATIAL_EXTENT = {
    'west': 716200,
    'south': 605530,
    'east': 732380,
    'north': 622410,
    'crs': CRS  
}

#coffee
SPATIAL_EXTENT = {
    'west': 470787,
    'south': 654996,
    'east': 488151,
    'north': 673899,
    'crs': CRS  
}

CLASS_MAPPING = {
    0: "Background",
    1: "Other Large Scale Cropland",
    2: "Pasture",
    3: "Mining",
    4: "Other Small Scale Cropland",
    5: "Roads",
    6: "Forest",
    7: "Plantation Forest",
    8: "Coffee",
    9: "Built up",
    10: "Water",
    11: "Oil Palm",
    12: "Rubber",
    13: "Cacao",
    14: "Avocado",
    15: "Soy",
    16: "Sugar",
    17: "Maize",
    18: "Banana",
    19: "Pineapple",
    20: "Rice",
    21: "Wood Logging",
    22: "Cashew",
    23: "Tea",
}

#TODO discuss model management; versioning, storage, access, STAC?
MODEL_NAME = "best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sentAfrica6"

TEMPORAL_EXTENT = ['2023-01-01', '2024-01-01'] 
MAX_CLOUD_COVER = 85

JOB_OPTIONS = {'driver-memory': '2000m',
 'driver-memoryOverhead': '2000m',
 'executor-memory': '3000m',
 'executor-memoryOverhead': '3000m',
 'python-memory': '8000m',
 'max-executors': 20,
 "udf-dependency-archives": [
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_dependencies_1.16.3.zip#onnx_deps",
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/WorldAgriCommodities/dynamic_models.zip#onnx_models"
        ]
 }