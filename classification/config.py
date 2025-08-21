CRS = "EPSG:3035"
RESOLUTION = 10  # important; the resolution is implicitely tied to the CRS; so we need to use UTM based CRS here

PATCH_SIZE = 128
OVERLAP_SIZE = 32

SPATIAL_EXTENT = {
    'west': 3718000.31,
    'south': -1612348.84,
    'east': 3726980.14,
    'north': -1604715.33,
    'crs': CRS  
}

#TODO discuss model management; versioning, storage, access, STAC?
N_CLASSES = 24
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