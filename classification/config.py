TEMPORAL_EXTENT = ['2020-01-01', '2021-01-01'] 

#coffee
CRS = "EPSG:32629" 
RESOLUTION = 10  # important; the resolution is implicitely tied to the CRS; so we need to use UTM based CRS here

SPATIAL_EXTENT = {
    'west': 677736,
    'south': 624010,
    'east': 694576,
    'north': 638629,
    'crs': CRS,
}

MAX_CLOUD_COVER = 75
QUANTILE = 0.85

PATCH_SIZE = 128
OVERLAP_SIZE = 16


JOB_OPTIONS = {'driver-memory': '500m',
 'driver-memoryOverhead': '2000m',
 'executor-memory': '2000m',
 'executor-memoryOverhead': '1000m',
 'python-memory': '3000m',
 'max-executors': 10,
 "udf-dependency-archives": [
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_dependencies_1.16.3.zip#onnx_deps",
        ]
 }


#TEST SITE 4
#CRS = "EPSG:32629"
#RESOLUTION = 10
#SPATIAL_EXTENT = {
 #   'west': 677736,
#    'south': 624010,
#    'east': 694576,
#    'north': 638629,
#    'crs': EPSG:32629  
#}


#TEST SITE 5
#CRS = "EPSG:32648"
#RESOLUTION = 10
#SPATIAL_EXTENT = {
#    'west': 811494,
#    'south': 1543913,
#    'east': 831624,
#    'north': 1564783,
#    'crs': EPSG:32648,
#}


#TEST SITE 8
#CRS = "EPSG:32750"
#RESOLUTION = 10
#SPATIAL_EXTENT = {
#    'west': 818051,
#    'south': 9495261,
#    'east': 836471,
#    'north': 9515181,
#    'crs': EPSG:32750,
#}
#CONTINENT = "Asia"

#TEST SITE 7
#CRS = "EPSG:32618"
#RESOLUTION = 10
#SPATIAL_EXTENT = {
#    'west': 371673,
#    'south': 192066,
#    'east': 392483,
#    'north': 210946,
#    'crs': EPSG:32618,
#}
