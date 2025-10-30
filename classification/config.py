TEMPORAL_EXTENT = ["2023-01-01", "2024-01-01"]

# coffee
CRS = "EPSG:32750"
RESOLUTION = 10  # important; the resolution is implicitely tied to the CRS; so we need to use UTM based CRS here

SPATIAL_EXTENT = {
    "west": 818051,
    "south": 9495261,
    "east": 836471,
    "north": 9515181,
    "crs": CRS,
}

MAX_CLOUD_COVER = 75
QUANTILE = 0.9

PATCH_SIZE = 128
OVERLAP_SIZE = 32


JOB_OPTIONS = {
    "driver-memory": "500m",
    "driver-memoryOverhead": "2000m",
    "executor-memory": "2000m",
    "executor-memoryOverhead": "1000m",
    "python-memory": "3000m",
    "max-executors": 10,
    "image-name": "python38",
    "udf-dependency-archives": [
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_dependencies_1.16.3.zip#onnx_deps",
    ],
}
