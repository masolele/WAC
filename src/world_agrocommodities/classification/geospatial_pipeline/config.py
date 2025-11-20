TEMPORAL_EXTENT = ["2020-01-01", "2021-01-01"]

# coffee
CRS = "EPSG:32629"
RESOLUTION = 10  # important; the resolution is implicitely tied to the CRS; so we need to use UTM based CRS here

SPATIAL_EXTENT = {
    "west": 677736,
    "south": 624010,
    "east": 679736,
    "north": 626010,
    "crs": CRS,
}

MAX_CLOUD_COVER = 75
QUANTILE = 0.85

PATCH_SIZE = 128
OVERLAP_SIZE = 16


JOB_OPTIONS = {
    "driver-memory": "500m",
    "driver-memoryOverhead": "2000m",
    "executor-memory": "2000m",
    "executor-memoryOverhead": "1000m",
    "python-memory": "3000m",
    "max-executors": 10,
    "allow_empty_cubes": True,
    "udf-dependency-archives": [
        "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_deps_python311.zip#onnx_deps",
    ],
}
