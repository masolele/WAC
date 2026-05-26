#  Add cloud-percentage in composite, based on SCL, after classification

from openeo import Connection, DataCube
from openeo.processes import or_, eq


def add_cloud_percentage(
    conn: Connection,
    cube: DataCube,
    temporal_extent: list,
    spatial_extent: dict,
    max_cloud_cover: int,
    resolution: float,
    crs: str,
) -> DataCube:
    scl = conn.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SCL"],
        max_cloud_cover=max_cloud_cover,
    ).resample_spatial(resolution=resolution, projection=crs)

    # Consider 0, 1, 3, 8, 9, 10 as cloud pixels based on SCL values

    cloud = scl.apply(
        lambda x: or_(
            eq(x, 0),  # No data
            or_(
                eq(x, 1),  # Saturated or defective
                or_(
                    eq(x, 3),  # Cloud shadows
                    or_(
                        eq(x, 8),  # Clouds medium probability
                        or_(
                            eq(x, 9),  # Clouds high probability
                            eq(x, 10),  # Thin cirrus
                        ),
                    ),
                ),
            ),
        )
    )

    cloud = cloud.rename_labels(dimension="bands", target=["Cloud_percentage"])

    cloud_percentage = (
        cloud.aggregate_temporal_period(period="year", reducer="mean") * 100
    )

    return cube.merge_cubes(cloud_percentage)
