import numpy as np
import openeo
import openeo.processes as eop

SCL_LEGEND = {
    "no_data": 0,
    "saturated_or_defective": 1,
    "dark_area_pixels": 2,
    "cloud_shadows": 3,
    "vegetation": 4,
    "not_vegetated": 5,
    "water": 6,
    "unclassified": 7,
    "cloud_medium_probability": 8,
    "cloud_high_probability": 9,
    "thin_cirrus": 10,
    "snow": 11,
}


def compute_scl_aux(
    scl_datacube: openeo.DataCube,
    erode_r=None,
    dilate_r=None,
    max_invalid_ratio=None,
    snow_dilate_r=3,
    max_invalid_snow_cover=0.9,
) -> openeo.DataCube:
    """
    Compute the SCL auxiliary datacube
    """

    classified = (
        scl_datacube.apply_dimension(dimension="bands", process=scl_to_masks)
        .apply(lambda x: x * 1)
        .rename_labels(
            dimension="bands",
            target=[
                "clouds",  # 0
                "saturated",  # 1
                "dark",  # 2
                "snow",  # 3
                "water",  # 4
                "veg",  # 5
                "notveg",  # 6
                "ts_obs",  # 7
            ],
        )
    )

    summed = classified.reduce_dimension("t", "sum")

    covers = summed.apply_dimension(
        dimension="bands", process=compute_cover
    ).rename_labels(
        dimension="bands",
        target=[
            "cover_clouds",  # 0
            "cover_saturated",  # 1
            "cover_dark",  # 2
            "cover_snow",  # 3
            "cover_water",  # 4
            "cover_veg",  # 5
            "cover_notveg",  # 6
        ],
    )
    # return None, covers

    saturated = classified.band("saturated")  # 1
    ts_obs = classified.band("ts_obs")  # 7
    clouds = classified.band("clouds")  # 0

    obs = summed.band("ts_obs")

    nonpermanent_snow_mask = compute_nonpermantent_snow_mask(
        classified=classified,
        covers=covers,
        snow_dilate_r=snow_dilate_r,
        max_invalid_snow_cover=max_invalid_snow_cover,
    )

    mask, ma_mask = get_mask_and_ma_mask(
        clouds, saturated, nonpermanent_snow_mask, ts_obs
    )
    invalid_before = ma_mask.reduce_dimension(dimension="t", reducer="sum").merge_cubes(
        obs,
        overlap_resolver=eop.divide,
    )

    if erode_r:
        kernel = create_disc_kernel(erode_r)
        clouds = (clouds.apply(eop.not_)).apply_kernel(kernel) == 0

    if dilate_r:
        kernel = create_disc_kernel(dilate_r)
        clouds = clouds.apply_kernel(kernel) > 0

    mask, ma_mask = get_mask_and_ma_mask(
        clouds, saturated, nonpermanent_snow_mask, ts_obs
    )
    invalid_after = ma_mask.reduce_dimension(dimension="t", reducer="sum").merge_cubes(
        obs,
        overlap_resolver=eop.divide,
    )

    invalid_ratios = invalid_before.add_dimension(
        name="bands", label="scl_invalid_before", type="bands"
    ).merge_cubes(
        invalid_after.add_dimension(
            name="bands", label="scl_invalid_after", type="bands"
        ),
    )

    # In contrast to the implementation in satio, False values in mask are considered as valid pixels
    # as he openEO process `mask` replaces all values that are True in a mask with null values.

    if max_invalid_ratio is not None:
        max_invalid_mask = invalid_after > max_invalid_ratio
        mask = mask | max_invalid_mask

    # Merge the aux data
    aux = obs.add_dimension(name="bands", label="obs_l2a", type="bands")
    aux = aux.merge_cubes(invalid_ratios)
    aux = aux.merge_cubes(
        covers.filter_bands(
            bands=[
                "cover_snow",
                "cover_dark",
                "cover_water",
                "cover_veg",
                "cover_notveg",
            ]
        ).rename_labels(
            dimension="bands",
            target=[
                "scl_snow_cover",
                "scl_dark_cover",
                "scl_water_cover",
                "scl_veg_cover",
                "scl_notveg_cover",
            ],
        )
    )

    return mask, aux


def create_disc_kernel(radius: int):
    """
    Create a discrete circular kernel with a given radius
    """
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    return kernel.astype(int).tolist()


def get_mask_and_ma_mask(
    clouds: openeo.DataCube,
    saturated: openeo.DataCube,
    nonpermanent_snow_mask: openeo.DataCube,
    ts_obs: openeo.DataCube,
) -> tuple[openeo.DataCube, openeo.DataCube]:
    mask = clouds.merge_cubes(saturated, overlap_resolver=eop.or_).merge_cubes(
        nonpermanent_snow_mask,
        overlap_resolver=eop.or_,
    )
    ma_mask = mask.merge_cubes(ts_obs, overlap_resolver=eop.and_)
    return mask, ma_mask


def scl_to_masks(bands):
    scl_layer = bands[0]
    clouds = eop.any(
        eop.array_create(
            [
                (scl_layer == SCL_LEGEND["cloud_shadows"]),
                (scl_layer == SCL_LEGEND["cloud_medium_probability"]),
                (scl_layer == SCL_LEGEND["cloud_high_probability"]),
                (scl_layer == SCL_LEGEND["thin_cirrus"]),
            ]
        )
    )
    saturated = scl_layer == SCL_LEGEND["saturated_or_defective"]
    dark = scl_layer == SCL_LEGEND["dark_area_pixels"]
    snow = scl_layer == SCL_LEGEND["snow"]
    water = scl_layer == SCL_LEGEND["water"]
    veg = scl_layer == SCL_LEGEND["vegetation"]
    notveg = scl_layer == SCL_LEGEND["not_vegetated"]

    ts_obs = scl_layer != SCL_LEGEND["no_data"]

    return eop.array_create([clouds, saturated, dark, snow, water, veg, notveg, ts_obs])


def compute_cover(bands):
    obs = bands[7]
    return eop.array_create(
        [
            bands[0] / obs,
            bands[1] / obs,
            bands[2] / obs,
            bands[3] / obs,
            bands[4] / obs,
            bands[5] / obs,
            bands[6] / obs,
        ]
    )


def compute_nonpermantent_snow_mask(
    classified, covers, snow_dilate_r=3, max_invalid_snow_cover=0.9
):
    permanent_snow = covers.reduce_dimension(
        "bands", reducer=lambda x: compute_nonpermanent_snow(x, max_invalid_snow_cover)
    )
    nonpermanent_snow = permanent_snow.apply(eop.not_)

    snow = classified.band("snow")

    # permanent_snow_mask = snow & permanent_snow
    permanent_snow_mask = snow.merge_cubes(permanent_snow, overlap_resolver=eop.and_)
    # nonpermanent_snow_mask = snow & nonpermanent_snow
    nonpermanent_snow_mask = snow.merge_cubes(
        nonpermanent_snow, overlap_resolver=eop.and_
    )

    kernel = create_disc_kernel(snow_dilate_r)
    snow_dil = nonpermanent_snow_mask.apply_kernel(kernel).apply(lambda x: x > 0)

    return snow_dil.merge_cubes(
        permanent_snow_mask.apply(eop.not_),
        overlap_resolver=eop.and_,
    )


def compute_nonpermanent_snow(bands, max_invalid_snow_cover=0.9):
    cover_valid = 1 - bands[0] - bands[1]

    permanent_snow = bands[3] / cover_valid > max_invalid_snow_cover
    return eop.array_create(permanent_snow)