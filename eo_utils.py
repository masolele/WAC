import openeo
import openeo.processes as eop
from openeo.extra.spectral_indices.spectral_indices import compute_and_rescale_indices


def compute_yearly_features_and_monthly_composites(
    s2_datacube: openeo.DataCube, bands: list = None, nb_of_months: int = 12
) -> openeo.DataCube:
    """
    Compute yearly features and monthly composites
    """
    s2_rescaled = ndvi_and_rescale(s2_datacube)

    if bands is not None:
        s2_rescaled = s2_rescaled.filter_bands(bands)

    s2_yearly = compute_yearly_features(s2_rescaled)
    s2_monthly = compute_monthly_composites(s2_rescaled, nb_of_months=nb_of_months)

    s2_merged = s2_yearly.merge_cubes(s2_monthly)
    return s2_merged

def ndvi_and_rescale(s2_datacube: openeo.DataCube) -> openeo.DataCube:
    """
    Compute NDVI and rescale to 0-30000
    """
    s2_index_dict = {
        "collection": {
            "input_range": [-32_768, 32_766],
            "output_range": [-32_768, 32_766],
        },
        "indices": {
            "NDVI": {"input_range": [-1, 1], "output_range": [-10_000, 10_000]}
        },
    }
    return compute_and_rescale_indices(s2_datacube, s2_index_dict, append=True)


def compute_yearly_features(
    s2_datacube,
) -> openeo.DataCube:
    # Aggregate to dekad
    s2_dekad = s2_datacube.aggregate_temporal_period("dekad", reducer="median")

    s2_dekad = s2_dekad.apply_dimension(
        dimension="t", process="array_interpolate_linear"
    )

    # Compute statistics
    s2_features = compute_percentiles(s2_dekad)

    return s2_features


def compute_percentiles(base_features):
    """
    Computes P10, P25, P50, P75, P90
    """

    def computeStats(input_timeseries):
        return input_timeseries.quantiles(probabilities=[0.1, 0.25, 0.50, 0.75, 0.9])

    stats = base_features.apply_dimension(
        dimension="t", target_dimension="bands", process=computeStats
    )
    all_bands = [
        band + "_" + stat
        for band in base_features.metadata.band_names
        for stat in ["P10", "P25", "P50", "P75", "P90"]
    ]
    return stats.rename_labels("bands", all_bands)


def compute_monthly_composites(s2_datacube, nb_of_months=12) -> openeo.DataCube:
    # Aggregate to monthly composites
    s2_monthly = s2_datacube.aggregate_temporal_period("month", reducer="median")

    # Save the monthly composites as bands
    s2_timeless = timesteps_as_bands(s2_monthly, nb_of_months=nb_of_months)

    return s2_timeless


def timesteps_as_bands(datacube, nb_of_months=12):
    band_names = [
        band + "_M" + str(i + 1)
        for band in datacube.metadata.band_names
        for i in range(nb_of_months)
    ]

    # for each timestep and each band, create a new band
    # this results in T*B bands, with T the number of timesteps and B the number of bands
    result = datacube.apply_dimension(
        dimension="t",
        target_dimension="bands",
        process=lambda d: eop.array_create(data=d),
    )
    return result.rename_labels("bands", band_names)

def divide_bands(bands):
    vv = bands[0]
    vh = bands[1]
    return eop.array_append(bands, eop.divide(vh, vv))

