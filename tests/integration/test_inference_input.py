"""Integrationtest for the inference benchmark"""

import openeo
import pytest
import world_agrocommodities.classification.config as config
from pathlib import Path
from world_agrocommodities import map_commodities
from openeo.testing.results import assert_job_results_allclose


@pytest.mark.integration
def test_inference_input_benchmark(tmp_path):
    # TODO: authentication with refresh token requires interactive login once, for CI we need to use a service account
    c = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

    spatial_extent = {
        "west": 817475.0741295104,
        "south": -522776.47462956805,
        "east": 817838.336850123,
        "north": -522493.8634415434,
        "crs": "EPSG:32634",
    }
    temporal_extent = ["2025-01-01", "2025-12-31"]
    crs = "EPSG:32634"

    model_id = "WorldAgriCommodities_Africa_v1"

    pg = map_commodities(
        connection=c,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        model_id=model_id,
        crs=crs,
        skip_inference=True,
    )

    job = pg.execute_batch(
        title="WorldAgroCommodities Inference Inputs Benchmark Test",
        out_format="netcdf",
        job_options=config.JOB_OPTIONS,
    )

    reference_dir = Path(__file__).parent / "reference" / "inference_input_benchmark"
    actual_dir = tmp_path / "actual"
    results = job.get_results()

    results.download_files(target=actual_dir, include_stac_metadata=True)

    assert_job_results_allclose(actual=actual_dir, expected=reference_dir)
