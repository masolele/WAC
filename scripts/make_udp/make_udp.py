# Script that generates an openEO UDP based on the local WAC

import json
from pathlib import Path
from typing import Callable

import world_agrocommodities.classification.config as config
from world_agrocommodities import map_commodities

import openeo
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

SPATIAL_PARAM = Parameter.spatial_extent(
    name="spatial_extent",
    description="spatial extent.",
)

TEMPORAL_PARAM = Parameter.temporal_interval(name="temporal_extent")
CRS_PARAM = Parameter.string(
    name="crs", description="CRS identifier of the output, e.g. 'EPSG:4326'"
)

MODEL_PARAM = Parameter.string(name="model_id", description="Model identifier")


connection = openeo.connect(
    "https://openeo.dataspace.copernicus.eu/"
).authenticate_oidc()

pg = map_commodities(
    connection=connection,
    spatial_extent=SPATIAL_PARAM,
    temporal_extent=TEMPORAL_PARAM,
    model_id=MODEL_PARAM,
    crs=CRS_PARAM,
).save_result(format="netCDF")


udp_dict = build_process_dict(
    process_graph=pg,
    process_id="world_agrocommodities",
    parameters=[SPATIAL_PARAM, TEMPORAL_PARAM, CRS_PARAM, MODEL_PARAM],
    default_job_options=config.JOB_OPTIONS,
)


output_path = (
    Path(__file__).parent.parent.parent
    / "src/world_agrocommodities/udp/world_agrocommodities.json"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
# Save the UDP JSON file
with open(output_path, "w") as f:
    json.dump(udp_dict, f, indent=2)

print(f"Saved UDP JSON to {output_path}")
