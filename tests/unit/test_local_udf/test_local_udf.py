import xarray as xr
import importlib
import tempfile
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from openeo.testing.results import assert_job_results_allclose


UDF_DEPENDENCY_ARCHIVE = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_deps_python311.zip"
TEST_FILE_PATH = Path(__file__).parent / "input" / "wac-udf-input-nolonlat.nc"
EPSG = 32634
REFERENCE = Path(__file__).parent / "reference" / "wac-udf-output.nc"


def ensure_onnxruntime():
    try:
        import onnxruntime  # already available?

        return onnxruntime
    except ImportError:
        pass

    temp_dir = os.path.join(tempfile.gettempdir(), "onnxruntime_zip_deps")
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, "onnxruntime_deps.zip")

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(UDF_DEPENDENCY_ARCHIVE, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(temp_dir)

    if temp_dir not in sys.path:
        sys.path.insert(0, temp_dir)

    # Now import
    return importlib.import_module("onnxruntime")


ensure_onnxruntime()
from world_agrocommodities.classification.UDF.udf_inference import (
    apply_udf_data,
    load_ort_session,
)


def test_local_udf(tmp_path):
    ds = xr.open_dataset(TEST_FILE_PATH)
    context = {
        "model_id": "WorldAgriCommodities_Africa_v1",
    }
    # crs_attrs = ds["crs"].attrs
    # arr = ds.drop_vars("crs").astype("uint16").to_array(dim="bands")
    arr = ds.to_array(dim="bands")

    udf_data = UdfData(
        proj={"EPSG": EPSG},
        datacube_list=[XarrayDataCube(arr)],
        user_context=context,
    )

    udf_data = apply_udf_data(udf_data)
    output: xr.DataArray = udf_data.datacube_list[0].get_array()

    # Adjust band names based on model metadata
    model_id = context.get("model_id")
    _, metadata_dict = load_ort_session(model_id)

    output_classes = metadata_dict["output_classes"] + ["ARGMAX"]
    output = output.assign_coords(bands=output_classes)
    output.to_dataset(dim="bands").to_netcdf(tmp_path / "wac-udf-output.nc")

    assert_job_results_allclose(
        actual=tmp_path / "wac-udf-output.nc",
        expected=REFERENCE,
    )
