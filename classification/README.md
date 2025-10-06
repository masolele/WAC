# Classification Module — WAC

This folder contains the code, notebooks, and configuration for running the **classification / mapping** step of the WAC pipeline.  
Users must first edit the **`config.py`** to match their region, temporal extent, and model settings before running the workflows.

---

## 🛠️ Setup & Configuration

### 1. Edit `config.py`

Before running any classification workflows, open `config.py` and review/adjust the following important settings:

| Parameter | Purpose | What You Need to Do |
|---|---|---|
| `xmin`, `xmax`, `ymin`, `ymax` , `epsg` | Spatial bounding box (in UTM coordinates) for your target area | Change these to the UTM (eastings / northings) bounds covering your study region | The area should maximally be 20kmx20km
| `t0` (or `start_date`), `t1` (or `end_date`) | Temporal extent for the classification (time window of observations) | Set to the start and end dates you want the model to consider; the model was trained expecting a full year of input data|
`epsg` | Coordinate Reference System / UTM zone of th eoutput | Ensure this matches the correct UTM zone for your region |
| `model_name` (or equivalent) | Which model to apply | Currently we only have the africa model onboarded. Future models may be selectable here. |


## 🧪 Running the Pipeline via Notebooks

After configuring `config.py`, you can run the full classification pipeline using the provided notebooks (e.g. `full_processing_pipeline.ipynb`).

### Steps:

1. **Open the notebook**  
   Use Jupyter / JupyterLab / VSCode to open `full_processing_pipeline.ipynb` (or similar).  
   Make sure the working directory is set properly (so relative paths resolve).

2. **Run all cells in order**  
   The notebook is intended to run steps such as:
   - Preprocessing / data loading  
   - Classification inference  
   - Output stitching & export  
   - (Optional) Visualization or basic validation  


### Prerequisites
- Python 3.x   
- Dependencies: openEO

