# Classification Pipeline (WAC Project)

This module is part of the **WAC** project, featuring an Attention U‑Net with multi-input fusion—designed for crop classification using multi-source satellite imagery and geographic features.

## Folder Structure & Overview

classification
  - band_normalization.py        # Preprocessing: normalize satellite bands, radar, and geo-data
  -  cube_loader.py              # Build input cubes for inference
  - onnx_inference.py            # Run ONNX model inference
  - udp_generator.py             # Generate UDP (User-Defined Processing) payloads
model_conversion                 # Scripts for converting the TensorFlow model to ONNX
UDF                              # openEO-compatible user-defined functions
UDP                              # UDP outputs for integration testing
config.py         # Configuration file containing all adjustable parameters

full_processing_pipeline.ipynb  # End-to-end notebook: full pipeline and UDP creation

## Module Descriptions

### classification
- band_normalization.py: Applies preprocessing routines such as log-transformation and percentile normalization to Sentinel‑2 bands, radar (VV/VH), and geographic features.
- cube_loader.py: Packages the normalized data into the input tensor expected by the ONNX model with shape [1, 64, 64, 17 feature channels].
- onnx_inference.py: Runs inference using the ONNX model, accepting the formatted input cube.
- udp_generator.py: Creates a UDP payload for downstream use and integration testing.

### model_conversion
Contains scripts to convert the original TensorFlow model (TF 2.10.0) into an ONNX model with IR v7, opset v13.

### UDF
Houses openEO-compatible user-defined functions for modular remote processing. These UDFs are used for the dedicated band normalisation lat-lon calculation and orchestrating the model inference

### UDP
Contains generated UDPs.

### Configuration
Central configuration file which contains all adjustable parameters.

### Notebook – Full Pipeline
The full_processing_pipeline.ipynb notebook walks through preprocessing, inference, UDP generation, and visualization.

### Prerequisites
- Python 3.x   
- Dependencies: openEO

