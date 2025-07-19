#%%

import tensorflow as tf
import tf2onnx
import onnx
import sys
import os

import sys
sys.path.append(r"C:\Git_projects\WAC\ML")

from Unet_RES_Att_models_IV import Attention_UNetFusion3I_Sentinel2_Binary 

model_name = "FusionUNet_OilpalmModel"

# Load the TensorFlow model
model_path = model_name + ".hdf5"
tf_model = tf.keras.models.load_model(model_path, compile=False)
tf_model


print(tf_model.input_shape)

#%%
# 2) Define an input signature with `None` for all dynamic dims:
#    (batch, height, width, bands)
spec = (tf.TensorSpec((None, None, None, tf_model.input_shape[-1]),
                      tf.float32,
                      name="input"),)

# 3) Convert!
onnx_model, _ = tf2onnx.convert.from_keras(
    tf_model,
    input_signature=spec,
    output_path=model_name + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + ".onnx",
    opset=13, # specific ONNX opset version
)


#%%
import onnxruntime as ort
import numpy as np
from datetime import datetime

sess = ort.InferenceSession('FusionUNet_OilpalmModel_20250718131837.onnx')

# Try any shape: batch=2, height=128, width=128
sample = np.random.rand(100, 128, 128, 17).astype(np.float32)

result = sess.run(None, {"input": sample})
print(result[0].shape)