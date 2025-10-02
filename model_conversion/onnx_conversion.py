#%%
import os
import sys
from datetime import datetime

import tensorflow as tf
import tf2onnx

# Make sure Keras backend alias is available for any Lambdas
from tensorflow.keras import backend as K

# 1) Load your old HDF5 model (this may still reference Lambda layers, but we’ll strip them out)
model_name  = "best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sent_Southeast_Asia20"
h5_path     = r"C:\\Git_projects\\WAC\\ML\\best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sent_Southeast_Asia20.hdf5"

orig_model  = tf.keras.models.load_model(h5_path, compile=False,
                                         custom_objects={'K': K})
print("✅ Loaded HDF5 model, input shape:", orig_model.input_shape)


# 2) Load your pretrained weights (no Lambda pickling issues)


#%% 3) Define an input signature with fully dynamic H×W
spec = [
    tf.TensorSpec((None, None, None, orig_model.input_shape[-1]),
                  tf.float32,
                  name="input")
]

# 4) Convert to ONNX
timestamp     = datetime.now().strftime("%Y%m%d%H%M%S")
onnx_filename = f"{model_name}.onnx"

tf2onnx.convert.from_keras(
    orig_model,
    input_signature=spec,
    opset=17,
    output_path=onnx_filename,
)

print(f"✅ ONNX model saved to: {onnx_filename}")

#%%



#%%
import onnxruntime as ort
import numpy as np
from datetime import datetime

sess = ort.InferenceSession(onnx_filename)

# Try any shape: batch=2, height=128, width=128
sample = np.random.rand(2, 128, 128, 15).astype(np.float32)

result = sess.run(None, {"input": sample})
print(result[0].shape)

#%%
