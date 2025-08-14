# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
import tf2onnx
import numpy as np
import onnx  # Required to load and print the ONNX model description

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Unet_RES_Att_models_IV import Attention_UNetFusion3I_Sentinel

# Set environment variable to allow multiple OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def convert_model_to_onnx():
    # Load the TensorFlow model
    model_path = os.path.join("..", "Model", "comcrop_udf_test.hdf5")
    tf_model = tf.keras.models.load_model(model_path, compile=False)

    # Define input shape and signature for conversion
    input_shape = (1, 64, 64, 17)  #batch size 1, 64x64 image, 17 channels
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name='input')]

    # Convert the TensorFlow model to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature, opset=13)

    # Save the ONNX model to a file
    output_path = "comcrop_udf_test.onnx"
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model converted and saved to {output_path}")

    # Load the ONNX model to print its description
    onnx_model = onnx.load(output_path)

    # Print the ONNX model description to the console
    print("\nONNX Model Description:")
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Verify the ONNX model (requires onnxruntime)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        print("\nSuccessfully verified ONNX model loading")

        # Create sample input for testing
        sample_input = np.random.rand(*input_shape).astype(np.float32)

        # Run inference with TensorFlow model
        tf_output = tf_model.predict(sample_input)

        # Run inference with ONNX model
        ort_inputs = {sess.get_inputs()[0].name: sample_input}
        ort_output = sess.run(None, ort_inputs)[0]

        # Compare outputs
        max_diff = np.max(np.abs(tf_output - ort_output))
        print(f"Maximum difference between TF and ONNX outputs: {max_diff}... which should be under 1e-03, which is "
              f"generally considered normal and acceptable for most TensorFlow-to-ONNX conversions.")

    except ImportError:
        print("Note: Install onnxruntime package to verify the model")


if __name__ == "__main__":
    # Open Neural Network Exchange conversion, we can add cmd arguments to specify the model path and output path later
    convert_model_to_onnx()
