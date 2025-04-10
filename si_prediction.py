import os
from keras.models import load_model
import numpy as np
from skimage import io
# keep below line in script at ALL times
from Unet_RES_Att_models_IV import Attention_UNetFusion3I_SentinelMLP
import matplotlib.pyplot as plt

# Set environment variable to allow multiple OpenMP libraries to be loaded
# This is necessary to prevent conflicts with OpenMP runtime libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def normalise_vv(raster):
    """
    Normalize the VV radar data in the raster.

    Values below -25 are set to -25 and those above 0 are set to 0.
    Normalizes the raster to a range of [0, 1].
    """
    raster[raster < -25] = -25
    raster[raster > 0] = 0
    return (raster + 25) / 25


def normalise_vh(raster):
    """
    Normalize the VH radar data in the raster.

    Values below -30 are set to -30 and those above -5 are set to -5.
    Normalizes the raster to a range of [0, 1].
    """
    raster[raster < -30] = -30
    raster[raster > -5] = -5
    return (raster + 30) / 25


def normalise_longitude(raster):
    """
    Normalize the longitude data in the raster.

    Values below -180 are set to -180 and those above 180 are set to 180.
    Normalizes the raster to a range of [0, 1].
    """
    raster[raster < -180] = -180
    raster[raster > 180] = 180
    return (raster + 180) / 360


def normalise_latitude(raster):
    """
    Normalize the latitude data in the raster.

    Values below -60 are set to -60 and those above 60 are set to 60.
    Normalizes the raster to a range of [0, 1].
    """
    raster[raster < -60] = -60
    raster[raster > 60] = 60
    return (raster + 60) / 120


def normalise_altitude(raster):
    """
    Normalize the altitude data in the raster.

    Values below -400 are set to -400 and those above 8000 are set to 8000.
    Normalizes the raster to a range of [0, 1].
    """
    raster[raster < -400] = -400
    raster[raster > 8000] = 8000
    return (raster + 400) / 8400


def norm(image):
    """
    Normalize the image data using predefined percentiles.

    Applies logarithmic transformation, scaling, and sigmoid transfer to the image data.

    """
    NORM_PERCENTILES = np.array([
        [1.7417268007636313, 2.023298706048351],
        [1.7261204997060209, 2.038905204308012],
        [1.6798346251414997, 2.179592821212937],
        [2.3828939530384052, 2.7578332604178284],
        [1.7417268007636313, 2.023298706048351],
        [1.7417268007636313, 2.023298706048351],
        [1.7417268007636313, 2.023298706048351],
        [1.7417268007636313, 2.023298706048351],
        [1.7417268007636313, 2.023298706048351]
    ])

    # Apply logarithmic transformation and normalization to the image
    image = np.log(image * 0.005 + 1)
    image = (image - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]

    # Apply sigmoid transfer to the image and normalize to [0, 1]
    image = np.exp(image * 5 - 1)
    image = image / (image + 1)

    return image


# Load the pre-trained model from the specified path
model_path = os.path.join("Model", "comcrop_udf_test.hdf5")
model = load_model(model_path, compile=False)

# TODO: Define size of patches and number of classes
#  patch_size = 64
#  n_classes = 18
#  - Add code to preprocess and process the whole image in patches of size patch_size.
#  - The model should be applied to each patch and the predictions should be combined to form the final prediction.
#  - The patches should all be exactly 64 by 64 pixels. If the image dimensions are not divisible by 64, the image
#    should be padded with zeros.
#  - The overlap between patches can be customized, default is 50%.
#  - Overlapping patches should be averaged to get the final prediction.
#  - Overlapping patches should overlap in both vertical and horizontal directions.
#  - The final prediction should be a single image with the same dimensions as the input image.

# Load and preprocess input image, for now only use 64 by 64 pixels because model is trained on this size, later
# we will implement patching!
im_in_path = os.path.join("Example_Images", "Example1.tif")
x_img = io.imread(im_in_path)[0:64, 0:64, :]

# Print shapes of the loaded data
print(f"input image: {x_img.shape}")

# Select and preprocess bands
x_img = x_img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
x_img1 = x_img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]]

x_img1 = norm(x_img1)
vv = normalise_vv(x_img[:, :, 9])
vh = normalise_vh(x_img[:, :, 10])
alt = normalise_altitude(x_img[:, :, 11])
lon = normalise_longitude(x_img[:, :, 12])
lat = normalise_latitude(x_img[:, :, 13])

# Compute additional indices
SIZE_X = x_img.shape[0]
SIZE_Y = x_img.shape[1]
swir2 = x_img1[:, :, 8]
swir1 = x_img1[:, :, 7]
nir = x_img1[:, :, 6]
red_edge3 = x_img1[:, :, 5]
red_edge2 = x_img1[:, :, 4]
red_edge1 = x_img1[:, :, 3]
red = x_img1[:, :, 2]
green = x_img1[:, :, 1]
blue = x_img1[:, :, 0]
ndvi = np.where((nir + red) == 0., 0, (nir - red) / (nir + red))
ndwi = np.where((green + nir) == 0., 0, (green - nir) / (green + nir))

# Reshape and normalize indices
ndvi = np.reshape(ndvi, (SIZE_X, SIZE_Y, 1))
ndwi = np.reshape(ndwi, (SIZE_X, SIZE_Y, 1))
vv = np.reshape(vv, (SIZE_X, SIZE_Y, 1))
vh = np.reshape(vh, (SIZE_X, SIZE_Y, 1))
alt = np.reshape(alt, (SIZE_X, SIZE_Y, 1))
lon = np.reshape(lon, (SIZE_X, SIZE_Y, 1))
lat = np.reshape(lat, (SIZE_X, SIZE_Y, 1))

# Concatenate all processed bands and indices to form the input image
image = np.concatenate((x_img1, ndvi, vv, vh, alt, lon, lat), axis=2)
# Replace NaNs with zero
image = np.nan_to_num(image)
# Reshape the image to match the model input shape (1, 64, 64, 15)
image = np.reshape(image, (1, 64, 64, 15))

print(f"image going into model: {image.shape}")

# Predict using the pre-trained model and the preprocessed image
predictions_smooth = model.predict(image)

# Get the final predicted classes by taking the argmax of the predictions
final_prediction = np.argmax(predictions_smooth, axis=-1)

def show_prediction(final_prediction):
    """
    Visualize the final classified image.

    This function ensures that the final prediction is a 2D array and then
    visualizes it using a color map.

    :param final_prediction: The final predicted classes as a NumPy array.
    """
    # Ensure final_prediction is a 2D array
    if final_prediction.ndim == 3 and final_prediction.shape[0] == 1:
        final_prediction = final_prediction.squeeze(axis=0)
    elif final_prediction.ndim == 1:
        final_prediction = final_prediction.reshape((int(np.sqrt(final_prediction.size)), -1))

    # Visualize the final predicted image
    plt.imshow(final_prediction, cmap='viridis')
    plt.colorbar()
    plt.title(f"{model_path} prediction")
    plt.show()

show_prediction(final_prediction)

# Save the final predicted image
io.imsave(r'prediction.tif', final_prediction.astype(np.uint8))
