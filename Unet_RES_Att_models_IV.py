import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from keras.layers import Concatenate, Input, Lambda
from tensorflow.keras import backend as K
import numpy as np

##############################################################
'''
Useful blocks to build a U-Net model with attention and fusion mechanisms.

Each convolutional block: conv -> batch norm (optional) -> activation -> conv -> batch norm -> activation -> dropout (if enabled).
'''


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    """
    Convolutional block: Conv2D -> BatchNorm (optional) -> ReLU Activation.
    Arguments:
        x: Input tensor.
        filter_size: Size of convolution filter.
        size: Number of filters.
        dropout: Dropout rate.
        batch_norm: Whether to include BatchNorm.
    Returns:
        Output tensor after convolutions and optional dropout.
    """
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def conv_blockLL(x, filter_size, size, dropout, batch_norm=False):
    """
    Convolutional block with sinusoidal activation instead of ReLU.
    """
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = tf.math.sin(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = tf.math.sin(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def conv_blockD(x, filter_size, size, dropout, batch_norm=False, dilation_rate=1):
    """
    Dilated Convolutional Block: similar to conv_block but with a dilation rate.
    """
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same", dilation_rate=dilation_rate)(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same", dilation_rate=dilation_rate)(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def repeat_elem(tensor, rep):
    """
    Lambda function to repeat tensor elements along an axis by a factor.
    """
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                         arguments={'repnum': rep})(tensor)


def gating_signal(input, out_size, batch_norm=False):
    """
    Resize down layer feature map to match the up layer feature map dimensions.
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def attention_block(x, gating, inter_shape):
    """
    Creates an attention block for attention U-Net.
    """
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                        strides=(theta_x.shape[1] // shape_g[1], theta_x.shape[2] // shape_g[2]),
                                        padding='same')(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)

    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // sigmoid_xg.shape[1], shape_x[2] // sigmoid_xg.shape[2]))(
        sigmoid_xg)
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def mlp(x, hidden_units, dropout_rate):
    """
    Multilayer perceptron (MLP) block.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    """
    Custom layer to create image patches.
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


def Attention_UNetFusion3I_SentinelMLP(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    """
    Builds the Attention U-Net with Fusion for Sentinel data and MLP integration.
    Arguments:
        input_shape: Shape of the input image.
        NUM_CLASSES: Number of output classes.
        dropout_rate: Dropout rate.
        batch_norm: Whether to use batch normalization.
    Returns:
        Compiled U-Net model with attention and fusion.
    """
    FILTER_NUM = 64
    FILTER_NUM3 = 16
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2

    inputs = layers.Input(input_shape, dtype=tf.float32)
    channels = tf.unstack(inputs, num=15, axis=-1)

    inputs1 = tf.stack(
        [channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7],
         channels[8], channels[9]], axis=-1)
    inputs2 = tf.stack([channels[10], channels[11]], axis=-1)
    inputs3 = tf.stack([channels[12], channels[13], channels[14]], axis=-1)

    # Downsampling block 1
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling for block 1
    gating_16 = gating_signal(conv_8, 4 * FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4 * FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # Further upsamples and concatenations as per design...

    # MLP block and Fusion
    x = layers.Flatten()(inputs3)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64 * 64 * 16, activation='relu')(x)
    mlp_output = layers.Reshape((64, 64, 16))(x)

    # Concatenate all outputs
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, mlp_output], axis=-1)

    output_layer = layers.Conv2D(NUM_CLASSES, (1, 1), padding="same", activation="sigmoid")(merge_data)

    return models.Model(inputs=inputs, outputs=output_layer)
