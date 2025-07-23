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


def Attention_UNetFusion3I_Sentinel2_Binary(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    channels = tf.unstack (inputs, num=17, axis=-1)
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9], channels[10], channels[11]], axis=-1)
    
    inputs2  = tf.stack ([channels[12], channels[13]], axis=-1)
    
    inputs3  = tf.stack ([channels[14], channels[15], channels[16]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 3
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 4, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 3
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 4, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3 - locational enconding altitude, longitude, and latitude
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 3
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 4, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)

    # UpRes 8
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multiclass commodity crops

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

def Attention_UNetFusion3I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Comm Crop Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    channels = tf.unstack (inputs, num=17, axis=-1)
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9], channels[10], channels[11]], axis=-1)
    
    inputs2  = tf.stack ([channels[12], channels[13]], axis=-1)
    
    inputs3  = tf.stack ([channels[14], channels[15], channels[16]], axis=-1)

    ############################MODEL BLOCK 1 SENTINEL -2, input 1############################
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 3
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 4, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 5, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 6
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 7
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2 SENTINEL -1, input 2############################

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 3
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 4, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 5, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 6
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 7
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     ############################MODEL BLOCK 3 LOCATIONAL DATA (Latitude, longitude, elevation), input 3############################

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 3
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 4, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 5, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 6
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model