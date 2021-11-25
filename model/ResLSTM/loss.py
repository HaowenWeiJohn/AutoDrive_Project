from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, TimeDistributed, ConvLSTM2D, \
    BatchNormalization, AveragePooling2D
import tensorflow as tf

# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel
from tensorflow.python.layers.convolutional import Conv2DTranspose



# tf.keras.losses.BinaryCrossentropy
# def loss():
#     pass