import math

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, TimeDistributed, ConvLSTM2D, \
    BatchNormalization, AveragePooling2D
import tensorflow as tf

# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel
from tensorflow.python.layers.convolutional import Conv2DTranspose

# cce = tf.keras.losses.CategoricalCrossentropy()
class_weight = [0, 9, 251]

class Custom_loss(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super().__init__()
        self.cce = tf.losses.CategoricalCrossentropy()
        self.class_weight = class_weight

    def call(self, y_true, y_pred):
        # y = (batch_size, 64, 2048, 3)

        loss = self.cce(y_true, y_pred)
        return loss

# def custom_loss(y_true, y_pred):
#     loss = cce(y_true, y_pred, sample_weight=tf.constant(class_weight))
#     return loss
