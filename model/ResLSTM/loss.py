import math

import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, TimeDistributed, ConvLSTM2D, \
    BatchNormalization, AveragePooling2D
import tensorflow as tf

# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel
from tensorflow.python.layers.convolutional import Conv2DTranspose

# cce = tf.keras.losses.CategoricalCrossentropy()
from model.ResLSTM.lovasz_losses_tf import lovasz_softmax

class_weight = [0, 9, 251]

class Custom_loss(tf.keras.losses.Loss):
    def __init__(self, class_weight):
        super().__init__()
        self.cce = tf.losses.CategoricalCrossentropy()
        self.class_weight = np.sqrt(class_weight)


    def call(self, y_true, y_pred):
        # y = (batch_size, 64, 2048, 3)
        # print(y_pred.numpy())
        flatten = tf.reshape(y_pred, (-1, y_pred.shape[-1]))
        # loss = self.cce(y_true, y_pred)
        tensor_1 = -tf.reduce_sum(class_weight*flatten*tf.math.log(y_pred))
        tensor_2 = tf.cast(tf.size(y_pred), dtype=dtypes.float32)
        # print('pred:', y_pred)
        # print('tesnor_2:', tensor_2)
        weighted_categorical_cross_entropy = tf.divide(tensor_1, tensor_2)

        lovasz = lovasz_softmax(probas=y_pred, labels=y_true, classes='present', per_image=False, ignore=None, order='BHWC')

        loss = weighted_categorical_cross_entropy + lovasz

        return loss



# def custom_loss(y_true, y_pred):
#     loss = cce(y_true, y_pred, sample_weight=tf.constant(class_weight))
#     return loss
