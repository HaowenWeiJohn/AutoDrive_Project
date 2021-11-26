import math

import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, TimeDistributed, ConvLSTM2D, \
    BatchNormalization, AveragePooling2D
import tensorflow as tf


class MOS_IoU(tf.keras.metrics.Metric):

  def __init__(self, name='mos_iou', ignore_class=0, **kwargs):
    super(MOS_IoU, self).__init__(name=name, **kwargs)
    self.mos_iou = self.add_weight(name='mos_iou', initializer='zeros')
    mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true_flatten = tf.reshape(y_true, (-1, y_true.shape[-1]))
    y_pred_flatten = tf.reshape(y_pred, (-1, y_pred.shape[-1]))

    y_true_index = tf.math.argmax(y_true_flatten, axis=-1)
    y_pred_index = tf.math.argmax(y_pred_flatten, axis=-1)

    # [0, 1, 2]

    # y_true = tf.cast(y_true, tf.bool)
    # y_pred = tf.cast(y_pred, tf.bool)
    #
    # values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    # values = tf.cast(values, self.dtype)
    # if sample_weight is not None:
    #   sample_weight = tf.cast(sample_weight, self.dtype)
    #   sample_weight = tf.broadcast_to(sample_weight, values.shape)
    #   values = tf.multiply(values, sample_weight)

    self.m.update_state(y_true_index, y_pred_index)
    mos_iou = self.m.result().numpy()
    self.mos_iou.assign_add(mos_iou)

  def result(self):
    return self.mos_iou


# tf.keras.metrics.MeanIoU(
#     num_classes, name=None, dtype=None
# )