# author: Haowen Wei

import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder
import os
import tensorflow as tf













category = [0, 1, 2]

raw_label = np.array([
    [0,1,2],
    [1,1,1],
    [2,1,0],
    [2,1,1]
])

encoder = OneHotEncoder(categories='auto')
encoder.fit(np.reshape(category, (-1, 1)))

a = tf.one_hot(
    raw_label,#your image with label
    3, #the number of classes
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)

