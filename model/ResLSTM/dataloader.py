import time

import pandas as pd
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, TimeDistributed, ConvLSTM2D, \
    BatchNormalization, AveragePooling2D
import tensorflow as tf

# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel
from tensorflow.python.layers.convolutional import Conv2DTranspose
import numpy as np

from preprocessing.utils import load_files


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df=None, X_dir=None, y_dir=None, X_col='x_files', y_col='y_files',
                 batch_size=10,
                 input_size=(5,64,2048,12),
                 shuffle=True):
        if df is not None:
            self.df = df.copy()
        else:
            self.gen_df(x_dir=X_dir, y_dir=y_dir)

        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)
        # self.n_name = df[y_col['name']].nunique()
        # self.n_type = df[y_col['type']].nunique()

    def gen_df(self, x_dir, y_dir):
        x_files = load_files(x_dir)
        y_files = load_files(y_dir)

        # create_data_frame
        dict = {'x_files': x_files, 'y_files': y_files}
        self.df = pd.DataFrame(data=dict)


    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        x = np.load(path)

        current_frame = x[:, :, 0:5]

        lstm_1 = np.dstack((x[:, :, 9:13], current_frame))
        lstm_2 = np.dstack((x[:, :, 8:12], current_frame))
        lstm_3 = np.dstack((x[:, :, 7:11], current_frame))
        lstm_4 = np.dstack((x[:, :, 6:10], current_frame))
        lstm_5 = np.dstack((x[:, :, 5:9], current_frame))

        x = [lstm_1, lstm_2, lstm_3, lstm_4, lstm_5]
        x = np.array(x)

        return x

    def __get_output(self, path):
        y = np.load(path)
        # y preprocessing
        # y[y <=1] = 0
        # y[y >=1] = 1
        # y = np.expand_dims(y, axis=-1)
        # return tf.keras.utils.to_categorical(label, num_classes=num_classes)
        return y

    def __get_data(self, batches):
        print(batches)
        x_batch_files = batches[self.X_col]
        y_batch_files = batches[self.y_col]

        x_batch = np.asarray([self.__get_input(x_file) for x_file in x_batch_files])
        y_batch = np.asarray([self.__get_output(y_file) for y_file in y_batch_files])


        return x_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size



# # create data file
#
x_dir = '../../data/train_test_val/val/x'
y_dir = '../../data/train_test_val/val/y'
#
# x_files = load_files(x_dir)
# y_files = load_files(y_dir)
#
# # create_data_frame
# dict = {'x_files': x_files, 'y_files': y_files}
# df = pd.DataFrame(data=dict)

data_gen = CustomDataGen(df=None, X_dir=x_dir, y_dir=y_dir, X_col='x_files', y_col='y_files', batch_size=10, shuffle=True)
data_gen.__getitem__(index=20)
data_gen.on_epoch_end()
