# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel


import os.path
import sys

import numpy as np
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')
from model.ResLSTM.ResLSTM_tf import *
from model.ResLSTM.dataloader import *

# train_data = np.random.rand(100, 8, 64, 2048, 10)
# val_data = np.random.rand(10, 8, 64, 2048, 10)
# train_output = np.random.rand(100, 64, 2048, 1)
# val_output = np.random.rand(10, 64, 2048, 1)

# train_data = np.ones((100, 5, 64, 2048, 10))
# val_data = np.ones((10, 5, 64, 2048, 10))
# train_output = np.ones((100, 64, 2048, 1), dtype=int)
# val_output = np.ones((10, 64, 2048, 1), dtype=int)
# prepare sudo data

model = ResLSTM()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

# transfer model csv log path
csv_logger = CSVLogger(filename='train_save/history.csv',
                       append=True)
# save model path

mc = ModelCheckpoint(
    filepath='train_save/test_model.h5',
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# make data generator
x_train_dir = '../../data/pesudo_train_test_val/val/x'
y_train_dir = '../../data/pesudo_train_test_val/val/y'
x_val_dir = '../../data/pesudo_train_test_val/val/x'
y_val_dir = '../../data/pesudo_train_test_val/val/y'

#
# x_files = load_files(x_dir)
# y_files = load_files(y_dir)
#
# # create_data_frame
# dict = {'x_files': x_files, 'y_files': y_files}
# df = pd.DataFrame(data=dict)

train_data_gen = CustomDataGen(df=None, X_dir=x_train_dir, y_dir=y_train_dir, X_col='x_files', y_col='y_files',
                               batch_size=8, shuffle=True)
val_data_gen = CustomDataGen(df=None, X_dir=x_val_dir, y_dir=y_val_dir, X_col='x_files', y_col='y_files', batch_size=8,
                             shuffle=True)
# data_gen.__getitem__(index=20)
# data_gen.on_epoch_end()


history = model.fit(
    train_data_gen,
    validation_data=val_data_gen,
    epochs=1000,
    # batch_size=1,
    # validation_batch_size=1,
    callbacks=[es, mc, csv_logger],
    verbose=1,
    # shuffle=True
)
