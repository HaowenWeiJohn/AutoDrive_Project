# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel
import os.path
import sys

import numpy as np
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from ResLSTM_tf import *

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')


train_data = np.random.rand(100, 8, 64, 2048, 10)
val_data = np.random.rand(10, 8, 64, 2048, 10)
train_output = np.random.rand(100, 64, 2048, 1)
val_output = np.random.rand(10, 64, 2048, 1)
# prepare sudo data

model = ResLSTM()


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

# transfer model csv log path
csv_logger = CSVLogger(filename='test_log',
                       append=True)
# save model path

mc = ModelCheckpoint(
    filepath='test_model',
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
#

history = model.fit(train_data, train_output,
                             validation_data=(
                                 val_data, val_output),
                             epochs=1000,
                             batch_size=1,
                             validation_batch_size=1,
                             callbacks=[es, mc, csv_logger],
                             verbose=1, shuffle=True)





