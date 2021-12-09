import csv
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.training_config import root_data_dir
from model.CustomDataLoader.dataloader import BiSeNet_DataLoader, ResLSTM_DataLoader
from model.ResLSTM_torch.ResLSTM_torch import ResLSTM
from model.prediction_utils.utils import val, reverse_one_hot

class Training_Logger:
    def __init__(self, root_dir, logger_dir, model_name='best_model', over_write=True):
        self.root_dir = root_dir
        self.history_csv='history.csv'
        self.model=None
        self.loss=np.Inf
        self.logger_dir=logger_dir
        self.over_write=over_write
        self.init_logger()

    def init_logger(self):
        if os.path.isdir(self.root_dir):
            print('Root Dir: ', self.root_dir)
        else:
            print('Cannot find dir:', self.root_dir)
            exit(-1)

        self.logger_dir = os.path.join(self.root_dir, self.logger_dir)

        if os.path.isdir(self.logger_dir):
            if self.over_write:
                shutil.rmtree(self.logger_dir)
            else:
                print('Logger Dir:', self.logger_dir, ' already exist!')

        os.mkdir(self.logger_dir)

        # create csv and write
        self.history_csv = os.path.join(self.logger_dir, self.history_csv)


        with open(self.history_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_miou'])
            f.close()

    def save_model(self, model=None):
        if model:
            torch.save(model.state_dict(), os.path.join(self.logger_dir))

    def log_hist(self, epoch, train_loss, val_loss=None, val_miou=None):

        with open(self.history_csv, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_miou])

if __name__ == '__main__':
    logger = Training_Logger(root_dir='../../utils/training_logger', logger_dir='test_save')
    logger.log_hist(epoch=1, train_loss=3.4)
    logger.log_hist(epoch=1, train_loss=3.4)

