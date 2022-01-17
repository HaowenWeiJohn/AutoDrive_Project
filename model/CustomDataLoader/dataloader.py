# import __init__ as booger
import os
import sys
import time

import numpy as np

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.utils import load_files


class ResLSTM_DataLoader(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

        self.data_files = load_files(data_dir)
        if label_dir:
            self.label_files = load_files(label_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # x = np.array([1,2,3]) # read index
        # y = np.array([1,2,3]) # read index
        # y_mask = np.array([1,2,3])
        x = np.load(self.data_files[index])
        x = self.input_preprocessing(x)

        if self.label_dir:
            y = np.load(self.label_files[index], allow_pickle=True)
            # current = time.time()
            y_label = y[0]
            y_onehot = y[1]
            # y_mask = torch.from_numpy(y_mask)
            # print('use: ', str(time.time()-current))
            return torch.from_numpy(x), torch.from_numpy(y_label).to(dtype=torch.long)

        else:
            return torch.from_numpy(x)

        # if self.transform

    def input_preprocessing(self, x):
        current_frame = x[0:5, :, :]

        lstm_1 = np.append(current_frame, x[9:13, :, :], axis=0)
        lstm_2 = np.append(current_frame, x[8:12, :, :], axis=0)
        lstm_3 = np.append(current_frame, x[7:11, :, :], axis=0)
        lstm_4 = np.append(current_frame, x[6:10, :, :], axis=0)
        lstm_5 = np.append(current_frame, x[5:9, :, :], axis=0)

        x = [lstm_1, lstm_2, lstm_3, lstm_4, lstm_5]
        x = np.array(x)

        return x


class BiSeNet_DataLoader(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

        self.data_files = load_files(data_dir)
        self.label_files = load_files(label_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # current_time = time.time()
        # print(index)
        # x = np.array([1,2,3]) # read index
        # y = np.array([1,2,3]) # read index
        # y_mask = np.array([1,2,3])
        x = np.load(self.data_files[index])
        # x = self.input_preprocessing(x)
        y = np.load(self.label_files[index], allow_pickle=True)
        y_label = y[0]
        y_onehot = y[1]
        # print(time.time()-current_time)
        # # y_mask = torch.from_numpy(y_mask)

        return torch.from_numpy(x), torch.from_numpy(y_label).to(dtype=torch.long)

        # if self.transform

    # def input_preprocessing(self, x):
    #     current_frame = x[0:5, :, :]
    #
    #     lstm_1 = np.append(current_frame, x[9:13, :, :], axis=0)
    #     lstm_2 = np.append(current_frame, x[8:12, :, :], axis=0)
    #     lstm_3 = np.append(current_frame, x[7:11, :, :], axis=0)
    #     lstm_4 = np.append(current_frame, x[6:10, :, :], axis=0)
    #     lstm_5 = np.append(current_frame, x[5:9 , :, :], axis=0)
    #
    #     x = [lstm_1, lstm_2, lstm_3, lstm_4, lstm_5]
    #     x = np.array(x)
    #
    #     return x


