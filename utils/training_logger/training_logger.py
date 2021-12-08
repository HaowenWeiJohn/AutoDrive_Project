import os

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
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.training_history='training_history.csv'
        self.validation_history='val_history.csv'
        self.model=None
        self.loss=np.Inf



    # def init_csv(self):

