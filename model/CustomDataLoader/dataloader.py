
# import __init__ as booger
import os
import sys
sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.utils import load_files



class Custom_DataLoader(Dataset):
    def __init__(self, data_dir, label_dir, transform = None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

        self.data_files = load_files(data_dir)
        self.label_files = load_files(label_dir)


    def __len__(self):
        return len(self.data_files)


    def __getitem__(self, index):

        x = 1 # read index
        y = 2 # read index
        y_mask = 3

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        y_mask = torch.from_numpy(y_mask)

        return x, y, y_mask


        # if self.transform