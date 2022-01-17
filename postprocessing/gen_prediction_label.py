import shutil
import yaml
import os
import sys
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from config.data_path_config import root_data_dir
from model.CustomDataLoader.dataloader import ResLSTM_DataLoader
from model.ResLSTM_torch.ResLSTM_torch import ResLSTM
from utils.auxiliary.KNN import KNN
from utils.prediction_helper.predictor import Torch_Predictor
from utils.utils import load_poses, load_calib, load_files, load_vertex
from preprocessing.utils import *
from example.laserscan import *
from config.post_processing_config import *


# generate prediction result and KNN result

# load model
model_path = ''
data_dir = ''
save_prediction_dir = ''

# device = torch.device('cuda:0')
# model = ResLSTM(3).to(device=device)
# model.load_state_dict(torch.load(model_state_dict_path))
# model.eval()


# init knn
knn_params={'knn': 5, 'search': 5, 'sigma': 1.0, 'cutoff': 1.0}
post_knn = KNN(knn_params, 20)

# get files in the test dir


dataset = ResLSTM_DataLoader(data_dir=data_dir, label_dir=None)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

predictor = Torch_Predictor(model=ResLSTM(nclasses=3),
                            model_path=model_path,
                            data_loader=dataloader,
                            save_dir=save_prediction_dir,
                            num_class=3, use_gpu=True)

predictor.predict()



# for data_file in data_files:
#     model_input = np.load(data_file)
#     output =

# analysis_sequence=['11','12','13','14','15','16','17','18','19','20','21']
#
# for sequence in analysis_sequence:
#
#     sequence_dir = os.path.join(root_data_dir, 'dataset', 'sequences')
#     for data in

    # create label folder in target test folder
    # for rot

    # for sequence in this sequence we can get more information











