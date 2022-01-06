import shutil

import yaml
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.data_path_config import root_data_dir
from utils.auxiliary.KNN import KNN
from utils.utils import load_poses, load_calib, load_files, load_vertex
from preprocessing.utils import *
from example.laserscan import *
from config.post_processing_config import *


# generate prediction result and KNN result



knn_params={'knn': 5, 'search': 5, 'sigma': 1.0, 'cutoff': 1.0}
post_knn = KNN(knn_params, 20)

analysis_sequence=['11','12','13','14','15','16','17','18','19','20','21']

for sequence in analysis_sequence:

    sequence_dir = os.path.join(root_data_dir, 'dataset', 'sequences')

    # create label folder in target test folder
    # for rot

    # for sequence in this sequence we can get more information











