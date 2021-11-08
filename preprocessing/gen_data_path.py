import shutil

import yaml
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_poses, load_calib, load_files, load_vertex
from preprocessing.utils import *
from example.laserscan import *

# data_path = '../data/sequences/08/velodyne/000030.bin'
# label_path = '../data/sequences/08/labels/000030.label'

CFG = yaml.safe_load(open('../config/semantic-kitti-mos.yaml', 'r'))

config_filename = '../config/data_loader.yaml'
if len(sys.argv) > 1:
    config_filename = sys.argv[1]

if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
else:
    config = yaml.load(open(config_filename))

path_folder = config['path_folder']
train_folder = config['train_folder']
valid_folder = config['valid_folder']
test_folder = config['test_folder']


train_seq = CFG['split']['train']
valid_seq = CFG['split']['valid']
test_seq = CFG['split']['test']

if os.path.isdir(path_folder):
    shutil.rmtree(path_folder)
os.mkdir(path_folder)
os.mkdir(train_folder)
os.mkdir(valid_folder)
os.mkdir(test_folder)

counter = 0

0

# save data path as a pickle dictionary

