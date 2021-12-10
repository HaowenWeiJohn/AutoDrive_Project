import os
import shutil
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_poses, load_calib, load_files, load_vertex

try:
    from c_gen_virtual_scan import gen_virtual_scan as range_projection
except:
    print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
    print("Currently using python-lib to generate range images.")
    from utils import range_projection

# sequence_folder = '../data/sequences'
# data_folder = '../data/train_test_val'

sequence_folder = '/media/server-ak209/ROS/hwei/data/dataset/sequences'
data_folder = '/media/server-ak209/ROS/hwei/data/train_test_val'

folder_name = 'val'

# the data structure will be range image + res_image_1_2_3_4_5_6_7

sequences = [ '08'
    # '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
]



for sequence in sequences:
    # sample_index = 0
    this_sequence_folder = os.path.join(sequence_folder, sequence)

    samples_folder = os.path.join(this_sequence_folder, 'mask_image')
    sample_file_names = os.listdir(samples_folder)
    sample_file_names.sort()

    for sample_file_name in tqdm(sample_file_names):

        samples_file_path = os.path.join(samples_folder, sample_file_name)
        copy_file_name = os.path.join(data_folder, folder_name, 'y', sequence+'_'+sample_file_name)
        shutil.copyfile(samples_file_path, copy_file_name)



