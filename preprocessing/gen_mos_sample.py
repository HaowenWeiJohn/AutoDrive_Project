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

sequence_folder = '../data/sequences'
data_folder = '../data/train_test_val'
folder_name = 'val'

# the data structure will be range image + res_image_1_2_3_4_5_6_7

sequences = ['08']
back_track_ns = [1,2,3,4,5,6,7,8]

sample_index = 0

for sequence in sequences:
    sequence_folder = os.path.join(sequence_folder, sequence)

    samples_folder = os.path.join(sequence_folder, 'range_images_folder')
    samples_file_name = os.listdir(samples_folder)



    for sample_file_name in tqdm(samples_file_name):
        range_image_path = os.path.join(samples_folder, sample_file_name)
        sample = np.load(range_image_path)
        sample = np.moveaxis(sample, -1, 0)
        x_folder = os.path.join(folder_name, 'x')
        sample_save_file_name = os.path.join(data_folder, folder_name, sample_file_name)
        # res images
        for back_track_n in back_track_ns:
            res_image_path = os.path.join(sequence_folder, 'residual_images_'+str(back_track_n), sample_file_name)
            res_image = np.load(res_image_path)
            sample = np.append(sample, np.expand_dims(res_image, 0))

        np.save(sample_save_file_name, sample)
# test_sequences = [8]
# val_sequences = [8]



