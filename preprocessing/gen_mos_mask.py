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

config_filename = '../config/mask_preparing.yaml'
if len(sys.argv) > 1:
    config_filename = sys.argv[1]

if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
else:
    config = yaml.load(open(config_filename))

# ground truth info
color_dict = CFG["color_map"]
label_transfer_dict = CFG["learning_map"]
nclasses = len(color_dict)

# mask config
debug = config['debug']
visualize = config['visualize']
visualization_folder = config['visualization_folder']
scan_folder = config['scan_folder']
label_folder = config['label_folder']
mask_image_folder = config['mask_image_folder']
range_image_params = config['range_image']

# create mask folder
if not os.path.exists(mask_image_folder):
    os.makedirs(mask_image_folder)

# create mask image visualization folder
if visualize:
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)


# load labels
scan_paths = load_files(scan_folder)
label_paths = load_files(label_folder)

# create scan object
sem_scan = SemLaserScan(nclasses=nclasses,
                        sem_color_dict=color_dict,
                        project=True,
                        flip_sign=False,
                        H=range_image_params['height'],
                        W=range_image_params['width'],
                        fov_up=range_image_params['fov_up'],
                        fov_down=range_image_params['fov_down'])


for frame_idx in tqdm(range(len(scan_paths))):
    mask_file_name = os.path.join(mask_image_folder, str(frame_idx).zfill(6))
    sem_scan.open_scan(scan_paths[frame_idx])
    sem_scan.open_label(label_paths[frame_idx])

    original_label = np.copy(sem_scan.proj_sem_label)
    label_new = sem_label_transform(original_label, label_transfer_dict=label_transfer_dict)

    if visualize:
        fig = plt.figure(frameon=False, figsize=(16, 10))
        fig.set_size_inches(20.48, 0.64)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = label_new.copy()
        img[img<2]=0
        ax.imshow(img, vmin=0, vmax=1)
        image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6))
        plt.savefig(image_name)
        plt.close()

    # save to npy file
    label_new_one_hot = depth_onehot(matrix=label_new, category=[0, 1, 2], on_value=1, off_value=0, channel_first=True)

    np.save(mask_file_name, [label_new, label_new_one_hot])





