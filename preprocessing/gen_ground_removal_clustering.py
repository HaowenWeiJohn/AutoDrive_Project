import cv2
import yaml
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from PC_cluster.FGS import lidar_projection
from PC_cluster.FGS.ground_removal import Processor
from utils import load_poses, load_calib, load_files, load_vertex
from preprocessing.utils import *
from example.laserscan import *
from PC_cluster.ScanLineRun_cluster.build import ScanLineRun_Cluster

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
data_folder = config['data_folder']
debug = config['debug']
visualize = config['visualize']
range_image_params = config['range_image']
sequences = config['sequences']

sem_scan = LaserScan(project=False,
                     flip_sign=False,
                     H=range_image_params['height'],
                     W=range_image_params['width'],
                     fov_up=range_image_params['fov_up'],
                     fov_down=range_image_params['fov_down'])
process = Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line=0.15,
                    sensor_height=1.73, max_start_height=0.5, long_threshold=8)

cluster=ScanLineRun_Cluster.ScanLineRun_Cluster(0.5, 1)

# create mask folder
for sequence in sequences:

    sequence_folder = os.path.join(data_folder, sequence)

    visualization_folder = config['visualization_folder']
    scan_folder = config['scan_folder']
    label_folder = config['label_folder']
    mask_image_folder = config['mask_image_folder']

    visualization_folder = os.path.join(sequence_folder, visualization_folder)
    scan_folder = os.path.join(sequence_folder, scan_folder)
    label_folder = os.path.join(sequence_folder, label_folder)
    mask_image_folder = os.path.join(sequence_folder, mask_image_folder)


    # if not os.path.exists(mask_image_folder):
    #     os.makedirs(mask_image_folder)
    #
    # # create mask image visualization folder
    # if visualize:
    #     if not os.path.exists(visualization_folder):
    #         os.makedirs(visualization_folder)


    # load labels
    scan_paths = load_files(scan_folder)
    # label_paths = load_files(label_folder)

    # create scan object

    # index_range = list(range(0,len(scan_paths)))
    print('Clustering:', sequence, 'Frames: ', str(len(scan_paths)))

    for frame_idx in tqdm(range(len(scan_paths))):
        result_dict = dict()

        sem_scan.open_scan(scan_paths[frame_idx])
        points = sem_scan.points
        points = points * np.array([1, 1, -1])
        points_non_ground = process(points)

        img_raw = lidar_projection.birds_eye_point_cloud(points,
                                                         side_range=(-50, 50), fwd_range=(-50, 50),
                                                         res=0.25, min_height=-2, max_height=4)

        img_non_ground = lidar_projection.birds_eye_point_cloud(points[process.segments_index],
                                                                side_range=(-50, 50), fwd_range=(-50, 50),
                                                      res=0.25, min_height=-2, max_height=4)

        gr_points = points[process.segments_index]* np.array([1, 1, -1])
        gr_remission = sem_scan.remissions[process.segments_index]



        sem_scan.set_points(gr_points, remissions=gr_remission, gt_idx=process.segments_index)
        sem_scan.do_range_projection()
        # do the clustering

        mask = sem_scan.proj_mask
        range_img_x = sem_scan.proj_xyz[:, :, 0] * mask
        range_img_y = sem_scan.proj_xyz[:, :, 1] * mask
        range_img_z = sem_scan.proj_xyz[:, :, 2] * mask
        instance_label = cluster.ScanLineRun_cluster(range_img_x, range_img_y, range_img_z, mask, sem_scan.proj_H, sem_scan.proj_W)
        instance_label = np.array(instance_label)






        # if visualize:
        #     fig = plt.figure(frameon=False, figsize=(16, 10))
        #     fig.set_size_inches(20.48, 0.64)
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     img = label_new.copy()
        #     img[img<2]=0
        #     ax.imshow(img, vmin=0, vmax=1)
        #     image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6))
        #     plt.savefig(image_name)
        #     plt.close()
        #
        # # save to npy file
        # label_new_one_hot = depth_onehot(matrix=label_new, category=[0, 1, 2], on_value=1, off_value=0, channel_first=True)
        #
        # np.save(mask_file_name, [label_new, label_new_one_hot, sem_scan.proj_idx])





