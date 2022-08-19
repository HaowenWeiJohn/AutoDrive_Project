#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates residual images

import os
import pickle
import shutil
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import yaml
import os
import sys
import yaml
from tqdm import tqdm
from PC_cluster.FGS import lidar_projection
from PC_cluster.FGS.ground_removal import Processor
from utils import load_poses, load_calib, load_files, load_vertex
from preprocessing.utils import *
from example.laserscan import *
from PC_cluster.ScanLineRun_cluster.build import ScanLineRun_Cluster
from utils import load_poses, load_calib, load_files, load_vertex

try:
    from c_gen_virtual_scan import gen_virtual_scan as range_projection
except:
    print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
    print("Currently using python-lib to generate range images.")
    from utils import range_projection

# if __name__ == '__main__':
# load config file
config_filename = '../config/data_preparing.yaml'
if len(sys.argv) > 1:
    config_filename = sys.argv[1]

if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
else:
    config = yaml.load(open(config_filename))


# specify parameters
num_frames = config['num_frames']
debug = config['debug']
normalize = config['normalize']
num_last_ns = config['num_last_ns']
visualize = config['visualize']
data_folder = config['data_folder']
range_image_params = config['range_image']
sequences = config['sequences']


sem_scan_current_frame = LaserScan(project=False,
                     flip_sign=False,
                     H=range_image_params['height'],
                     W=range_image_params['width'],
                     fov_up=range_image_params['fov_up'],
                     fov_down=range_image_params['fov_down'])
sem_scan_last_frame = LaserScan(project=False,
                     flip_sign=False,
                     H=range_image_params['height'],
                     W=range_image_params['width'],
                     fov_up=range_image_params['fov_up'],
                     fov_down=range_image_params['fov_down'])

process = Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line=0.15,
                    sensor_height=1.73, max_start_height=0.5, long_threshold=8)

cluster=ScanLineRun_Cluster.ScanLineRun_Cluster(0.5, 1)




for sequence in sequences:
    sequence_folder = os.path.join(data_folder, sequence)

    current_frame_bird_view_visualization_folder = os.path.join(sequence_folder,'current_frame_bird_view_visualization_folder')
    current_frame_bird_view_gr_visualization = os.path.join(sequence_folder,'current_frame_gr_bird_view_visualization')

    if os.path.isdir(current_frame_bird_view_visualization_folder):
        shutil.rmtree(current_frame_bird_view_visualization_folder)
    os.mkdir(current_frame_bird_view_visualization_folder)


    if os.path.isdir(current_frame_bird_view_gr_visualization):
        shutil.rmtree(current_frame_bird_view_gr_visualization)
    os.mkdir(current_frame_bird_view_gr_visualization)








    # load poses
    pose_file = config['pose_file']
    pose_file = os.path.join(sequence_folder, pose_file)

    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    calib_file = config['calib_file']
    calib_file = os.path.join(sequence_folder, calib_file)

    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # load LiDAR scans
    scan_folder = config['scan_folder']
    scan_folder = os.path.join(sequence_folder, scan_folder)

    scan_paths = load_files(scan_folder)

    # test for the first N scans
    if num_frames >= len(poses) or num_frames <= 0:
        print('generate training data for all frames with number of: ', len(poses))
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    range_image_params = config['range_image']

    # generate residual images for the whole sequence
    for frame_idx in tqdm(range(len(scan_paths))):
        process = Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line=0.15,
                            sensor_height=1.73, max_start_height=0.5, long_threshold=8)
        gr_clustering_dict = dict()
        current_pose = poses[frame_idx]

        sem_scan_current_frame.open_scan(scan_paths[frame_idx])
        points = sem_scan_current_frame.points
        points = points * np.array([1, 1, -1])
        points_non_ground = process(points)

        current_bird_view = lidar_projection.birds_eye_point_cloud(points,
                                                                       side_range=(-50, 50), fwd_range=(-50, 50),
                                                                       res=0.25, min_height=1, max_height=4)


        current_bird_view_gr = lidar_projection.birds_eye_point_cloud(points_non_ground,
                                                                       side_range=(-50, 50), fwd_range=(-50, 50),
                                                                       res=0.25, min_height=1, max_height=4)


        cv2.imwrite(os.path.join(current_frame_bird_view_visualization_folder, str(frame_idx).zfill(6))+'.png', current_bird_view)
        cv2.imwrite(os.path.join(current_frame_bird_view_gr_visualization, str(frame_idx).zfill(6))+'.png', current_bird_view_gr)


