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




    residual_visualization_folder_ns = []
    residual_images_folder_ns = []

    # projection folder
    range_images_folder = os.path.join(sequence_folder,'gr_range_images_folder')# gr

    if os.path.isdir(range_images_folder):
        shutil.rmtree(range_images_folder)
    os.mkdir(range_images_folder)

    for num_last_n in num_last_ns:
        residual_visualization_folder_n = os.path.join(sequence_folder, 'gr_residual_visualization_'+str(num_last_n))
        if os.path.isdir(residual_visualization_folder_n):
            shutil.rmtree(residual_visualization_folder_n)
        os.mkdir(residual_visualization_folder_n)
        residual_visualization_folder_ns.append(residual_visualization_folder_n)

        residual_image_folder_n = os.path.join(sequence_folder, 'gr_residual_'
                                                                'images_'+str(num_last_n))
        if os.path.isdir(residual_image_folder_n):
            shutil.rmtree(residual_image_folder_n)
        os.mkdir(residual_image_folder_n)
        residual_images_folder_ns.append(residual_image_folder_n)




    # specify the output folders




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
        gr_clustering_dict = dict()
        current_pose = poses[frame_idx]

        sem_scan_current_frame.open_scan(scan_paths[frame_idx])
        points = sem_scan_current_frame.points
        points = points * np.array([1, 1, -1])
        points_non_ground = process(points)

        gr_points = points[process.segments_index]* np.array([1, 1, -1])
        gr_remission = sem_scan_current_frame.remissions[process.segments_index]

        # do range projection
        sem_scan_current_frame.set_points(gr_points, remissions=gr_remission, gt_idx=process.segments_index)
        sem_scan_current_frame.do_range_projection()
        mask = sem_scan_current_frame.proj_mask

        # do the clustering
        range_img_x = sem_scan_current_frame.proj_xyz[:, :, 0] * mask
        range_img_y = sem_scan_current_frame.proj_xyz[:, :, 1] * mask
        range_img_z = sem_scan_current_frame.proj_xyz[:, :, 2] * mask
        instance_label = cluster.ScanLineRun_cluster(range_img_x, range_img_y, range_img_z, mask,
                                                     sem_scan_current_frame.proj_H,
                                                     sem_scan_current_frame.proj_W)
        instance_label = np.array(instance_label)


        gr_clustering_dict['proj_xyz'] = sem_scan_current_frame.proj_xyz
        gr_clustering_dict['remissions'] = sem_scan_current_frame.remissions
        gr_clustering_dict['proj_range'] = sem_scan_current_frame.proj_range
        gr_clustering_dict['mask'] = sem_scan_current_frame.proj_mask
        gr_clustering_dict['proj_index'] = sem_scan_current_frame.proj_idx_gt


        # will use for substraction
        current_proj_vertex = sem_scan_current_frame.proj_xyz
        current_range = sem_scan_current_frame.proj_range


        range_image_file_name = os.path.join(range_images_folder, str(frame_idx).zfill(6))
        with open(range_image_file_name, 'wb') as f:
            pickle.dump(gr_clustering_dict, f)

        # np.save(range_image_file_name, current_proj_vertex)

        for n_index, num_last_n in enumerate(num_last_ns):

            file_name = os.path.join(residual_images_folder_ns[n_index], str(frame_idx).zfill(6))
            diff_image = np.full((range_image_params['height'], range_image_params['width']), 0,
                                 dtype=np.float32)  # [H,W] range (0 is no data)

        # load current scan and generate current range image

        # for the first N frame we generate a dummy file
            if frame_idx < num_last_n:
                np.save(file_name, diff_image)

                if visualize:
                    fig = plt.figure(frameon=False, figsize=(16, 10))
                    fig.set_size_inches(20.48, 0.64)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(diff_image, vmin=0, vmax=1)
                    image_name = os.path.join(residual_visualization_folder_ns[n_index], str(frame_idx).zfill(6))
                    plt.savefig(image_name)
                    plt.close()

            else:


                # load last scan, transform into the current coord and generate a transformed last range image
                last_pose = poses[frame_idx - num_last_n]

                current_pose = poses[frame_idx]

                sem_scan_last_frame.open_scan(scan_paths[frame_idx - num_last_n])
                # transformation
                last_vertex = np.ones((sem_scan_last_frame.points.shape[0], sem_scan_last_frame.points.shape[1] + 1))
                last_vertex[:, :-1] = sem_scan_last_frame.points
                sem_scan_last_frame.points = np.linalg.inv(current_pose).dot(last_pose).dot(last_vertex.T).T[:,:-1]

                # sem_scan_last_frame.points = np.linalg.inv(current_pose).dot(last_pose).dot(sem_scan_last_frame.points.T).T

                points = sem_scan_last_frame.points
                points = points * np.array([1, 1, -1])
                points_non_ground = process(points)

                gr_points = points[process.segments_index] * np.array([1, 1, -1])
                gr_remission = sem_scan_last_frame.remissions[process.segments_index]


                # do range projection
                sem_scan_last_frame.set_points(gr_points, remissions=gr_remission, gt_idx=process.segments_index)
                sem_scan_last_frame.do_range_projection()




                # last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T


                last_range_transformed = sem_scan_last_frame.proj_range

                # generate residual image
                valid_mask = (current_range > range_image_params['min_range']) & \
                             (current_range < range_image_params['max_range']) & \
                             (last_range_transformed > range_image_params['min_range']) & \
                             (last_range_transformed < range_image_params['max_range'])
                difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask])

                if normalize:
                    difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / current_range[
                        valid_mask]

                diff_image[valid_mask] = difference

                if debug:
                    fig, axs = plt.subplots(3)
                    axs[0].imshow(last_range_transformed)
                    axs[1].imshow(current_range)
                    axs[2].imshow(diff_image, vmin=0, vmax=10)
                    plt.show()

                if visualize:
                    fig = plt.figure(frameon=False, figsize=(16, 10))
                    fig.set_size_inches(20.48, 0.64)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(diff_image, vmin=0, vmax=1)
                    image_name = os.path.join(residual_visualization_folder_ns[n_index], str(frame_idx).zfill(6))
                    plt.savefig(image_name)
                    plt.close()

                # save residual image
                np.save(file_name, diff_image)