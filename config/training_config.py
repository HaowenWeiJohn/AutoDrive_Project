import os
import platform

os_type = platform.system()

root_data_dir = ''


if os_type == "Windows":
    root_data_dir = 'C:/Users/Haowe/PycharmProjects/AutoDrive_Project/data'
    root_save_dir = 'C:/Users/Haowe/PycharmProjects/AutoDrive_Project/training_save'
else:
    root_data_dir = '/media/server-ak209/ROS/hwei/data'
    root_save_dir = '/media/server-ak209/ROS/hwei/AutoDrive_Project/training_save'










