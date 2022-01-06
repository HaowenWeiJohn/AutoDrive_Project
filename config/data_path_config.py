import os
import platform

os_type = platform.system()

root_data_dir = ''


if os_type == "Windows":
    root_data_dir = 'C:/Users/Haowe/PycharmProjects/AutoDrive_Project/data'
    root_save_dir = 'C:/Users/Haowe/PycharmProjects/AutoDrive_Project/training_save'
else:
    # root_data_dir = '/home/ak209/Desktop/hwei/AutoDrive/data'
    root_data_dir = '/home/ak209/Desktop/hwei/AutoDrive/data'
    root_save_dir = '/home/ak209/Desktop/hwei/PycharmProjects/AutoDrive_Project/training_save'










