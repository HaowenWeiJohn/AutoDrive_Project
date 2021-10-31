import yaml

from preprocessing.utils import *
from example.laserscan import *

data_path = '../data/sequences/08/velodyne/000030.bin'
label_path = '../data/sequences/08/labels/000030.label'

thing_list = [1, 2, 3, 4, 5, 6, 7, 8]
CFG = yaml.safe_load(open('../config/semantic-kitti-mos.yaml', 'r'))

color_dict = CFG["color_map"]

label_transfer_dict = CFG["learning_map"]

nclasses = len(color_dict)


def sem_label_transform(raw_label_map):
    for i in label_transfer_dict.keys():
        pre_map = raw_label_map == i
        raw_label_map[pre_map] = label_transfer_dict[i]
    return raw_label_map


sem_scan =SemLaserScan(nclasses=nclasses , sem_color_dict=color_dict, project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0)

sem_scan.open_scan(data_path)
sem_scan.open_label(label_path)

original_label=np.copy(sem_scan.proj_sem_label)
label_new=sem_label_transform(original_label)



