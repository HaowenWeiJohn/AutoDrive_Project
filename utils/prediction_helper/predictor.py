import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.data_path_config import root_data_dir
from model.CustomDataLoader.dataloader import BiSeNet_DataLoader, ResLSTM_DataLoader
from model.ResLSTM_torch.ResLSTM_torch import ResLSTM
from model.prediction_utils.utils import val, reverse_one_hot, compute_global_accuracy, fast_hist


class Torch_Predictor:

    def __init__(self,model, model_path, data_loader, save_dir, num_class=3, visualize_dir=None, test=False, use_gpu=True):
        # self.model = model
        self.model = model
        self.model_path = model_path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.num_class=num_class
        self.data_loader = data_loader
        self.use_gpu=use_gpu
        self.save_dir = save_dir
        self.visualize_dir=visualize_dir
        if use_gpu:
            self.model.cuda()


    def validation(self):
        with torch.no_grad():
            precision_record = []
            hist = np.zeros((self.num_class, self.num_class))
            for i, (data, label) in enumerate(self.data_loader):
                print(i)
                if torch.cuda.is_available() and self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()

                predict = self.model(data).squeeze()
                predict = reverse_one_hot(predict)
                predict = np.array(predict.cpu().numpy())

                label = label.squeeze()
                label = np.array(label.cpu().numpy())

                precision = compute_global_accuracy(predict, label)
                hist += fast_hist(label.flatten(), predict.flatten(), self.num_class)
                precision_record.append(precision)


                # save predict image
                if self.visualize_dir:
                    fig = plt.figure(frameon=False, figsize=(16, 10))
                    fig.set_size_inches(20.48, 0.64)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    img = predict.copy()
                    img[img < 2] = 0
                    ax.imshow(img, vmin=0, vmax=1)
                    image_name = os.path.join(self.visualize_dir, str(i).zfill(6))
                    plt.savefig(image_name)
                    plt.close()

    def test(self):
        with torch.no_grad():
            for data, file_name in tqdm(self.data_loader):

                if torch.cuda.is_available() and self.use_gpu:
                    data = data.cuda()

                predict = self.model(data).squeeze()
                predict = reverse_one_hot(predict)
                predict = np.array(predict.cpu().numpy())
                prediction_file_name = os.path.join(self.save_dir, os.path.basename(file_name[0]))
                np.save(prediction_file_name, predict)

                # save predict image
                if self.visualize_dir:
                    fig = plt.figure(frameon=False, figsize=(16, 10))
                    fig.set_size_inches(20.48, 0.64)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    img = predict.copy()
                    img[img < 2] = 0
                    ax.imshow(img, vmin=0, vmax=1)
                    image_name = os.path.join(self.visualize_dir, os.path.basename(file_name[0]))
                    plt.savefig(image_name)
                    plt.close()




if __name__ == '__main__':
    # val_data_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'x')
    # val_label_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'y')
    #
    # val_dataset = ResLSTM_DataLoader(data_dir=val_data_dir, label_dir=val_label_dir)
    #
    # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    #
    # predictor = Torch_Predictor(model=ResLSTM(nclasses=3),
    #                             model_path='../../model/ResLSTM_torch/save/test1/ResLSTM_model_state_dict',
    #                             data_loader=val_loader,
    #                             save_dir='../../data/sequences/08/pred',
    #                             num_class=3, use_gpu=True)
    #
    # predictor.predict()

    data_dir = '/home/ak209/Desktop/hwei/AutoDrive/data/train_test_val/test/x'
    model_path = '/home/ak209/Desktop/hwei/PycharmProjects/AutoDrive_Project/training_save/ResLSTM_new_machine/best_model'
    save_prediction_dir = '/home/ak209/Desktop/hwei/AutoDrive/data/train_test_val/test/y'
    visualize_dir= '/home/ak209/Desktop/hwei/AutoDrive/data/train_test_val/test/visualize'


    dataset = ResLSTM_DataLoader(data_dir=data_dir, label_dir=None)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    predictor = Torch_Predictor(model=ResLSTM(nclasses=3),
                                model_path=model_path,
                                data_loader=dataloader,
                                save_dir=save_prediction_dir,
                                visualize_dir=visualize_dir,
                                num_class=3, use_gpu=True)

    predictor.test()





