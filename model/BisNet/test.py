# import __init__ as booger
import os.path
import sys

from utils.auxiliary.np_ioueval import iouEval
from utils.training_logger.training_logger import Training_Logger

# sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.prediction_utils.utils import val

from model.CustomDataLoader.dataloader import ResLSTM_DataLoader, BiSeNet_DataLoader
from config.data_path_config import root_data_dir, root_save_dir

from build_BiSeNet import BiSeNet
from model.ResLSTM_torch.Lovasz_Softmax import Lovasz_softmax


# class config
# class weight

nclasses = 3
weight = [0.0, 9.0, 251.0]
validation_step = 1
max_miou = 0
max_moving_iou=0

# check if gpu is used
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0')

# construct BiSeNet
BiSeNet_MOS = BiSeNet(3, 'resnet18')
# to GPU
BiSeNet_MOS.to(device)

print(torch.cuda.memory_summary())

# weight is from the class imbalance
weight=torch.tensor(weight).to(device)

# construct loss
WCE_out = nn.CrossEntropyLoss(weight=weight, ignore_index=0, reduction='none').to(device)
WCE_1 = nn.CrossEntropyLoss(weight=weight, ignore_index=0, reduction='none').to(device)
WCE_2 = nn.CrossEntropyLoss(weight=weight, ignore_index=0, reduction='none').to(device)

# loss
LS = Lovasz_softmax(ignore=0).to(device)

# select optimizer
optimizer = torch.optim.AdamW(BiSeNet_MOS.parameters(), lr=1e-3,weight_decay=5e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

#################
# data loader

# the data path
train_data_dir = os.path.join(root_data_dir, 'train_test_val', 'train', 'x')
train_label_dir = os.path.join(root_data_dir, 'train_test_val', 'train', 'y')
val_data_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'x')
val_label_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'y')

# data loader
train_dataset = BiSeNet_DataLoader(data_dir=train_data_dir, label_dir=train_label_dir)
val_dataset = BiSeNet_DataLoader(data_dir=val_data_dir, label_dir=val_label_dir)

# this is the dataloader goes into the trainer
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
#################
# training logger
training_logger = Training_Logger(root_dir=root_save_dir, logger_dir='XiaoExample')

#################


# val_loader = tqdm(val_loader)

# N, T, C = 3, 5, 9
# input_tensor = torch.randn(N, T, C, 16, 16).to(device)
# semantic_label = np.zeros((N, 1, 16, 16))
# semantic_label = torch.from_numpy(semantic_label).to(dtype=torch.long)
# semantic_label_mask = torch.ones(N, 1, 16, 16).to(dtype=torch.long)


# train_iou_eval = iouEval(n_classes=3, ignore=0)
val_iou_eval = iouEval(n_classes=3, ignore=0)


for current_epoch in range(0, 200):

    print('Epoch: ', current_epoch)
    print('learning rate: ', optimizer.param_groups[0]['lr'])

    # turn to training mode
    BiSeNet_MOS.train()
    # flash the buffer
    looper = tqdm(train_loader)

    epoch_loss = []

    for (semantic_input, semantic_label) in looper:

        # send tensor to device
        semantic_input = semantic_input.to(device)
        semantic_label = semantic_label.to(device)

        # forward
        output, output_sup1, output_sup2 = BiSeNet_MOS(semantic_input)
        # a = output.detach().numpy()

        # compute loss
        pixel_loss1 = WCE_out(output, semantic_label).to(device)
        pixel_loss2 = WCE_1(output_sup1, semantic_label).to(device)
        pixel_loss3 = WCE_2(output_sup2, semantic_label).to(device)
        ls_loss = LS(output, semantic_label).to(device)
        total_pixel_loss = pixel_loss1+pixel_loss2+pixel_loss3+ls_loss
        total_pixel_loss = total_pixel_loss.contiguous().view(-1)
        ce_loss = total_pixel_loss.mean()

        # ls_loss = LS(F.softmax(output, dim=1), semantic_label)
        total_loss = ce_loss
        # remove the previous gradient
        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm(BiSeNet_MOS.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss.append(total_loss.item())

        # set the process bar information
        looper.set_description("Total_loss = %s" % str(total_loss.item()))
        looper.refresh()
        # break


    scheduler.step()
    # calculate epoch loss
    train_ave_epoch_loss = np.average(epoch_loss)
    print('train_ave_epoch_loss: ', str(train_ave_epoch_loss))

    history_content = [current_epoch, train_ave_epoch_loss, None, None, None]
                      # current epoch, train loss, miou, iou, accuracy



    if current_epoch % validation_step == 0:  # make one val after the first epoch
        iou_mean, iou, acc = val(model=BiSeNet_MOS, dataloader=val_loader) # the evaluation function
        history_content[2] = iou_mean
        history_content[3] = iou
        history_content[4] = acc
        moving_iou = iou[-1]

        print('validation iou mean: ', str(iou_mean), 'validation iou: ', iou,'validation acc: ', str(acc))   #
        # save the best model
        if moving_iou > max_moving_iou:
            max_moving_iou = moving_iou
            training_logger.save_model(model=BiSeNet_MOS)

    training_logger.log_hist(epoch=history_content[0],
                             train_loss=history_content[1],
                             val_miou=history_content[2],
                             val_iou = history_content[3],
                             val_acc=history_content[4])


