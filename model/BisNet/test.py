# import __init__ as booger
import os.path
import sys


sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.prediction_utils.utils import val

from model.CustomDataLoader.dataloader import ResLSTM_DataLoader, BiSeNet_DataLoader
from config.training_config import root_data_dir

from build_BiSeNet import BiSeNet
from model.ResLSTM_torch.Lovasz_Softmax import Lovasz_softmax

nclasses = 3
weight = [0, 9.0, 251.0]
validation_step = 3
max_miou = 0

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0')

BiSeNet_MOS = BiSeNet(3, 'resnet18')
BiSeNet_MOS.to(device)
print(torch.cuda.memory_summary())


weight=torch.tensor(weight).to(device)

WCE = nn.CrossEntropyLoss(weight=weight, ignore_index=0, reduction='none').to(device)
# NLL = nn.NLLLoss(weight=weight).to(device)
LS = Lovasz_softmax(ignore=0).to(device)

optimizer = torch.optim.AdamW(BiSeNet_MOS.parameters(), lr=0.0001,weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

scaler = torch.cuda.amp.GradScaler()


#################
# data loader
train_data_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'x')
train_label_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'y')

train_dataset = BiSeNet_DataLoader(data_dir=train_data_dir, label_dir=train_label_dir)
val_dataset = BiSeNet_DataLoader(data_dir=train_data_dir, label_dir=train_label_dir)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
#################


N, T, C = 3, 5, 9

# input_tensor = torch.randn(N, T, C, 16, 16).to(device)
# semantic_label = np.zeros((N, 1, 16, 16))
# semantic_label = torch.from_numpy(semantic_label).to(dtype=torch.long)
# semantic_label_mask = torch.ones(N, 1, 16, 16).to(dtype=torch.long)

for current_epoch in range(0, 100):

    print('Epoch: ', current_epoch)
    print(optimizer.param_groups[0]['lr'])

    for batch_index, (semantic_input, semantic_label) in enumerate(train_loader):
        semantic_input = semantic_input.to(device)
        semantic_label = semantic_label.to(device)

        output, output_sup1, output_sup2 = BiSeNet_MOS(semantic_input)

        loss1 = WCE(output, semantic_label).to(device)
        loss2 = WCE(output_sup1, semantic_label).to(device)
        loss3 = WCE(output_sup2, semantic_label).to(device)

        total_loss = loss1 + loss2 + loss3




        # semantic_output = BiSeNet_MOS(semantic_input) # (b, c, h, w)



        # pixel_losses = WCE(semantic_output, semantic_label).to(device)
        # print(pixel_losses)
        # pixel_losses = pixel_losses.to(device)
        # pixel_losses = pixel_losses.contiguous().view(-1)
        # loss_ce = pixel_losses.mean()

        # LS_loss = LS(semantic_output, semantic_label)
        # total_loss = loss_ce +  LS_loss.mean()
        # print('Loss: ', total_loss)
        total_loss = total_loss.contiguous().view(-1)
        loss_ce = total_loss.mean()

        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()
        print(loss_ce)
        # scaler.step(optimizer)
        # scaler.update()
        # break

    # if current_epoch % validation_step == 0:
    #     precision, miou = val(BiSeNet_MOS, val_loader)
    #     print('precision: ', str(precision), 'miou', miou)
    #     if miou > max_miou:
    #         max_miou = miou
    #         torch.save(BiSeNet_MOS,
    #                    'save/BiSeNet_model_state_dict')



