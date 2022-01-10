# # import __init__ as booger
# import os.path
# import sys
# import time
#
# sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# from model.CustomDataLoader.dataloader import ResLSTM_DataLoader
# from config.data_path_config import root_data_dir
#
#
# from model.ResLSTM_torch.Lovasz_Softmax import Lovasz_softmax
# from model.ResLSTM_torch.ResLSTM_torch import ResLSTM
#
# nclasses = 3
#
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# device = torch.device('cuda:0')
#
#
# ResLSTM_model = ResLSTM(nclasses).to(device)
# pytorch_total_params = sum(p.numel() for p in ResLSTM_model.parameters())
#
# print(torch.cuda.memory_summary())
# weight = [0, 9.0, 251.0]
# weight=torch.tensor(weight).to(device)
#
# WCE = nn.CrossEntropyLoss(weight=weight, ignore_index=0, reduction='none').to(device)
# # NLL = nn.NLLLoss(weight=weight).to(device)
# LS = Lovasz_softmax(ignore=0).to(device)
#
# optimizer = torch.optim.AdamW(ResLSTM_model.parameters(), lr=0.0001,weight_decay=0.00001)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#
# scaler = torch.cuda.amp.GradScaler()
#
#
# #################
# # data loader
# train_data_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'x')
# train_label_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'y')
#
# train_dataset = ResLSTM_DataLoader(data_dir=train_data_dir, label_dir=train_label_dir)
# val_dataset = ResLSTM_DataLoader(data_dir=train_data_dir, label_dir=train_label_dir)
#
# train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=True)
# #################
#
#
# N, T, C = 3, 5, 9
#
# # input_tensor = torch.randn(N, T, C, 16, 16).to(device)
# # semantic_label = np.zeros((N, 1, 16, 16))
# # semantic_label = torch.from_numpy(semantic_label).to(dtype=torch.long)
# # semantic_label_mask = torch.ones(N, 1, 16, 16).to(dtype=torch.long)
# # batch_iteration = tqdm(train_loader)
#
# epoch_loss = []
# for current_epoch in range(0, 100):
#
#     print('Epoch: ', current_epoch)
#     print(optimizer.param_groups[0]['lr'])
#     batch_loss = []
#     for batch_index, (semantic_input, semantic_label) in enumerate(train_loader):
#         semantic_input = semantic_input.to(device)
#         semantic_label = semantic_label.to(device)
#         # print(optimizer.param_groups[0]['lr'])
#         # print('batch_index:', batch_index)
#         # semantic_label = torch.squeeze(semantic_label, dim=1).to(device)
#         # semantic_label_mask = torch.squeeze(semantic_label_mask, dim=1)
#         with torch.cuda.amp.autocast(enabled=True):
#             semantic_output = ResLSTM_model(semantic_input) # (b, c, h, w)
#             pixel_losses = WCE(semantic_output, semantic_label).to(device)
#             # print(pixel_losses)
#             # pixel_losses = pixel_losses.to(device)
#             pixel_losses = pixel_losses.contiguous().view(-1)
#             loss_ce = pixel_losses.mean()
#
#         LS_loss = LS(semantic_output, semantic_label)
#         total_loss = loss_ce +  LS_loss.mean()
#         # print('Loss: ', total_loss)
#         optimizer.zero_grad()
#         scaler.scale(total_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         batch_loss.append(total_loss.item())
#         # batch_iteration.set_description('loss: '+ str(total_loss))
#         # batch_loss.append(total_loss.item())
#         print('Epoch: %s,  Batch Index: %s, Total Loss: %s' % (str(current_epoch), str(batch_index), str(total_loss) ))
#     torch.save(ResLSTM_model.state_dict(), 'save/test1/ResLSTM_model_state_dict')
#     epoch_loss.append(np.average(batch_loss))
#     np.save('loss_his', np.array(epoch_loss))
#     print(epoch_loss)
#     # print('Epoch: %s,  Batch Index: %s, Total Loss: %s' % (str(current_epoch), str(batch_index), str(total_loss) ))
#
#
#
#
# import __init__ as booger
import os.path
import sys

from model.ResLSTM_torch.ResLSTM_torch import ResLSTM
from model.SalsaNext.SalsaNext import SalsaNext
from utils.auxiliary.np_ioueval import iouEval
from utils.training_logger.training_logger import Training_Logger

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.prediction_utils.utils import val
from model.CustomDataLoader.dataloader import ResLSTM_DataLoader, BiSeNet_DataLoader
from config.data_path_config import root_data_dir, root_save_dir
from model.ResLSTM_torch.Lovasz_Softmax import Lovasz_softmax

nclasses = 3
weight = [0.0, 9.0, 251.0]
validation_step = 1
max_miou = 0

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0')

Model = SalsaNext(nclasses, input_channel_num=13).to(device)
# BiSeNet_MOS.to(device)

print(torch.cuda.memory_summary())


weight=torch.tensor(weight).to(device)

WCE = nn.CrossEntropyLoss(weight=weight, ignore_index=0, reduction='none').to(device)

LS = Lovasz_softmax(ignore=0).to(device)

optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-5,weight_decay=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

#################
# data loader
train_data_dir = os.path.join(root_data_dir, 'train_test_val', 'train', 'x')
train_label_dir = os.path.join(root_data_dir, 'train_test_val', 'train', 'y')
val_data_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'x')
val_label_dir = os.path.join(root_data_dir, 'train_test_val', 'val', 'y')

train_dataset = BiSeNet_DataLoader(data_dir=train_data_dir, label_dir=train_label_dir)
val_dataset = BiSeNet_DataLoader(data_dir=val_data_dir, label_dir=val_label_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
#################
# training logger
training_logger = Training_Logger(root_dir=root_save_dir, logger_dir='SalsaNext_MOS')

#################


# val_loader = tqdm(val_loader)

# N, T, C = 3, 5, 9
# input_tensor = torch.randn(N, T, C, 16, 16).to(device)
# semantic_label = np.zeros((N, 1, 16, 16))
# semantic_label = torch.from_numpy(semantic_label).to(dtype=torch.long)
# semantic_label_mask = torch.ones(N, 1, 16, 16).to(dtype=torch.long)


# train_iou_eval = iouEval(n_classes=3, ignore=0)
val_iou_eval = iouEval(n_classes=3, ignore=0)


for current_epoch in range(0, 60):

    print('Epoch: ', current_epoch)
    print('learning rate: ', optimizer.param_groups[0]['lr'])

    Model.train()
    looper = tqdm(train_loader)

    epoch_loss = []

    for (semantic_input, semantic_label) in looper:
        semantic_input = semantic_input.to(device)
        semantic_label = semantic_label.to(device)

        # semantic_input = semantic_input.to(device)

        #         semantic_label = semantic_label.to(device)
        #         # print(optimizer.param_groups[0]['lr'])
        #         # print('batch_index:', batch_index)
        #         # semantic_label = torch.squeeze(semantic_label, dim=1).to(device)
        #         # semantic_label_mask = torch.squeeze(semantic_label_mask, dim=1)
        #         with torch.cuda.amp.autocast(enabled=True):
        #             semantic_output = ResLSTM_model(semantic_input) # (b, c, h, w)
        #             pixel_losses = WCE(semantic_output, semantic_label).to(device)
        #             # print(pixel_losses)
        #             # pixel_losses = pixel_losses.to(device)
        #             pixel_losses = pixel_losses.contiguous().view(-1)
        #             loss_ce = pixel_losses.mean()

        output = Model(semantic_input)
        # a = output.detach().numpy()

        pixel_loss = WCE(output, semantic_label).to(device)
        # pixel_loss2 = WCE(output_sup1, semantic_label).to(device)
        # pixel_loss3 = WCE(output_sup2, semantic_label).to(device)
        #
        # total_pixel_loss = pixel_loss1+pixel_loss2+pixel_loss3

        total_pixel_loss = pixel_loss.contiguous().view(-1)
        ce_loss = total_pixel_loss.mean()

        ls_loss = LS(F.softmax(output, dim=1), semantic_label)

        total_loss = ce_loss+ls_loss.mean()
        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm(BiSeNet_MOS.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss.append(total_loss.item())

        looper.set_description("Total_loss = %s" % str(total_loss.item()))
        looper.refresh()
        # break


    scheduler.step()
    train_ave_epoch_loss = np.average(epoch_loss)
    print('train_ave_epoch_loss: ', str(train_ave_epoch_loss))

    history_content = [current_epoch, train_ave_epoch_loss, None, None]



    if current_epoch % validation_step == 0:  # make one val after the first epoch
        iou_mean, iou, acc = val(model=Model, dataloader=val_loader)
        history_content[2] = iou_mean
        history_content[3] = acc

        print('validation iou mean: ', str(iou_mean), 'validation acc: ', str(acc))
        if iou_mean > max_miou:
            max_miou = iou_mean
            training_logger.save_model(model=Model)

    training_logger.log_hist(epoch=history_content[0],
                             train_loss=history_content[1],
                             val_miou=history_content[2],
                             val_acc=history_content[3])


