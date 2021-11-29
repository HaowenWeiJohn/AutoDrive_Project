# import __init__ as booger
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')

from model.ResLSTM_torch.Lovasz_Softmax import Lovasz_softmax
from model.ResLSTM_torch.ResLSTM_torch import ResLSTM

nclasses = 3

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda:0')


ResLSTM_model = ResLSTM(nclasses).to(device)
print(torch.cuda.memory_summary())
weight = [0,9,251]
weight=torch.tensor(weight).to(device)

WCE = nn.CrossEntropyLoss(weight=weight, ignore_index=0,reduction='none').to(device)
# NLL = nn.NLLLoss(weight=weight).to(device)
LS = Lovasz_softmax(ignore=0).to(device)

optimizer = torch.optim.AdamW(ResLSTM_model.parameters(), lr=0.0001,weight_decay=0.005)

scaler = torch.cuda.amp.GradScaler()

N, T, C = 3, 5, 9

for current_epoch in range(0, 100):

    print('Epoch: ', current_epoch)

    for batch_index in range(0,100):

        input_tensor = torch.randn(N, T, C, 64, 2048, device=device)
        semantic_label = torch.empty(N, 1, 64, 2048).to(dtype=torch.long)
        semantic_label_mask = torch.empty(N, 1, 64, 2048).to(dtype=torch.long)

        semantic_label = torch.squeeze(semantic_label, dim=1).to(device)
        semantic_label_mask = torch.squeeze(semantic_label_mask, dim=1)
        with torch.cuda.amp.autocast(enabled=True):
            print('John1')
            semantic_output = ResLSTM_model(input_tensor) # (b, c, h, w)
            pixel_losses = WCE(semantic_output, semantic_label)
            print('John2')
            pixel_losses = pixel_losses* semantic_label_mask.to(device)
            pixel_losses = pixel_losses.contiguous().view(-1)
            loss_ce = pixel_losses.mean()

        LS_loss = LS( semantic_output, semantic_label)
        total_loss = loss_ce +  LS_loss.mean()

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)




