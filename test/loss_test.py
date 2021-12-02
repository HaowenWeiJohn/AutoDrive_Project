import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/MOS_Project/AutoDrive_Project')

device = torch.device('cuda:0')
N, T, C = 3, 5, 9

input_tensor = torch.randn(N, 3, 16, 16, requires_grad=True)
semantic_label = np.zeros((N, 16, 16))
semantic_label = torch.from_numpy(semantic_label).to(dtype=torch.long)
semantic_label_mask = torch.ones(N, 16, 16).to(dtype=torch.long)

criterion = nn.CrossEntropyLoss()

loss = 0 + criterion(input_tensor, semantic_label)
print(loss)
loss.backward()