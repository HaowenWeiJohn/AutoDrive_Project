# import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from SalsaNext import *
from ConvLSTM_pytorch import *
from TimeDis import *

class ResLSTM(nn.Module):

    def __init__(self, nclasses, time_seq=5):
        super(ResLSTM, self).__init__()

        self.tdconv1 = TimeDistributed(ResContextBlock(time_seq,32), tdim=1)
        self.tdconv2 = TimeDistributed(ResContextBlock(32,32), tdim=1)
        self.tdconv3 = TimeDistributed(ResContextBlock(32,32), tdim=1)

        self.convlstm1 = ConvLSTM(input_dim=32, hidden_dim=32,
                            kernel_size=3, num_layers=1,
                            batch_first=True, bias=True,
                            return_all_layers=False)

        self.salsanext = SalsaNext(nclasses=nclasses)


    def forward(self, x):
        x = self.tdconv1(x)
        x = self.tdconv2(x)
        x = self.tdconv3(x)

        x = self.convlstm1(x)
        last_state_list, layer_output = self.convlstm1(x) # list of layer output

        output = self.salsanext(last_state_list[0])

        return output # logistic output