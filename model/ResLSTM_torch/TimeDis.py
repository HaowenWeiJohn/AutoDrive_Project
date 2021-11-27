# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# import imp

# import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F




#
# class ResContextBlock(nn.Module):
#     def __init__(self, in_filters, out_filters):
#         super(ResContextBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=(1, 1))
#         self.act1 = nn.LeakyReLU()
#
#         self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
#         self.act2 = nn.LeakyReLU()
#         self.bn1 = nn.BatchNorm2d(out_filters)
#
#         self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=(2, 2), padding=2)
#         self.act3 = nn.LeakyReLU()
#         self.bn2 = nn.BatchNorm2d(out_filters)
#
#     def forward(self, x):
#         shortcut = self.conv1(x)
#         shortcut = self.act1(shortcut)
#
#         resA = self.conv2(shortcut)
#         resA = self.act2(resA)
#         resA1 = self.bn1(resA)
#
#         resA = self.conv3(resA1)
#         resA = self.act3(resA)
#         resA2 = self.bn2(resA)
#
#         output = shortcut + resA2
#         return output


# export
class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step"

    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*args)
        else:
            # only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in args], **kwargs)
            out_shape = out.shape
            return out.view(bs, seq_len, *out_shape[1:])

    def low_mem_forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out, dim=self.tdim)

    def __repr__(self):
        return f'TimeDistributed({self.module})'

# class TimeDisConv(nn.Module):
#
#     def __init__(self, conv_module, timedis_module):
#         super(TimeDisConv, self).__init__()
#         self.conv_module = conv_module
#         self.timedis_module = timedis_module

