'''
@Description: Dual Attention Network
@Author: Songyang Zhang
@Email: sy.zhangbuaa@gmail.com
@Date: 2019-07-13 17:10:33
@LastEditTime: 2019-08-15 10:22:03
@LastEditors: Songyang Zhang
'''

import numpy as np

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-attention Network/Non-local Network

    Args:
        
    Return:

    """
    def __init__(self, inplane, outplane, channel_stride=8):
        super(SelfAttention, self).__init__()
        
        self.inplane = inplane
        self.outplane = outplane
        
        self.inter_channel = inplane // channel_stride
        
        self.query_conv = nn.Conv2d(in_channels=inplane, out_channels=self.inter_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=inplane, out_channels=self.inter_channel, kernel_size=1)
        
        self.value_conv = nn.Conv2d(in_channels=inplane, out_channels=outplane, kernel_size=1)
        if outplane != inplane:
            self.input_conv = nn.Conv2d(in_channels=inplane, out_channels=outplane, kernel_size=1)
            
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """
        Args:
            inputs: (B, C, H, W)
        
        Return:
            augmented_feature: (B, C, H, W)
        """
        
        B, C, H, W = inputs.size()
        query = self.query_conv(inputs).view(B, -1, H*W).permute(0, 2, 1) # B,N,C
        key = self.key_conv(inputs).view(B, -1, H*W) # B,C,N

        affinity_matrix = torch.bmm(query, key)
        affinity_matrix = self.softmax(affinity_matrix) # B, N, N

        value = self.value_conv(inputs).view(B, -1, H*W)

        out = torch.bmm(value, affinity_matrix) # B,C',N * B,N,N = B,C',N
        if self.inplane != self.outplane:
            inputs = self.input_conv(inputs)
        augmented_feature = self.gamma * out.view(B,-1, H, W) + inputs

        return augmented_feature

# if __name__ == "__main__":
#     inputs = torch.randn(1,1024, 20,20)
#     model = SelfAttention(1024, 512,channel_stride=8)

#     out = model(inputs)
#     import pdb; pdb.set_trace()

        
