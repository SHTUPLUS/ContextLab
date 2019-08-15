import torch
from torch import nn
import torch.nn.functional as F
__all__ = ['ConvBNReLU']


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 norm_layer=nn.BatchNorm2d,
                 with_relu=True,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
        self.bn = norm_layer(out_channels)
        self.with_relu = with_relu
        if with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class GraphAdjNetwork(nn.Module):
    def __init__(self,
                 pair_function,
                 in_channels,
                 channel_stride):
        super(GraphAdjNetwork, self).__init__()
        self.pair_function = pair_function

        if pair_function == 'embedded_gaussian':
            inter_channel = in_channels // channel_stride
            self.phi = ConvBNReLU(
                in_channels=in_channels,
                out_channels=inter_channel,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d
                )
            self.theta = ConvBNReLU(
                in_channels=in_channels,
                out_channels=inter_channel,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d
                )
        elif pair_function == 'gaussian':
            pass
        elif pair_function == 'diff_learnable':
            self.learnable_adj_conv = ConvBNReLU(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d
                )
        elif pair_function == 'sum_learnable':
            self.learnable_adj_conv = ConvBNReLU(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d
                )
        elif pair_function == 'cat_learnable':
            self.learnable_adj_conv = ConvBNReLU(
                in_channels=in_channels*2,
                out_channels=1,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d
                )
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x (Tensor):
                (B, N, C)
        """
        if self.pair_function == 'gaussian':
            adj = self.gaussian(x, x.permute(0, 2, 1))
        elif self.pair_function == 'embedded_gaussian':
            x = x.permute(0, 2, 1).unsqueeze(-1)
            x_1 = self.phi(x)  # B, C, N, 1
            x_2 = self.theta(x)  # B, C, N, 1
            adj = self.gaussian(
                x_1.squeeze(-1).permute(0, 2, 1), x_2.squeeze(-1))
        elif self.pair_function == 'diff_learnable':
            adj = self.diff_learnable_adj(x.unsqueeze(2), x.unsqueeze(1))
        elif self.pair_function == 'sum_learnable':
            adj = self.sum_learnable_adj(x.unsqueeze(2), x.unsqueeze(1))
        elif self.pair_function == 'cat_learnable':
            adj = self.cat_learnable_adj(x.unsqueeze(2), x.unsqueeze(1))
        else:
            raise NotImplementedError(self.pair_function)

        return adj

    def gaussian(self, x_1, x_2):
        """
        Args:
            x_1:
            x_2:
        Return:
            adj: normalized in the last dimenstion
        """
        # (B, N, C) X (B, C, N) --> (B, N, N)
        adj = torch.bmm(x_1, x_2)  # B, N, N
        adj = F.softmax(adj, dim=-1)  # B, N, N
        return adj

    def diff_learnable_adj(self, x_1, x_2):
        """
        Learnable attention from the difference of the feature 

        Return:
            adj: normalzied at the last dimension
        """
        # x1:(B,N,1,C)
        # x2:(B,1,N,C)
        feature_diff = x_1 - x_2  # (B, N, N, C)
        feature_diff = feature_diff.permute(0, 3, 1, 2)  # (B, C, N, N)
        adj = self.learnable_adj_conv(feature_diff)  # (B, 1, N, N)
        adj = adj.squeeze(1)  # (B, N, N)
        # Use the number of nodes as the normalization factor
        adj = adj / adj.size(-1)  # (B, N, N)

        return adj

    def sum_learnable_adj(self, x_1, x_2):
        """
        Learnable attention from the difference of the feature 

        Return:
            adj: normalzied at the last dimension
        """
        # x1:(B,N,1,C)
        # x2:(B,1,N,C)
        feature_diff = x_1 + x_2  # (B, N, N, C)
        feature_diff = feature_diff.permute(0, 3, 1, 2)  # (B, C, N, N)
        adj = self.learnable_adj_conv(feature_diff)  # (B, 1, N, N)
        adj = adj.squeeze(1)  # (B, N, N)
        # Use the number of nodes as the normalization factor
        adj = adj / adj.size(-1)  # (B, N, N)

        return adj

    def cat_learnable_adj(self, x_1, x_2):
        """
        Learable attention from the concatnation of the features
        """
        x_1 = x_1.repeat(1, 1, x_1.size(1), 1)  # B, N, N, C
        x_2 = x_2.repeat(1, x_2.size(2), 1, 1)  # B, N, N, C
        feature_cat = torch.cat([x_1, x_2], dim=-1)  # B, N, N, 2C
        # import pdb; pdb.set_trace()
        feature_cat = feature_cat.permute(0, 3, 1, 2)  # B, 2C, N, N
        adj = self.learnable_adj_conv(feature_cat)  # B, 1, N, N
        adj = adj.squeeze(1)  # (B, N, N)
        # Use the number of nodes as the normalization factor
        adj = adj / adj.size(-1)  # (B, N, N)

        return adj
