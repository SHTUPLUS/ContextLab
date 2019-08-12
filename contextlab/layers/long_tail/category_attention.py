'''
@Description: Category Attention Network for Long Tail Computer Vision
@Author: Songyang Zhang
@Email: sy.zhangbuaa@gmail.com
@Date: 2019-08-11 13:40:08
@LastEditors: Songyang Zhang
@LastEditTime: 2019-08-12 15:11:01
'''
import torch
from torch import nn

import torch.nn.functional as F
from contextlab.utils import ConvBNReLU, GraphAdjNetwork
__all__ = ['CategoryAttentionNetwork', 'GraphConvNetwork']


class CategoryAttentionNetwork(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 graph_out_channel,
                 channel_stride,
                 norm_cfg,
                 conv_cfg
                 ):
        super(CategoryAttentionNetwork, self).__init__()

        inter_channel = in_channels // channel_stride
        self.graph_out_channel = graph_out_channel
        self.conv1 = ConvModule(
            in_channels,
            inter_channel,
            3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.linear1 = nn.Linear(inter_channel, out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.graph_conv = nn.Sequential(
            nn.Linear(out_channel, graph_out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_list, bbox_head, category_adjacency, cls_score, img_meta):
        """
        Args:
            feature_list: (list[Tensor])
            bbox_head: (callable)
            category_adjacency: (Tensor),
                shape: (num_class, num_class)
            cls_score: (Tensor)
                shape: (num_image*num_roi, num_class)
            img_meta: (list(dict))
                image information

        Return:
            enhanced_feature: (Tensor)
                Shape: (num_image*num_roi, feature_dim)
        """

        # ---------------------------------------------------
        # Step-1: Generate base feature for calculate image attention
        # ---------------------------------------------------
        conv_features = []
        if len(feature_list) > 1:
            for feature in feature_list[1:]:
                feature = F.interpolate(feature,
                                        scale_factor=(
                                            feature_list[2].size(
                                                2) / feature.size(2),
                                            feature_list[2].size(
                                                3) / feature.size(3)
                                        ))
                conv_features.append(feature)
        else:
            conv_features = feature_list
        conv_feature = torch.cat(conv_features, 1)

        # B, C_in, H, W
        conv_feature = self.conv1(conv_feature)
        # B, C_out
        conv_feature = conv_feature.mean(3).mean(2)
        se_attention_score = self.relu(self.linear1(conv_feature))

        # ---------------------------------------------------
        # Step-2: Build Global Semantic Pool
        # ---------------------------------------------------
        global_semantic_pool = torch.cat((bbox_head.fc_cls.weight,
                                          bbox_head.fc_cls.bias.unsqueeze(1)), 1).detach()

        # ---------------------------------------------------
        # Step-3: Adaptive gloabl Reasoning
        # ---------------------------------------------------
        image_wise_attention = F.softmax(
            torch.mm(
                se_attention_score,  # B, C
                # num_class, C --> C, num_class
                global_semantic_pool.transpose(0, 1)
            ), dim=1
        )  # B, num_class

        semantic_pool_updated = torch.mm(
            category_adjacency,
            global_semantic_pool
        ).unsqueeze(0)  # (num_class, num_class) X (num_class, C) --> (1, num_class, C)
        # (B, num_class, C)
        semantic_pool_updated = image_wise_attention.unsqueeze(
            -1) * semantic_pool_updated
        # B*num_class, C
        semantic_pool_updated = semantic_pool_updated.view(
            -1, global_semantic_pool.size(-1))
        semantic_pool_updated = self.graph_conv(semantic_pool_updated)

        # ---------------------------------------------------
        # Step-4: Generate Enhanced features
        # ---------------------------------------------------
        num_class = bbox_head.fc_cls.weight.size(0)
        cls_prob = F.softmax(cls_score, dim=1).view(
            len(img_meta), -1, num_class)  # B, num_region, num_class
        enhanced_feature = torch.bmm(cls_prob,
                                     semantic_pool_updated.view(len(img_meta),
                                                                num_class,
                                                                self.graph_out_channel))  # B, num_region, C_out_graph
        enhanced_feature = enhanced_feature.view(-1, self.graph_out_channel)

        return enhanced_feature

class CategorayGCN(nn.Module):
    """Build Category Graph for 
    """
        
class GraphConvNetwork(nn.Module):

    def __init__(self, 
    in_channels, 
    out_channels, 
    output_mode='BCN', 
    adj_pair_func='embedded_gaussian', 
    adj_channel_stride=1):
        super(GraphConvNetwork, self).__init__

        self.graph_conv = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            norm_layer=nn.BatchNorm2d
        )
        self.adj_generator = GraphAdjNetwork(
            pair_function=adj_pair_func,
            in_channels=in_channels,
            channel_stride=adj_channel_stride,

        )
        # self.adj_generator = builder.build_adj_generator(adj_cfg)

    def forward(self, x):
        if len(x.size()) == 3:
            B, C, N = x.size()
            x_nodes = x.permute(0, 2, 1)
        elif len(x.size()) == 4:
            B, C, H, W = x.size()
            x_nodes = x.view(B, C, W*H).permute(0, 2, 1)

        # B, N, N
        graph_adj = self.adj_generator(x_nodes)
        # (B, N, N)*(B, N, C) --> (B, N, C)
        updated_feature = torch.bmm(graph_adj, x_nodes)
        # (B,N,C) -> (B,C,N,1)->(B,C_out,N,1)
        output_feature = self.graph_conv(
            updated_feature.permute(2, 1).unsqueeze(-1))

        if self.mode == 'BCN':
            return output_feature.squeeze(-1)  # B, C, N
        elif self.mode == 'BCHW':
            return output_feature.view(B, -1, H, W)

        return output_feature
