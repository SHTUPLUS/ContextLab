# from contextlab.layers.long_tail import CategoryAttentionNetwork
from contextlab.utils import GraphAdjNetwork

import torch

if __name__ == "__main__":
    inputs = torch.randn(2, 1024, 8).permute(0, 2, 1)
    network = GraphAdjNetwork(
        pair_function='embedded_gaussian',
        in_channels=1024,
        channel_stride=8,
    )
    output = network(inputs)
    print('Embedding Gaussian run pass')

    network = GraphAdjNetwork(
        pair_function='gaussian',
        in_channels=1024,
        channel_stride=8,
    )
    output = network(inputs)
    
    print('Gaussian run pass')

    network = GraphAdjNetwork(
        pair_function='diff_learnable',
        in_channels=1024,
        channel_stride=8,
    )
    output = network(inputs)
    
    print('Diff learnable run pass')

    network = GraphAdjNetwork(
        pair_function='sum_learnable',
        in_channels=1024,
        channel_stride=8,
    )
    output = network(inputs)
    print('Sum learnable run pass')

    network = GraphAdjNetwork(
        pair_function='cat_learnable',
        in_channels=1024,
        channel_stride=8,
    )
    output = network(inputs)
    print('Cat learnable run pass')
