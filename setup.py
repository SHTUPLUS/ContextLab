'''
@Description: Setup for contextlab project
@Author: Songyang Zhang
@Email: sy.zhangbuaa@gmail.com
@Date: 2019-08-11 12:30:28
@LastEditors: Songyang Zhang
@LastEditTime: 2019-08-11 12:35:24
'''

import glob
import os 

import torch
from setuptools import find_packages
from setuptools import setup

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ['torch']

setup(
    name='contextlab',
    version="0.1",
    author="Songyang Zhang",
    url="https://github.com/SHTUPLUS/contextlab",
    description="Context Feature Augmentation Lab developed with PyTorch from ShanghaiTech PLUS Lab",
    packages=find_packages(exclude=("src",)),
)