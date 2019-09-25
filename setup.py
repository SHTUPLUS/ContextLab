'''
@Description: Setup for contextlab project
@Author: Songyang Zhang
@Email: sy.zhangbuaa@gmail.com
@Date: 2019-08-11 12:30:28
@LastEditors: Songyang Zhang
@LastEditTime: 2019-09-25 20:04:53
'''

import glob
import os 

import subprocess
import time
import platform 

import torch

from setuptools import find_packages
from setuptools import setup

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize

requirements = ['torch']

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

version_file = 'contextlab/version.py'


MAJOR = 0
MINOR = 1
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        
        # LANGUAGE is used on win 32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'
    
    return sha

def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from pluscv.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'
    
    return sha

def write_version_py():
    content = """# Generated Version File
# Time: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    
    return locals()['__version__']

if __name__ == "__main__":
    
    write_version_py()

    setup(        
        name='contextlab',
        version=get_version(),
        author="Songyang Zhang",
        url="https://github.com/SHTUPLUS/contextlab",
        long_description=readme(),
        description="Context Feature Augmentation Lab developed with PyTorch from ShanghaiTech PLUS Lab",
        packages=find_packages(exclude=("src",)),
        license='Apache License 2.0',
        install_requires=requirements,
        ext_modules=[],
        zip_safe=False
    )