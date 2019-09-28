'''
@Description: Setup for contextlab project
@Author: Songyang Zhang
@Email: sy.zhangbuaa@gmail.com
@Date: 2019-08-11 12:30:28
@LastEditors: Songyang Zhang
@LastEditTime: 2019-09-27 19:58:41
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

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

requirements = ['torch']

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

version_file = 'contextlab/version.py'

if torch.cuda.is_available():
    if 'LD_LIBRARY_PATH' not in os.environ:
            raise Exception('LD_LIBRARY_PATH is not set.')
    cuda_lib_path = os.environ['LD_LIBRARY_PATH'].split(':')
else:
    raise Exception('This implementation is only avaliable for CUDA devices.')


MAJOR = 0
MINOR = 2
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


def make_cuda_ext(name, module, sources, include_dirs=[]):

  
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=include_dirs,
        library_dirs=cuda_lib_path,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                # '-D__CUDA_NO_HALF_OPERATORS__',
                # '-D__CUDA_NO_HALF_CONVERSIONS__',
                # '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def tree_filter_files():

    extensions_dir = 'contextlab/layers/tree_filter/src'
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "*", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*", "*.cu"))
    
    sources = source_cpu + source_cuda + main_file
    
    return extensions_dir, sources

if __name__ == "__main__":
    
    write_version_py()

    tree_extensions_dir, tree_sources = tree_filter_files()
    
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
        ext_modules=[
            # make_cuda_ext(
            #     name='tree_filter_cuda',
            #     module='contextlab.layers.tree_filter',
            #     include_dirs=[tree_extensions_dir],
            #     sources=tree_sources),
            CUDAExtension(
                name='contextlab.layers.tree_filter.functions.tree_filter_cuda',
                # module='contextlab.layers.tree_filter',
                include_dirs=[tree_extensions_dir],
                sources=tree_sources,
                library_dirs=cuda_lib_path,
                extra_compile_args={'cxx':['-O3'],
                                    'nvcc':['-O3']}),
            CUDAExtension(
                name='contextlab.layers.cc_attention.rcca',
                sources=['contextlab/layers/cc_attention/src/lib_cffi.cpp',
                         'contextlab/layers/cc_attention/src/ca.cu'],
                extra_compile_args= ['-std=c++11'],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["--expt-extended-lambda"],
            )
        ],
        cmdclass={
        'build_ext': BuildExtension
        },
        zip_safe=False
    )