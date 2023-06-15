from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp

ROOT = osp.dirname(osp.abspath(__file__))

setup(
  name='depth_cov', 
  version='0.1.0', 
  author='edexheim',
  packages=['depth_cov'],

  ext_modules=[
    CUDAExtension('depth_cov_backends', 
      include_dirs=[
        osp.join(ROOT, 'depth_cov/backend/include'), 
      ],
      sources=[
        'depth_cov/backend/src/cov.cpp', 
        'depth_cov/backend/src/cov_gpu.cu',
        'depth_cov/backend/src/cov_cpu.cpp',
        'depth_cov/backend/src/sampler.cpp',
        'depth_cov/backend/src/depth_cov_backends.cpp'
      ],
      extra_compile_args={
        'cores': ['j8'],
        'cxx': ['-O2'], 
        'nvcc': ['-O2',
            '-gencode=arch=compute_86,code=sm_86',
        ]
      }
    )
  ],
  cmdclass={ 'build_ext': BuildExtension }
)