"""Build script for zo_rng.

Builds CUDA extension if nvcc is available, otherwise CPU-only.
Install with: pip install -e .
"""

import os
from setuptools import setup, find_packages

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
RANDOM123_INCLUDE = os.path.join(CSRC, 'random123', 'include')


def get_extensions():
    import torch
    from torch.utils.cpp_extension import (
        CppExtension, CUDAExtension, CUDA_HOME,
    )

    sources_cpu = [
        os.path.join(CSRC, 'bindings.cpp'),
        os.path.join(CSRC, 'generate_cpu.c'),
    ]
    include_dirs = [CSRC, RANDOM123_INCLUDE]

    # Check if CUDA build is possible
    has_cuda = (
        torch.cuda.is_available() or
        (CUDA_HOME is not None and os.path.isdir(CUDA_HOME))
    )

    if has_cuda:
        sources_cuda = sources_cpu + [os.path.join(CSRC, 'generate_cuda.cu')]
        ext = CUDAExtension(
            name='zo_rng._ext_impl',
            sources=sources_cuda,
            include_dirs=include_dirs,
            define_macros=[('ZO_RNG_WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp', '-DZO_RNG_WITH_CUDA'],
                'nvcc': ['-O3', '--use_fast_math=false'],
            },
            extra_link_args=['-fopenmp'],
        )
    else:
        ext = CppExtension(
            name='zo_rng._ext_impl',
            sources=sources_cpu,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp'],
            },
            extra_link_args=['-fopenmp'],
        )

    return [ext]


def build_ext_class():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension


setup(
    name='zo_rng',
    version='0.1.0',
    description='Cross-device deterministic normal RNG for zeroth-order optimization',
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={'build_ext': build_ext_class()},
    python_requires='>=3.8',
    install_requires=['torch'],
)
