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

    # Disable Random123 SSE to avoid conflicts with NVIDIA HPC SDK headers
    r123_no_sse = ['-DR123_USE_SSE4_2=0', '-DR123_USE_SSE4_1=0', '-DR123_USE_SSE=0']

    if has_cuda:
        # Find cudart library path (HPC SDK puts it in a non-standard location)
        import glob
        cuda_link_dirs = []
        for p in glob.glob('/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/targets/x86_64-linux/lib'):
            cuda_link_dirs.append(f'-L{p}')

        sources_cuda = sources_cpu + [os.path.join(CSRC, 'generate_cuda.cu')]
        ext = CUDAExtension(
            name='zo_rng._ext_impl',
            sources=sources_cuda,
            include_dirs=include_dirs,
            define_macros=[('ZO_RNG_WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp', '-DZO_RNG_WITH_CUDA', '-ffp-contract=off'] + r123_no_sse,
                'nvcc': [
                    '-O3', '--fmad=false',
                    '-gencode', 'arch=compute_80,code=sm_80',  # Ampere (A100)
                    '-gencode', 'arch=compute_89,code=sm_89',  # Ada Lovelace (L40S, RTX 4090)
                    '-gencode', 'arch=compute_90,code=sm_90',  # Hopper (H100)
                ] + r123_no_sse,
            },
            extra_link_args=['-fopenmp'] + cuda_link_dirs,
        )
    else:
        ext = CppExtension(
            name='zo_rng._ext_impl',
            sources=sources_cpu,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp', '-ffp-contract=off'],
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
