from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sigmoid',
    ext_modules=[
        CUDAExtension('sigmoid', [
            './src/sigmoid_kernel.cu',
            './src/sigmoid.cxx',
        ]),
    ],
    include_dirs=['./include/'],
    cmdclass={
        'build_ext': BuildExtension
    }
)
