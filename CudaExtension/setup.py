# ./CxxExtension/setup.py
import os
from os import listdir
from os.path import join, dirname, abspath

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

name = "sigmoid_with_cuda_extension"
path = dirname(abspath(__file__))

is_source_file = lambda s: s.endswith(".cxx") or s.endswith(".cu")
sources = (lambda d: [join(d, s) for s in listdir(d) if is_source_file(s)])(
    join(path, "src")
)

extension = CUDAExtension(
    name, sources=sources,
    include_dirs=[join(path, "include")],
)

cmdclass = {"build_ext": BuildExtension}

setup(
    name=name, version="0.1",
    ext_modules=[extension],
    cmdclass=cmdclass,
)
