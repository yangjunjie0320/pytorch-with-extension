# ./CxxExtension/setup.py
import os
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension

srcdir = os.path.dirname(os.path.abspath(__file__))
name = "mysigmoid"
cxx_extension = CppExtension(
    name, include_dirs=[os.path.join(srcdir, "include")],
    sources=[os.path.join(srcdir, "src", s) for s in os.listdir(os.path.join(srcdir, "src")) if s.endswith(".cxx")],
    extra_compile_args=["-std=c++17"],
)

setup(
    name=name, version="0.1",
    ext_modules=[cxx_extension],
    cmdclass={"build_ext": BuildExtension},
)
