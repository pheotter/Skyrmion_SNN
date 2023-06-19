from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

# directory
include_dirs = os.path.dirname(os.path.abspath(__file__))

# directory of .cpp file
source_file = glob.glob(os.path.join(include_dirs, 'src', '*.cpp'))

setup(
    name='leaky_cpp',  # module name
    ext_modules=[CppExtension('leaky_cpp', sources=source_file, include_dirs=[include_dirs])],
    cmdclass={
        'build_ext': BuildExtension
    }
)
