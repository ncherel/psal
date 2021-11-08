from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='reconstruction',
      ext_modules=[cpp_extension.CUDAExtension('reconstruction', ['base_patchmatch.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
