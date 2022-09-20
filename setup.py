from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='psal',
      packages=find_packages(
          where="src"
      ),
      ext_modules=[
          cpp_extension.CUDAExtension('psal.patchmatch', ['src/patchmatch.cu'])
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      }
)
