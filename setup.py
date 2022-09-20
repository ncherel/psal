from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='psal',
      package_dir={"psal": "src"},
      py_modules=["psal.psal_attention"],
      ext_modules=[
          cpp_extension.CUDAExtension('psal.patchmatch', ['src/patchmatch.cu'])
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      }
)
