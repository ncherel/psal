# Patch-Based Stochastic Attention

This repository contains the code associated to the paper Patch-Based Stochastic Attention for Image Editing (https://arxiv.org/abs/2202.03163)
PSAL is an efficient attention layer for images and features-maps. It is based on an efficient nearest-neighbor search approach, fueled by PatchMatch.

## Installation

PSAL contains CUDA code and can only be run on GPU. PSAL is available as a PyTorch extension. You'll need CUDA with NVCC support and PyTorch.

Important packages are:
- PyTorch (with GPU support)
- Cudatoolkit-dev (for NVCC)
- C/C++ compiler compatible with NVCC

Suggested versions:
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install cudatoolkit-dev=10.1.243 -c conda-forge
conda install gcc_linux-64=7.3 gxx_linux-64=7.3
```

Once all requirements are met, you can install and test the module:
```
python setup.py install
python test.py
```


## Use

Cross-Attention with L2 distance is implemented in the following manner:

```python
import torch
from psal import PSAttention

q = torch.rand(1, 3, 128, 128)
k = torch.rand(1, 3, 64, 64)
v = torch.rand(1, 3, 64, 64)

attention = PSAttention()

output = attention(q, k, v)

```


## Citing

```
@misc{https://doi.org/10.48550/arxiv.2202.03163,
  doi = {10.48550/ARXIV.2202.03163},  
  url = {https://arxiv.org/abs/2202.03163},
  title = {Patch-Based Stochastic Attention for Image Editing},
  author = {Cherel, Nicolas and Almansa, Andrés and Gousseau, Yann and Newson, Alasdair},
  publisher = {arXiv},
  year = {2022}
}
```

