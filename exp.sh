#!/bin/zsh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generative
cd workspace/colorizationXXX
mkdir -p output
python setup.py build_ext --inplace
python train.py
