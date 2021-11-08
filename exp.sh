#!/bin/zsh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generative
cd workspace/colorizationXXX
python train.py
