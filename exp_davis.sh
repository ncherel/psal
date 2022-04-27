#!/bin/zsh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate generative
cd workspace/reconstruction-psal-3
echo "PSAL - 3"
mkdir -p output
python setup.py build_ext --inplace --force
python train.py
