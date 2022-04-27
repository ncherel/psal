import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from time import time

import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Function
from torch.nn.functional import pad, unfold, conv2d

from torchvision import transforms
from torchvision.utils import save_image

from reconstruction import backward, patchmatch

from train import MyModel, PSIZE
from glob import glob
from os.path import join
from torchvision.transforms import ToTensor

tf = ToTensor()
model = torch.load("last_model.pth")

path = "../data/DAVIS-testchallenge/DAVIS/JPEGImages/480p/"
folders = {}
for f in glob(join(path, "*/")):
    images = sorted(glob(join(f, "*.jpg")))
    if len(images) > 0:
        folders[f] = images

def preprocessing(imga, size=256):
    return tf(Image.open(imga).resize((size, size))).to("cuda").unsqueeze(0)

# Go through all folders
for f in folders:
    # Predict the middle frame of the sequence
    sequence = folders[f]
    sequence_name = f.split("/")[-2]

    middle_idx = len(sequence) // 2
    main_frame = preprocessing(sequence[middle_idx])
    main_color = main_frame.clone()
    main_frame[:] = torch.mean(main_color, dim=1, keepdim=True)

    for d in [1,3,5,10]:
        for sign in [1, -1]:

            with torch.no_grad():
                # Pick the reference frame (clipping)
                ref_idx = middle_idx + sign * d

                if not 0 <= ref_idx < len(sequence):
                    print(f"No reference image for idx {ref_idx} ({middle_idx} {sign * d})")

                ref = preprocessing(sequence[ref_idx])

                reconstruction = model(main_frame, ref)
                diff = (reconstruction - main_color)**2

                # Compute loss (ignore boundaries)
                loss = torch.mean(diff[:,:,PSIZE:-PSIZE, PSIZE:-PSIZE])
                print(f"{sequence_name},{middle_idx},{sign*d},{loss.item()}")
                save_image(reconstruction.clone().detach(), f"results/{sequence_name}_{sign*d}.png")
