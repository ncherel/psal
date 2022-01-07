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


PSIZE = 7
HPSIZE = PSIZE//2

tf = transforms.ToTensor()

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_channels = 16
        self.convq = nn.Sequential(nn.Conv2d(3, n_channels, 3, padding=1), nn.ReLU())
        self.convk = nn.Sequential(nn.Conv2d(3, n_channels, 3, padding=1), nn.ReLU())
        self.convv = nn.Sequential(nn.Conv2d(3, n_channels, 3, padding=1), nn.ReLU())
        self.final = nn.Sequential(nn.Conv2d(3 + n_channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, a, b):
        # First conv layer for Q, K, V
        q = self.convq(a)
        k = self.convk(b)
        v = self.convv(b)

        att, _, _ = attention_layer(q,k,v)

        return self.final(torch.cat((a, att), dim=1))


class PatchMatch(Function):
    @staticmethod
    def forward(ctx, a, b, n_iters=10):
        shift_map, cost_map = patchmatch(a, b, n_iters=n_iters)
        torch.cuda.synchronize()
        ctx.save_for_backward(a, b, shift_map)
        shift_map = shift_map.type(torch.int64)
        return shift_map, cost_map

    @staticmethod
    def backward(ctx, shift_map_grad, cost_map_grad):
        a, b, shift_map = ctx.saved_tensors
        grad_a, grad_b = backward(a, b, shift_map, cost_map_grad)
        torch.cuda.synchronize()
        return grad_a, grad_b


def full_attention_layer(q, k, v, T=1):
    def extract_patches(img, stride=3):
        patches = unfold(img, PSIZE, stride=stride)
        patches = patches.view(img.shape[1], PSIZE, PSIZE, -1)
        patches = patches.permute(3, 0, 1, 2)
        return patches

    # Filters from image b
    patches = extract_patches(k)
    v_patches = extract_patches(v)
    b_norm = torch.sum(patches ** 2, dim=(1, 2, 3))

    # True distance
    dot_product = conv2d(q, patches, stride=1, padding=HPSIZE)
    sum_kernel = torch.ones((1, q.shape[1], PSIZE, PSIZE), device="cuda")
    a_norm = conv2d(q**2, sum_kernel, stride=1, padding=HPSIZE)
    distances = b_norm[None,:,None,None] + a_norm - 2*dot_product
    distances = torch.softmax(-T*distances, dim=1)

    # Reconstruction using central pixel
    reconstruction = torch.einsum('dc,dij->cij', v_patches[:, :, HPSIZE, HPSIZE], distances[0]).unsqueeze(0)

    return reconstruction


def attention_layer(q, k, v):
    """Can only handle batch of size 1"""
    # PatchMatch layer takes C H W tensor
    q = q[0]
    k = k[0]

    shift_map, cost_map = PatchMatch.apply(q, k)

    # Simple reconstruction using the central pixel and no weighting scheme
    cost_map = torch.softmax(-cost_map, dim=0)
    reconstruction = torch.sum(cost_map[None, None, :, :, :] * v[:,:,shift_map[0], shift_map[1]], dim=2)

    return reconstruction, shift_map, cost_map



model = MyModel().cuda()
optimizer = Adam(model.parameters(), lr=0.001)

from torch.utils.data import Dataset, DataLoader
from glob import glob
from os.path import join
class MyDataset(Dataset):
    def __init__(self, path):
        self.folders = {}
        for f in glob(join(path, "*/")):
            images = glob(join(f, "*.jpg"))
            if len(images) > 0:
                self.folders[f] = images

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        folder = np.random.choice(list(self.folders.keys()))
        imga, imgb = np.random.choice(self.folders[folder], 2, replace=False)
        s = 256
        a = tf(Image.open(imga).resize((s,s))).to("cuda")
        b = tf(Image.open(imgb).resize((s,s))).to("cuda")

        return a, b


dataset = MyDataset("../data/DAVIS/JPEGImages/480p")

for j in range(20):
    dataloader = DataLoader(dataset, batch_size=1)

    start_time = time()
    for i, batch in enumerate(dataloader):
        a, b = batch

        a_color = a.clone()
        a[:] = torch.mean(a_color, dim=1, keepdim=True)

        optimizer.zero_grad()
        reconstruction = model(a, b)
        diff = (reconstruction - a_color)**2
        loss = torch.mean(diff[:,:,PSIZE:-PSIZE, PSIZE:-PSIZE])
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"{j:02d},{i:05d},{loss.item():0.05f},{(time() - start_time)*10:0.02f}ms")
            start_time = time()
            save_image(reconstruction.clone().detach(), f"output/{i:05d}.png")

torch.save(model, "last_model.pth")
torch.save(optimizer, "last_optimizer.pth")
