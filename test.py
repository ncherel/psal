import torch

from PIL import Image
from time import time

from torchvision import transforms
from torchvision.utils import save_image

from psal import PSAttention

PSIZE = 7
HPSIZE = PSIZE//2

tf = transforms.ToTensor()

imgb = "data/img1_512.png"
imga = "data/img2_512.png"
s = 512

a = tf(Image.open(imga).resize((s,s))).to("cuda").unsqueeze(0)
b = tf(Image.open(imgb).resize((s,s))).to("cuda").unsqueeze(0)

attention = PSAttention(n_iters=10, patch_size=7, aggregation=False)

if __name__ == '__main__':
    start_time = time()
    reconstruction = attention(a, b, b)
    loss = torch.mean((reconstruction - a)**2)

    print("Success")
    print(f"Reconstruction loss: {loss.item():0.05f} in {(time() - start_time)*10:0.02f} ms")
    print("Saving reconstruction to: output.png")
    start_time = time()
    save_image(reconstruction.clone().detach(), f"output.png")
