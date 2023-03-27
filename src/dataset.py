import h5py
from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

@dataclass
class data_config:
    threshold = 0.2
    upper = 1 
    lower = 0
    x_shape = 16
    y_shape = 16
    z_shape = 16

def random_color(V):

    Hues_arr = np.array([60, 120, 240, 360])

    hues = np.random.choice(Hues_arr, 1)
    colored_image = np.zeros((3, V.shape[0], V.shape[1], V.shape[2]))
    H_i = round(int(hues)/60) % 6
    colored_image[0] = V
    colored_image[1] = V
    colored_image[2] = V
    for x in range(data_config.x_shape):
        for y in range(data_config.y_shape):
            for z in range(data_config.z_shape):
                if H_i == 0:              # red
                    colored_image[0, x, y, z] = int(V[x, y, z] > 0) * 255
                    colored_image[1, x, y, z] = 0
                    colored_image[2, x, y, z] = 0
                elif H_i == 1:            # yellow
                    colored_image[0, x, y, z] = int(V[x, y, z] > 0) * 255
                    colored_image[1, x, y, z] = int(V[x, y, z] > 0) * 255
                    colored_image[2, x, y, z] = 0
                elif H_i == 2:            # green
                    colored_image[0, x, y, z] = 0
                    colored_image[1, x, y, z] = int(V[x , y, z] > 0) * 255
                    colored_image[2, x, y, z] = 0
                elif H_i == 3:
                    colored_image[0, x, y, z] = 0
                    colored_image[1, x, y, z] = 0
                    colored_image[2, x, y, z] = 0
                elif H_i == 4:            # blue
                    colored_image[0, x, y, z] = 0
                    colored_image[1, x, y, z] = 0
                    colored_image[2, x, y, z] = int(V[x , y, z] > 0) * 255
                elif H_i == 5:
                    colored_image[0, x, y, z] = 0
                    colored_image[1, x, y, z] = 0
                    colored_image[2, x, y, z] = 0
    return colored_image

def make_3D(image):
    n = 6
    image_3D = (torch.ones(16, 16, 16) * image[0]).permute(1, 2, 0)
#     tresh = np.abs([-1.0]*n + list(np.linspace(-1.0, 1.0, 16-2*n)) + [1.0]*n)
    tresh = torch.tensor([1.0]*n + [0.0]*(16-2*n) + [1.0]*n)
    for i in range(16):
        image_3D[..., i] = image_3D[..., i] > tresh[i]
    return image_3D

from src.utils import get_colored_cat

class MNIST3D(Dataset):
    def __init__(self, number=2, path='data', train=True):
        self.number = number
        self.resize = transforms.Resize(16)
        mnist2D = datasets.MNIST(path, train=train, download=True, transform=transforms.ToTensor())
        self.data = mnist2D.data[mnist2D.targets == self.number]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        out = self.data[idx].unsqueeze(0)
        out = self.resize(out) / 255
        out = make_3D(out)
        colored = get_colored_cat(torch.tensor(out.flatten()).unsqueeze(0))
        # colored  = random_color(out) / 255
        out = torch.tensor(colored)
        return out.float().reshape(3, 16, 16, 16)
    