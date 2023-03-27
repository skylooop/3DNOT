from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial
from skimage import color
import os

def plot3d_static(x):
    img = x.reshape(3, 16, 16, 16)

    mask = img.to(torch.bool)
    mask = mask[0] | mask[1] | mask[2]

    colors = np.array([rgb2hex(rgb) for rgb in x.numpy().reshape(3, -1).T])
    colors = colors.reshape(*img.shape[1:])

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask, facecolors=colors)
    ax.axis('off')
    plt.show()

def plot3d_static_batch(X, azim=10, elev=80):
    img = X.reshape(-1, 3, 16, 16, 16)

    mask = img.to(torch.bool)
    mask = mask[:, 0] | mask[:, 1] | mask[:, 2]

    n = X.shape[0]
    fig = plt.figure(figsize=(15, 3))
    for i, x in enumerate(X):
        colors = np.array([rgb2hex(rgb) for rgb in x.numpy().reshape(3, -1).T])
        colors = colors.reshape(*img.shape[2:])

        ax = fig.add_subplot(100+10*n+1+i, projection='3d')
        ax.voxels(mask[i], facecolors=colors)
        ax.azim = azim
        ax.elev = elev
        ax.axis('off')
    plt.show()


def get_colored_uniform(imgs, thr = None):
    # for flattened image
    # Uniformly sample color
    assert len(imgs.shape) == 2
    init_shape = imgs.shape
    imgs = torch.unsqueeze(imgs, dim=-1)

    result = np.zeros((*imgs.shape, 3), float) 

    seeds = torch.Tensor(np.random.uniform(size=(init_shape[0], 2)))

    seeds = torch.broadcast_to(seeds, (*init_shape[::-1], 2))
    seeds = torch.permute(seeds, [1, 0, 2])

    result = torch.concat([seeds, imgs], dim=-1)

    result = torch.Tensor(color.hsv2rgb(result))
    return result


def get_colored_cat(imgs, thr = None):
    # Sample color from categorical distribution
    assert len(imgs.shape) == 2
    init_shape = imgs.shape
    imgs = torch.unsqueeze(imgs, dim=-1)
    
    result = np.zeros((*imgs.shape, 3), float)
    
    clrs = np.array([
        [0.2, 1], 
        [0.5, 1], 
        [0.95, 1]
    ])
    seeds = np.random.choice(range(3), size = init_shape[0])
    seeds = torch.Tensor(clrs[seeds])
    #print(seeds.shape)
    #torch.Tensor(np.random.uniform(size=(init_shape[0], 2)))

    seeds = torch.broadcast_to(seeds, (*init_shape[::-1], 2))
    seeds = torch.permute(seeds, [1, 0, 2])

    result = torch.concat([seeds, imgs], dim=-1)
    
    result = torch.Tensor(color.hsv2rgb(result))

    result = torch.transpose(result, 2, 1)

    return result

def save_checkpoint_and_pics(T, f, step):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    from torch.utils.data import DataLoader
    from src.dataset import MNIST3D
    from datetime import datetime

    path_name = os.path.join('checkpoints', datetime.now().isoformat(timespec='seconds') + f'_step_{step}')
    
    os.mkdir(path_name)

    dataset_2 = MNIST3D(number=2)
    train_dataloader_2 = DataLoader(dataset_2, batch_size=6, shuffle=False)
    
    sample_x = next(iter(train_dataloader_2))


    img = sample_x.reshape(-1, 3, 16, 16, 16)

    mask = img.to(torch.bool)
    mask = mask[:, 0] | mask[:, 1] | mask[:, 2]

    n = sample_x.shape[0]
    fig = plt.figure(figsize=(15, 3))
    for i, x in enumerate(sample_x):
        colors = np.array([rgb2hex(rgb) for rgb in x.numpy().reshape(3, -1).T])
        colors = colors.reshape(*img.shape[2:])

        ax = fig.add_subplot(100+10*n+1+i, projection='3d')
        ax.voxels(mask[i], facecolors=colors)
        ax.axis('off')

    plt.savefig(os.path.join(path_name, 'before_transport.png'))


    T_x = T(sample_x).detach().cpu()

    threshold = torch.nn.Threshold(0.2, 0, inplace=False)
    T_x = threshold(torch.clip(T_x, min=0, max=1))

    img = T_x.reshape(-1, 3, 16, 16, 16)

    mask = img.to(torch.bool)
    mask = mask[:, 0] | mask[:, 1] | mask[:, 2]

    n = sample_x.shape[0]
    fig = plt.figure(figsize=(15, 3))
    for i, x in enumerate(T_x):
        colors = np.array([rgb2hex(rgb) for rgb in x.numpy().reshape(3, -1).T])
        colors = colors.reshape(*img.shape[2:])

        ax = fig.add_subplot(100+10*n+1+i, projection='3d')
        ax.voxels(mask[i], facecolors=colors)
        ax.axis('off')

    plt.savefig(os.path.join(path_name, 'after_transport.png'))
    torch.save(T.state_dict(), os.path.join(path_name, f'T_checkpoint_step_{step}.pth'))
    torch.save(f.state_dict(), os.path.join(path_name, f'f_checkpoint_step_{step}.pth'))
