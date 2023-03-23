from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot3d_static(x):
    img = x.reshape(3, 16, 16, 16)

    mask = img.to(torch.bool)
    mask = mask[0] | mask[1] | mask[2]

    colors = np.array([rgb2hex(rgb) for rgb in x.numpy().reshape(3, -1).T])
    colors = colors.reshape(*img.shape[1:])

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask, facecolors=colors)
    ax.axis('off')
#     ax.azim = 10
#     ax.elev = 80  
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