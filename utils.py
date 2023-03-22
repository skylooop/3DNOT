from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import numpy as np

def plot3d_static(x):
    img = x.reshape(3, 16, 16, 16)

    mask = img.astype(np.bool_)
    mask = mask[0] | mask[1] | mask[2]

    voxelarray = mask
    colors = np.array([rgb2hex(rgb) for rgb in x.reshape(3, -1).T])
    colors = colors.reshape(*img.shape[1:])

    ax = plt.figure().add_subplot(projection='3d', )
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

    ax.axis('off')
    ax.azim = 10
    ax.elev = 80
    plt.show()

