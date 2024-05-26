import numpy as np
import ipyvolume as p3

import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.util import montage as montage2d

def show_3d_mesh(image, thresholds, edgecolor='none', alpha=0.5):
    p = image[::-1].swapaxes(1, 2)
    cmap = plt.get_cmap('nipy_spectral_r')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, c_threshold in list(enumerate(thresholds)):
        verts, faces, _, _ = marching_cubes(p, c_threshold)
        mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor=edgecolor, linewidth=0.1)
        mesh.set_facecolor(cmap(i / len(thresholds))[:3])
        ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0]); ax.set_ylim(0, p.shape[1]); ax.set_zlim(0, p.shape[2])

    ax.view_init(45, 45)
    return fig

def interactive_3d(img):
    fig = p3.figure()
    # create a custom LUT
    temp_tf = plt.cm.nipy_spectral(np.linspace(0, 1, 256))
    # make transparency more aggressive
    temp_tf[:, 3] = np.linspace(-.3, 0.5, 256).clip(0, 1)
    tf = p3.transferfunction.TransferFunction(rgba=temp_tf)
    p3.plot_isosurface((img/img.max()).astype(np.float32), level=0.85)

    p3.show()
    
def montage_pad(x): return montage2d(
    np.pad(x, [(0, 0), (10, 10), (10, 10)], mode='constant', constant_values=0))