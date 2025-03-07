import matplotlib.pyplot as plt
import os
import shutil
import numpy as np


def plot_3D_to_2D(color, x, projection, method, path=None, new_directory=None):
    """Plot a 3D dataset and its 2D projection."""
    fig = plt.figure(figsize=(14, 8))

    # Original data 3D plot
    ax1 = fig.add_subplot(234, projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)
    ax1.set_title("Original Data")

    # Original data 2D projections
    ax2 = fig.add_subplot(231)
    ax2.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_title("Original Data (dims 1,2)")

    ax3 = fig.add_subplot(232)
    ax3.scatter(x[:, 0], x[:, 2], c=color, cmap=plt.cm.Spectral)
    ax3.set_title("Original Data (dims 1,3)")

    ax4 = fig.add_subplot(233)
    ax4.scatter(x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)
    ax4.set_title("Original Data (dims 2,3)")

    # DR method projection
    ax5 = fig.add_subplot(235)
    ax5.scatter(projection[:, 0], projection[:, 1],
                c=color, cmap=plt.cm.Spectral)
    ax5.set_title(f"D&C {method} Embedding")

    plt.tight_layout()

    if new_directory:
        if os.path.exists(new_directory):
            shutil.rmtree(new_directory)
        os.makedirs(new_directory)

    if path:
        plt.savefig(path)
        plt.close()
    else:
        return fig
