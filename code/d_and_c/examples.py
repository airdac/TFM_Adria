import time
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll

from .methods import DRMethod
from .d_and_c import divide_conquer
from .utils import plot_3D_to_2D


def swiss_roll_example():
    """Example using Swiss Roll dataset."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Set parameters
    n = 10000
    l = 1000
    c_points = 100
    method = DRMethod.Isomap
    n_neighbors = 8

    # Generate data
    X, color = make_swiss_roll(n_samples=n, random_state=42)

    # Run divide and conquer
    start_time = time.time()
    d_and_c_result = divide_conquer(
        method=method,
        x=X,
        l=l,
        c_points=c_points,
        r=2,
        color=color,
        n_neighbors=n_neighbors
    )
    runtime = time.time() - start_time
    print(f"D&C runtime: {runtime:.2f} seconds")

    # Visualize and save final result
    fig = plot_3D_to_2D(color, X, d_and_c_result, str(method))

    # Create output directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Save final visualization
    plot_filename = f'dc_{method}-n{n}-l{l}-c{c_points}-n_neighbors{n_neighbors}'
    os.makedirs(os.path.join('figures', plot_filename), exist_ok=True)
    plt.savefig(os.path.join('figures', plot_filename, f'{plot_filename}.png'))
    plt.close()

    print(f"Visualization saved as '{plot_filename}.png'")


if __name__ == "__main__":
    swiss_roll_example()
