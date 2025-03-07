import time
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll

from .methods import DRMethod, get_method_function
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
    method_arguments = {
        "n_neighbors": 8
    }
    bare_method_arguments = {
        "n_neighbors": int(np.floor(method_arguments["n_neighbors"]*np.log(np.e + n//l)))
    }
    # Generate data
    print("Generating data...")
    X, color = make_swiss_roll(n_samples=n, random_state=42)

    # Run divide and conquer
    print(f'Running D&C {method}...')
    start_time = time.time()
    d_and_c_result = divide_conquer(
        method=method,
        x=X,
        l=l,
        c_points=c_points,
        r=2,
        color=color,
        **method_arguments
    )
    runtime = time.time() - start_time
    print(f"D&C runtime: {runtime:.2f} seconds")

    fig = plot_3D_to_2D(color, X, d_and_c_result, str(method))

    # Run bare method
    print(f'Running bare {method}...')
    start_time = time.time()
    bare_results = get_method_function(method)(x=X, r=2, **bare_method_arguments)
    bare_runtime = time.time() - start_time
    print(f"Bare runtime: {bare_runtime:.2f} seconds")

    ax6 = fig.add_subplot(236)
    ax6.scatter(bare_results[:, 0], bare_results
                [:, 1], c=color, cmap=plt.cm.Spectral)
    bare_method_arguments_str = [f'{key}_{value}' for key,
                            value in bare_method_arguments.items()]
    ax6.set_title(
        f"Bare {method} embedding {' '.join(bare_method_arguments_str)}")

    # Save results
    method_arguments_str = [f'{key}_{value}' for key,
                            value in method_arguments.items()]
    results_path = os.path.join('d_and_c',
                                'results',
                                str(method),
                                f'n_{n}',
                                f'l_{l}',
                                f'c_{c_points}',
                                *method_arguments_str
                                )
    plt.savefig(os.path.join(results_path, "d_and_c-vs-bare"))
    plt.close()
    print(f"Results saved in '{results_path}'")


if __name__ == "__main__":
    swiss_roll_example()
