import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import pandas as pd
from typing import Optional

from .methods import DRMethod, get_method_function
from .d_and_c import divide_conquer
from .utils import benchmark, plot_3D_to_2D


def example_3D_to_2D(x: np.ndarray,
                     l: int,
                     c_points: int,
                     method: DRMethod,
                     method_arguments: dict,
                     bare_method_arguments: dict,
                     dataset_name: str,
                     color: np.ndarray) -> None:
    """
    Example projecting 3D data to 2D using both a divide-and-conquer method 
    and a bare dimensionality reduction method.

    Parameters:
        x (np.ndarray): Input data matrix.
        l (int): Partition size.
        c_points (int): Number of common points.
        method (DRMethod): Dimensionality reduction method to use.
        method_arguments (dict): Additional parameters for the D&C method.
        bare_method_arguments (dict): Additional parameters for the bare method.
        dataset_name (str): Name of the dataset (used for naming results).
        color (np.ndarray): Color information for visualization.
    """

    n = x.shape[0]

    # Run divide and conquer
    print(f'Running D&C {method}...')
    d_and_c_result, d_and_c_runtime = benchmark(divide_conquer,
                                                method=method,
                                                x=x,
                                                l=l,
                                                c_points=c_points,
                                                r=2,
                                                color=color,
                                                dataset_name=dataset_name,
                                                **method_arguments
                                                )
    print(f"D&C runtime: {d_and_c_runtime:.2f} seconds")

    fig_title = f'D&C vs bare {method} on {dataset_name} with n={n}, l={l}, c_points={c_points}. In D&C {method}, {", ".join([f'{key}={value}' for key, value in method_arguments.items(
    )])}'
    fig = plot_3D_to_2D(x, d_and_c_result, str(method), fig_title, color)

    # Run bare method
    print(f'Running bare {method}...')
    bare_results, bare_runtime = benchmark(
        get_method_function(method),
        x=x, r=2, **bare_method_arguments
    )
    print(f"Bare runtime: {bare_runtime:.2f} seconds")

    ax6 = fig.add_subplot(236)
    ax6.scatter(bare_results[:, 0], bare_results
                [:, 1], c=color, cmap=plt.cm.Spectral)
    bare_method_arguments_str = [f'{key}_{value}' for key,
                                 value in bare_method_arguments.items()]
    ax6.set_title(
        f"Bare {method} embedding {' '.join(bare_method_arguments_str)}")

    #  Save results
    method_arguments_str = [f'{key}_{value}' for key,
                            value in method_arguments.items()]
    results_path = os.path.join('d_and_c',
                                'results',
                                dataset_name,
                                f'n_{n}',
                                f'l_{l}',
                                f'c_{c_points}',
                                str(method),
                                *method_arguments_str
                                )
    plt.savefig(os.path.join(results_path, "d_and_c-vs-bare.png"))
    plt.close()

    results_file = os.path.join(results_path, "results.txt")
    parameters_message = f"""Results of performing D&C and bare {method} on the {dataset_name} dataset with:
    n={n}
    l={l}
    c_points={c_points}
In D&C {method}:
    {"\n\t".join([f'{key}={value}' for key,
                  value in method_arguments.items()])}
In bare {method}:
    {"\n\t".join([f'{key}={value}' for key,
                      value in bare_method_arguments.items()])}

"""

    with open(results_file, "w") as f:
        f.write(parameters_message)
        f.write(f"D&C runtime: {d_and_c_runtime:.2f} seconds\n")
        f.write(f"Bare runtime: {bare_runtime:.2f} seconds\n")
    print(f"Results saved in '{results_path}'")


def benchmark_d_and_c(output_path: str,
                      dataset_name: str,
                      x: np.ndarray,
                      l: int,
                      c_points: int,
                      method: DRMethod,
                      method_arguments: dict,
                      parallel: Optional[bool] = False,
                      runs: Optional[int] = 20) -> None:
    for run in range(runs):
        print(f"Benchmark run {run + 1}/{runs}...")
        _, runtime = benchmark(divide_conquer,
                               method=method,
                               x=x,
                               l=l,
                               c_points=c_points,
                               r=2,
                               plot=False,
                               parallel=parallel,
                               **method_arguments
                               )
        df = pd.DataFrame(runtime, columns=["time_ns"])
        df["dataset"] = dataset_name
        df["n"] = x.size[0]
        df["l"] = l
        df["c_points"] = c_points
        df["method"] = str(method)
        for arg, value in method_arguments.items():
            df[arg] = value

        output_path = os.path.join(output_path, 'results.csv')
        if not os.path.exists(output_path):
            df.to_csv(output_path, index=False)
        else:
            df.to_csv(output_path, index=False, mode='a', header=False)


if __name__ == "__main__":
    np.random.seed(42)

    benchmark_path = os.path.join('d_and_c', 'benchmark')
    os.makedirs(benchmark_path, exist_ok=True)

    # Parameters to test (logarithmic sequences)
    # n = [1000, 1778, 3162, 5623, 10000,
    #      17783, 31623, 56234, 100000,
    #      177828,  316228,  562341, 1000000]
    # l = [1000, 3162, 10000]
    # c_points = 100
    # method = DRMethod.Isomap
    # method_arguments = {
    #     "n_neighbors": [7, 10, 15]
    # }

    # Set parameters
    n = 1000
    l = 1000
    c_points = 100
    method = DRMethod.Isomap
    method_arguments = {
        "n_neighbors": 7
    }

    # Generate data
    print("Generating data...")
    X, color = make_swiss_roll(n_samples=n, random_state=42)

    print("Benchmarking D&C...")
    benchmark_d_and_c(benchmark_path, 'swiss_roll', X, l,
                      c_points, method, method_arguments, parallel=True)
