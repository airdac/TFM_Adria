import numpy as np
import os
from sklearn.datasets import make_swiss_roll
from multiprocessing import freeze_support

from ..methods import DRMethod
from ..utils import log_stdout_and_warnings
from ..examples import benchmark_d_and_c

# Set constant parameters
c_points = 100
method = DRMethod.Isomap

# Parameters to test (logarithmic sequences)
# with parallel=True:
# DONE
#n_parallel = [1000,     1778,   3162,   5623,
#                10000,    17783,    31623,  56234,
#                100000,   177828,   316228, 562341
#                , 1000000, 3162278, 10000000]
#l_parallel = [1000, 3162, 10000]
#n_neighbors_parallel = [7, 10, 15]
n_parallel = [1778279, 5623413]
l_parallel = [3162]
n_neighbors_parallel = [10]

# with parallel=False:
n_linear = [3162,   5623,
            10000,    17783,    31623,  56234,
            100000]
l_linear = 3162
n_neighbors_linear = 10

if __name__ == '__main__':
    np.random.seed(42)

    benchmark_path = os.path.join('d_and_c', 'benchmark')
    plots_path = os.path.join(benchmark_path, 'plots')
    os.makedirs(benchmark_path, exist_ok=True)

    # Start logger
    log_stdout_and_warnings(os.path.join(benchmark_path, 'results.log'))

    # Run benchmark with parallel=True
    freeze_support()    # Fix for parallelization on Windows
    X, color = make_swiss_roll(n_samples=int(1e8), random_state=42)
    method_arguments = {
            "n_neighbors": 10
        }
    benchmark_d_and_c(benchmark_path, 'swiss_roll', X, color, 3162,
                            c_points, method, method_arguments, system = "Windows", parallel=True, runs=1)
    #for l, n_neighbors in zip(l_parallel, n_neighbors_parallel):
    #    method_arguments = {
    #        "n_neighbors": n_neighbors
    #    }
    #    for n in n_parallel:
    #        print(f"Benchmarking D&C with parallel=True, n={n}, l={l}, n_neighbors={n_neighbors}...")
    #        
    #        print("Generating data...")
    #        X, color = make_swiss_roll(n_samples=n, random_state=42)

    #        print("Starting benchmark...")
    #        benchmark_d_and_c(benchmark_path, 'swiss_roll', X, color, l,
    #                        c_points, method, method_arguments, system = "Windows", parallel=True)
            
    #        print("Benchmark completed!")

    # Run benchmark with parallel=False
    #method_arguments = {
    #        "n_neighbors": n_neighbors_linear
    #    }
    #for n in n_linear:
    #    print(f"Benchmarking D&C with parallel=False, n={n}, l={l_linear}, n_neighbors={n_neighbors_linear}...")
        
    #    print("Generating data...")
    #    X, color = make_swiss_roll(n_samples=n, random_state=42)

    #    print("Starting benchmark...")
    #    benchmark_d_and_c(benchmark_path, 'swiss_roll', X, color, l_linear,
    #                    c_points, method, method_arguments, system = "Windows", parallel=False)
        
    #    print("Benchmark completed!")
