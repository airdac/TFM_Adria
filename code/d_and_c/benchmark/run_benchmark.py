import numpy as np
import os
from sklearn.datasets import make_swiss_roll
from multiprocessing import freeze_support

from ..methods import DRMethod
from ..utils import log_stdout_and_warnings
from ..examples import benchmark_d_and_c

# Set constant parameters
c_points = 100
method = DRMethod.tSNE

# Parameters to test (logarithmic sequences)
# n_list = [1000,     1778,   3162,   5623,
#      10000,    17783,    31623,  56234,
#      100000,   177828,   316228, 562341,
#      1000000, 3162278,
#      10000000]
n_list = [1000000]
l = 1000
method_arguments = {'perplexity': 30,
                    'n_iter': 250,
                    'verbose': 2}

if __name__ == '__main__':
    np.random.seed(42)

    benchmark_path = os.path.join('d_and_c', 'benchmark')
    plots_path = os.path.join(benchmark_path, 'plots')
    os.makedirs(benchmark_path, exist_ok=True)

    # Start logger
    log_stdout_and_warnings(os.path.join(benchmark_path, 'results.log'))

    # Run benchmark
    freeze_support()    # Fix for parallelization on Windows
    for n in n_list:
        method_arguments_str = [f'{key}={argument}' for key, argument in method_arguments.items()]
        print(
            f"Benchmarking D&C with parallel=True, " + ', '.join(method_arguments_str) + "...")

        print("Generating data...")
        X, color = make_swiss_roll(n_samples=n, random_state=42)

        print("Starting benchmark...")
        benchmark_d_and_c(benchmark_path, 'swiss_roll', X, color, l,
                          c_points, method, method_arguments, system="Windows", parallel=True, runs=1)

        print("Benchmark completed!")
