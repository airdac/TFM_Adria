import math

import numba as nb
import numpy as np
import sklearn
from numba import cuda
import matplotlib.pyplot as plt

from synthetic import Synthetic
from sklearn.metrics import pairwise_distances


def setup_mds():
    n_samples_ref = 500
    n_samples_batch = 1000
    n_batches = 100

    dataset = Synthetic(16)
    ref_dataset, ref_labels = dataset.draw_samples(n_samples_ref)

    weights = np.ones([len(ref_dataset), len(ref_dataset)], dtype=np.float32)
    variable_points = np.arange(0, len(ref_dataset), dtype=np.int32)

    ref_low_dim_unoptimized = gen_random_points(len(ref_dataset), 2)

    ref_low_dim_optimized = mds_gradient_descent_gpu(ref_dataset,
                                                 ref_low_dim_unoptimized,
                                                 weights,
                                                 variable_points,
                                                 iterations=200,
                                                 alpha=0.0001)

    concatenated_dataset = ref_low_dim_optimized
    concatenated_labels = ref_labels

    for i in range(n_batches):
        print("Batch", i)
        oos_dataset, oos_labels = dataset.draw_samples(n_samples_batch)
        ref_oos_dataset = np.concatenate((ref_dataset, oos_dataset))

        oos_batch_unoptimized = gen_random_points(len(oos_dataset), 2)
        ref_oos_unoptimized = np.concatenate((ref_low_dim_optimized, oos_batch_unoptimized))

        variable_points = np.arange(len(ref_dataset), len(ref_dataset) + len(oos_dataset), dtype=np.int32)
        weights = np.zeros([len(ref_oos_dataset), len(ref_oos_dataset)], dtype=np.float32)
        weights[:, 0:ref_dataset.shape[0]] = 1

        ref_oos_optimized = mds_gradient_descent_gpu(ref_oos_dataset,
                                                     ref_oos_unoptimized,
                                                     weights,
                                                     variable_points,
                                                     iterations=500,
                                                     alpha=0.0001)

        concatenated_dataset = np.concatenate((concatenated_dataset, ref_oos_optimized[ref_dataset.shape[0]:]))
        concatenated_labels = np.concatenate((concatenated_labels, oos_labels))


    plt.scatter(concatenated_dataset[:, 0], concatenated_dataset[:, 1], c=concatenated_labels, s=1, marker='o')
    plt.show()


def mds_gradient_descent_gpu(high_dim: np.ndarray,
                             init_projection: np.ndarray,
                             weights: np.ndarray,
                             variable_points: np.ndarray,
                             iterations: int,
                             alpha: float = 0.0001) -> np.ndarray:
    low_dim = init_projection
    distance_matrix_high_dim = pairwise_distances(high_dim, high_dim, n_jobs=-1)
    distance_matrix_high_dim = distance_matrix_high_dim.astype(np.float32)
    distance_matrix_high_dim_d = cuda.to_device(distance_matrix_high_dim)

    weights_d = cuda.to_device(weights)
    variable_points_d = cuda.to_device(variable_points)

    threadsperblock = 32
    blockspergrid_1d = (math.ceil(high_dim.shape[0] / threadsperblock))

    deltas = np.zeros(shape=(low_dim.shape[0], low_dim.shape[1]), dtype=np.float32)
    deltas_d = cuda.to_device(deltas)

    low_dim_d = cuda.to_device(low_dim)
    for i in range(iterations):
        _calculate_deltas_gpu[blockspergrid_1d, threadsperblock](low_dim_d,
                                                                 distance_matrix_high_dim_d,
                                                                 weights_d,
                                                                 variable_points_d,
                                                                 deltas_d)
        update_values[blockspergrid_1d, threadsperblock](low_dim_d, variable_points_d, alpha, deltas_d)
    low_dim = low_dim_d.copy_to_host()
    return low_dim


@cuda.jit
def _calculate_deltas_gpu(low_dim,
                          distance_matrix_high_dim,
                          weights,
                          variable_points,
                          deltas):
    row = cuda.grid(1)
    if row < low_dim.shape[0] and row < variable_points.shape[0]:
        i = variable_points[row]
        g_x = 0
        g_y = 0
        for col in range(low_dim.shape[0]):
            d_x = low_dim[i, 0] - low_dim[col, 0]
            d_y = low_dim[i, 1] - low_dim[col, 1]
            dist_low = math.sqrt(d_x ** 2 + d_y ** 2) + 0.0001
            g_x += 2 * weights[i][col] * (1 - (distance_matrix_high_dim[i][col] / dist_low)) * (
                        low_dim[i][0] - low_dim[col][0])
            g_y += 2 * weights[i][col] * (1 - (distance_matrix_high_dim[i][col] / dist_low)) * (
                        low_dim[i][1] - low_dim[col][1])

        deltas[i, 0] = g_x
        deltas[i, 1] = g_y


@cuda.jit
def update_values(low_dim, variable_points, alpha, deltas):
    idx = cuda.grid(1)
    if idx < variable_points.shape[0]:
        row = variable_points[idx]
        for k in range(low_dim.shape[1]):
            low_dim[row][k] = low_dim[row][k] - (alpha * deltas[row][k])


def gen_random_points(n_points: int, n_dimensions: int) -> np.ndarray:
    random_points = np.zeros(shape=(n_points, n_dimensions), dtype=float)
    for i in range(n_points):
        for j in range(n_dimensions):
            random_points[i][j] = np.random.rand()
    return random_points


if __name__ == "__main__":
    setup_mds()
