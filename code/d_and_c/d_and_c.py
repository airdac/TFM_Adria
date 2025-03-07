import numpy as np
import os

from .methods import get_method_function
from .utils import plot_3D_to_2D
from .private_d_and_c import perform_procrustes, get_partitions_for_divide_conquer


def _main_divide_conquer(method, x_filtered, x_sample_1, r, original_sample_1,
                         partition_plots_path, color, **kwargs):
    """Process a single partition in the divide and conquer algorithm."""
    projection_method = get_method_function(method)

    # Combine anchor points and partition data
    x_join_sample_1 = np.vstack((x_sample_1, x_filtered))

    # Apply projection method
    projection = projection_method(x_join_sample_1, r, **kwargs)

    # Visualize results
    plot_3D_to_2D(color, x_join_sample_1, projection,
                  method, partition_plots_path)

    # Extract results and align using Procrustes
    n_sample = x_sample_1.shape[0]
    projection_sample_1 = projection[:n_sample, :]
    projection_partition = projection[n_sample:, :]

    return perform_procrustes(
        projection_sample_1, original_sample_1, projection_partition, translation=False)


def divide_conquer(method, x, l, c_points, r, color, **kwargs):
    """
    Apply divide and conquer dimensionality reduction.

    Parameters:
        method: DRMethod - Dimensionality reduction method to use
        x: np.ndarray - Input data matrix (n_samples, n_features)
        l: int - Maximum partition size
        c_points: int - Number of common/anchor points
        r: int - Target dimensionality
        color: np.ndarray - Colors for visualization
        **kwargs: Additional method-specific parameters

    Returns:
        np.ndarray - Low-dimensional representation of the data
    """
    projection_method = get_method_function(method)
    n_row_x = x.shape[0]

    # For small datasets, apply the method directly
    if n_row_x <= l:
        return projection_method(x, r, **kwargs)

    # Create partitions
    idx_list = get_partitions_for_divide_conquer(n_row_x, l, c_points, r)
    num_partitions = len(idx_list)
    length_1 = len(idx_list[0])

    # Process first partition
    print("Projecting partition 1...")
    x_1 = x[idx_list[0],]
    projection_1 = projection_method(x_1, r, **kwargs)

    # Create directory for visualizations
    method_str = str(method)
    partition_plots_directory = f'dc_{method_str}-n{n_row_x}-l{l}-c{c_points}'
    if 'n_neighbors' in kwargs:
        partition_plots_directory += f'-n_neighbors{kwargs["n_neighbors"]}'

    # Save first partition visualization
    partition_plots_filename = f'{partition_plots_directory}-part1'
    plot_3D_to_2D(
        color=color[idx_list[0]],
        x=x_1,
        projection=projection_1,
        method=method_str,
        path=os.path.join('figures', partition_plots_directory,
                          partition_plots_filename),
        new_directory=os.path.join('figures', partition_plots_directory)
    )

    # Sample anchor points from first partition
    sample_1_idx = np.random.choice(length_1, size=c_points, replace=False)
    x_sample_1 = x_1[sample_1_idx, :]
    projection_sample_1 = projection_1[sample_1_idx, :]

    # Process remaining partitions
    projections = [None] * (num_partitions - 1)
    for iteration, idx in enumerate(idx_list[1:]):
        print(f"Projecting partition {iteration + 2}...")
        partition_plots_path = os.path.join(
            'figures',
            partition_plots_directory,
            f'{partition_plots_directory}-part{iteration+2}'
        )

        # Get colors for visualization (anchor points + current partition)
        total_color = color[np.concatenate((idx_list[0][sample_1_idx], idx))]

        # Process the partition
        projections[iteration] = _main_divide_conquer(
            method=method,
            x_filtered=x[idx, :],
            x_sample_1=x_sample_1,
            r=r,
            original_sample_1=projection_sample_1,
            partition_plots_path=partition_plots_path,
            color=total_color,
            **kwargs
        )

    # Combine all projections
    all_projections = [projection_1] + projections
    combined_projection = np.vstack(all_projections)

    # Reorder rows to match original data order
    order_idx = np.concatenate(idx_list)
    order = np.argsort(order_idx)
    combined_projection = combined_projection[order, :]

    # Center and rotate for maximum variance
    combined_projection = combined_projection - \
        np.mean(combined_projection, axis=0)
    cov_matrix = np.cov(combined_projection, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    idx_sort = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx_sort]

    return combined_projection @ eigenvecs
