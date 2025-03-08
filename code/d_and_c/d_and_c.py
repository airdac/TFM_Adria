import numpy as np
import os

from .methods import DRMethod, get_method_function
from .utils import plot_3D_to_2D
from .private_d_and_c import perform_procrustes, get_partitions_for_divide_conquer, center_and_rotate


def _main_divide_conquer(method: DRMethod,
                         x_filtered: np.ndarray,
                         x_sample_1: np.ndarray,
                         r: int,
                         original_sample_1: np.ndarray,
                         partition_plots_path: str,
                         partition_plots_title: str,
                         color: np.ndarray,
                         **kwargs) -> np.ndarray:
    """
    Process a single partition in the divide and conquer algorithm.

    Parameters:
        method (DRMethod): Dimensionality reduction method to use
        x_filtered (np.ndarray): Data points of the current partition.
        x_sample_1 (np.ndarray): Anchor points sampled from the first partition.
        r (int): Target dimensionality.
        original_sample_1 (np.ndarray): Projection of the anchor points from the first partition.
        partition_plots_path (str): Path to store the partition's visualization.
        partition_plots_title (str): Title of the partition's visualization.
        color (np.ndarray): Color information for visualization.
        kwargs (Any): Additional method-specific parameters.

    Returns:
        projection (np.ndarray): The aligned projection of the current partition.
        """
    projection_method = get_method_function(method)

    # Combine anchor points and partition data
    x_join_sample_1 = np.vstack((x_sample_1, x_filtered))

    # Apply projection method
    projection = projection_method(x_join_sample_1, r, **kwargs)

    # Visualize results
    plot_3D_to_2D(x_join_sample_1, projection,
                  method, partition_plots_title, color, partition_plots_path)

    # Extract results and align using Procrustes
    n_sample = x_sample_1.shape[0]
    projection_sample_1 = projection[:n_sample, :]
    projection_partition = projection[n_sample:, :]

    return perform_procrustes(
        projection_sample_1, original_sample_1, projection_partition, translation=False)


def divide_conquer(method: DRMethod,
                   x: np.ndarray,
                   l: int,
                   c_points: int,
                   r: int,
                   color: np.ndarray,
                   dataset_name: str,
                   **kwargs) -> np.ndarray:
    """
    Apply divide and conquer to a dimensionality reduction method.

    Parameters:
        method (DRMethod): Dimensionality reduction method to use.
        x (np.ndarray): Input data matrix.
        l (int): Partition size.
        c_points (int): Number of common points.
        r (int): Target dimensionality.
        color (np.ndarray): Colors for visualization.
        dataset_name (str): Name of the dataset (used in results folder naming).
        kwargs (Any): Additional method-specific parameters.

    Returns:
        projection (np.ndarray): Low-dimensional representation of the data.
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

    # Create directory for results
    kwargs_str = [f'{key}_{value}' for key, value in kwargs.items()]
    results_path = os.path.join('d_and_c',
                                'results',
                                dataset_name,
                                f'n_{n_row_x}',
                                f'l_{l}',
                                f'c_{c_points}',
                                str(method),
                                *kwargs_str,
                                "d_and_c_partition_plots"
                                )

    # Save first partition visualization
    fig_title = f'D&C {method} on {dataset_name} with n={n_row_x}, l={l}, c_points={c_points}, {", ".join([f'{key}={value}' for key, value in kwargs.items(
    )])}'
    plot_3D_to_2D(
        x=x_1,
        projection=projection_1,
        method=str(method),
        title=fig_title + '. Part 1',
        color=color[idx_list[0]],
        path=os.path.join(results_path, 'part1'),
        empty=True
    )

    # Sample anchor points from first partition
    sample_1_idx = np.random.choice(length_1, size=c_points, replace=False)
    x_sample_1 = x_1[sample_1_idx, :]
    projection_sample_1 = projection_1[sample_1_idx, :]

    # Process remaining partitions
    projections = [None] * (num_partitions - 1)
    for iteration, idx in enumerate(idx_list[1:]):
        print(f"Projecting partition {iteration + 2}...")

        # Get colors for visualization (anchor points + current partition)
        total_color = color[np.concatenate((idx_list[0][sample_1_idx], idx))]

        # Process the partition
        projections[iteration] = _main_divide_conquer(
            method=method,
            x_filtered=x[idx, :],
            x_sample_1=x_sample_1,
            r=r,
            original_sample_1=projection_sample_1,
            partition_plots_path=os.path.join(
                results_path, f'part{iteration+2}'),
            partition_plots_title=fig_title + f'. Part {iteration + 2}',
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

    # Instead of using NumPy calls for centering, covariance, eigen-decomposition,
    # we pass the combined projection to our optimized center_and_rotate function.
    return center_and_rotate(combined_projection)

    # Center and rotate for maximum variance
    # combined_projection = combined_projection - \
    #     np.mean(combined_projection, axis=0)
    # cov_matrix = np.cov(combined_projection, rowvar=False)
    # eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    # idx_sort = np.argsort(eigenvals)[::-1]
    # eigenvecs = eigenvecs[:, idx_sort]

    # return combined_projection @ eigenvecs
