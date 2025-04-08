import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

from .methods import DRMethod, get_method_function
from .utils import plot_3D_to_2D
from .private_d_and_c import perform_procrustes, get_partitions_for_divide_conquer


def _main_divide_conquer(args: tuple) -> np.ndarray:
    """
    Process a single partition in the divide and conquer algorithm.

    Parameters:
        args (tuple) : tuple of the following arguments:
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
    method, x_filtered, x_sample_1, r, original_sample_1, plot, kwargs = args
    projection_method = get_method_function(method)

    # Combine anchor points and partition data
    x_join_sample_1 = np.vstack((x_sample_1, x_filtered))

    # Apply projection method
    projection = projection_method(
        x_join_sample_1, r, principal_components=False, **kwargs)

    # Visualize results
    if plot:
        plot_3D_to_2D(x_join_sample_1, projection,
                    method, **plot)

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
                   parallel: bool | None = False,
                   plot: dict | None = None,
                   **kwargs) -> np.ndarray:
    """
    Apply divide and conquer to a dimensionality reduction method.

    Parameters:
        method (DRMethod): Dimensionality reduction method to use.
        x (np.ndarray): Input data matrix.
        l (int): Partition size.
        c_points (int): Number of common points.
        r (int): Target dimensionality.
        plot (dict): if not None, plot the embedding. Moreover, if not None, it must be a dict with the arguments for the plot: 'color' (np.ndarray), 'dataset_name' (str).
        kwargs (Any): Additional method-specific parameters.

    Returns:
        projection (np.ndarray): Low-dimensional representation of the data.
    """
    projection_method = get_method_function(method)
    n_row_x = x.shape[0]

    # For small datasets, apply the method directly
    if n_row_x <= l:
        return projection_method(x, r, principal_components=True, **kwargs)

    # Create partitions
    idx_list = get_partitions_for_divide_conquer(n_row_x, l, c_points, r)
    num_partitions = len(idx_list)
    length_1 = len(idx_list[0])

    # Process first partition
    print("Projecting partition 1...")
    x_1 = x[idx_list[0],]
    projection_1 = projection_method(
        x_1, r, principal_components=False, **kwargs)

    if plot:
        # Create directory for results
        kwargs_str = [f'{key}_{value}' for key, value in kwargs.items()]
        results_path = os.path.join('d_and_c',
                                    'results',
                                    plot["dataset_name"],
                                    f'n_{n_row_x}',
                                    f'l_{l}',
                                    f'c_{c_points}',
                                    str(method),
                                    *kwargs_str,
                                    "d_and_c_partition_plots"
                                    )

         # Save first partition visualization
        fig_title = f'D&C {method} on {plot["dataset_name"]} with n={n_row_x}, l={l}, c_points={c_points}, {", ".join([f'{key}={value}' for key, value in kwargs.items(
        )])}'
        plot_3D_to_2D(
            x=x_1,
            projection=projection_1,
            method=str(method),
            title=fig_title + '. Part 1',
            color=plot["color"][idx_list[0]],
            path=os.path.join(results_path, 'part1'),
            empty=True
        )

    # Sample connecting points from first partition
    sample_1_idx = np.random.choice(length_1, size=c_points, replace=False)
    x_sample_1 = x_1[sample_1_idx, :]
    projection_sample_1 = projection_1[sample_1_idx, :]

    # Process remaining partitions
    # Build plot arguments
    if not plot:
        partition_plot = [None]*(num_partitions-1)
    else:
        partition_plot = [{"path": os.path.join(results_path, f'part{i + 2}'),
                            "color": np.concatenate((idx_list[0][sample_1_idx], idx)),
                            "title": fig_title + f'. Part {i + 2}'
                            } for i,idx in enumerate(idx_list[1:])]
    # Build list of arguments
    args_list = [
        (
            method,
            x[idx, :],
            x_sample_1,
            r,
            projection_sample_1,
            partition_plot[i],
            kwargs.copy()  # extra method-specific arguments
        )
        for i, idx in enumerate(idx_list[1:])
    ]
    if parallel:
        print("Projecting remaining partitions in parallel...")
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_main_divide_conquer, args)
                       for args in args_list]
            projections = [future.result() for future in futures]
    else:
        projections = [None] * (num_partitions - 1)
        for i in range(num_partitions-1):
            print(f"Projecting partition {i + 2}...")
            projections[i] = _main_divide_conquer(args_list[i])

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
