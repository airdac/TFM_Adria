import numpy as np
from sklearn.manifold import Isomap


def isomap(x, r=2, n_neighbors=5):
    """
    Perform Isomap on data matrix x.

    Parameters:
        x : np.ndarray
            Data matrix of shape (n, k).
        r : int
            Dimensionality of the projection.
        n_neighbors : int
            Number of neighbors for Isomap.

    Returns:
        A dictionary with keys:
          'points': np.ndarray of shape (n, r) with the embedding,
          'eigen' : np.ndarray of the first r eigenvalues from the internal kernel PCA.
    """
    isomap = Isomap(n_neighbors=n_neighbors, n_components=r)
    points = isomap.fit_transform(x)
    eigen = isomap.kernel_pca_.eigenvalues_[:r]
    return {'points': points, 'eigen': eigen}


def get_procrustes_parameters(x, target, translation=False):
    """
    Compute Procrustes rotation (and translation if requested)
    to align x to target (see, for instance, p.432, Chapter 20 of Borg and Groenen 2005).
    """
    if translation:
        m_target = np.mean(target, axis=0)
        m_x = np.mean(x, axis=0)
        c_target = target - m_target
        M = c_target.T @ x
        U, _, Vt = np.linalg.svd(M)
        rotation_matrix = Vt.T @ U.T
        translation_vector = m_target - rotation_matrix.T @ m_x
    else:
        M = target.T @ x
        U, _, Vt = np.linalg.svd(M)
        rotation_matrix = Vt.T @ U.T
        translation_vector = np.zeros(x.shape[1])
    return rotation_matrix, translation_vector


def perform_procrustes(x, target, matrix_to_transform, translation=False):
    """
    Align 'matrix_to_transform' to 'target' using Procrustes transformation.
    """
    if x.shape != target.shape:
        raise ValueError("x and target do not have the same shape")
    if x.shape[1] != matrix_to_transform.shape[1]:
        raise ValueError(
            "x and matrix_to_transform do not have the same number of columns")
    rotation_matrix, translation_vector = get_procrustes_parameters(
        x, target, translation)
    return matrix_to_transform @ rotation_matrix + translation_vector


def get_partitions_for_divide_conquer(n, l, c_points, r):
    """
    Divide indices 0,...,n-1 into partitions.
    """
    if l <= c_points:
        raise ValueError("l must be greater than c_points")
    if l - c_points < c_points:
        raise ValueError("l-c_points must be greater than c_points")
    if l - c_points <= r:
        raise ValueError("l-c_points must be greater than r")

    permutation = np.random.permutation(n)
    if n <= l:
        return [permutation]
    else:
        min_sample_size = max(r + 2, c_points)
        p = int(np.ceil(1 + (n - l) / (l - c_points)))
        last_partition_sample_size = n - (l + (p - 2) * (l - c_points))
        if 0 < last_partition_sample_size < min_sample_size:
            p = p - 1
            last_partition_sample_size = n - (l + (p - 2) * (l - c_points))
        first_partition = permutation[:l]
        if last_partition_sample_size > 0:
            last_partition = permutation[-last_partition_sample_size:]
            middle_indices = permutation[l:-last_partition_sample_size]
        else:
            last_partition = np.array([], dtype=int)
            middle_indices = permutation[l:]
        partitions = []
        partitions.append(first_partition)
        if p > 2:
            split_middle = np.array_split(middle_indices, p - 2)
            partitions.extend(split_middle[1:])
            partitions.append(split_middle[0])
        partitions.append(last_partition)
        return partitions


def main_divide_conquer_isomap(x_filtered, x_sample_1, r, original_isomap_sample_1, n_neighbors):
    """
    Compute the isomap configuration for a partition.
    """
    x_join_sample_1 = np.vstack((x_sample_1, x_filtered))
    isomap_all = isomap(x_join_sample_1, r, n_neighbors=n_neighbors)
    isomap_points = isomap_all['points']
    isomap_eigen = isomap_all['eigen']
    n_sample = x_sample_1.shape[0]
    isomap_sample_1 = isomap_points[:n_sample, :]
    isomap_partition = isomap_points[n_sample:, :]
    isomap_procrustes = perform_procrustes(
        isomap_sample_1, original_isomap_sample_1, isomap_partition, translation=False)
    isomap_eigen = isomap_eigen / x_filtered.shape[0]
    return {'points': isomap_procrustes, 'eigen': isomap_eigen}


def divide_conquer_isomap(x, l, c_points, r, n_neighbors):
    """
    Divide-and-conquer Isomap.

    Parameters:
        x : np.ndarray
            Data matrix with n points (rows) and k variables (columns).
        l : int
            Maximum size for Isomap on a subset.
        c_points : int
            Number of common points used for alignment.
        r : int
            Number of dimensions of the final configuration.

    Returns:
        A dictionary with keys:
          'points': Isomap configuration.
          'eigen' : Averaged eigenvalues.
    """
    n_row_x = x.shape[0]
    if n_row_x <= l:
        result = isomap(x, r)
        result['eigen'] = result['eigen'] / n_row_x
    else:
        idx_list = get_partitions_for_divide_conquer(n_row_x, l, c_points, r)
        num_partitions = len(idx_list)
        length_1 = len(idx_list[0])

        # Perform Isomap on the first partition.
        x_1 = x[idx_list[0],]
        isomap_1 = isomap(x_1, r, n_neighbors=n_neighbors)
        isomap_1_points = isomap_1['points']
        isomap_1_eigen = isomap_1['eigen'] / length_1

        # Sample c_points points from the first partition.
        sample_1_idx = np.random.choice(length_1, size=c_points, replace=False)
        x_sample_1 = x_1[sample_1_idx, :]
        isomap_sample_1 = isomap_1_points[sample_1_idx, :]

        # Process remaining partitions.
        results = [main_divide_conquer_isomap(x[idx, :], x_sample_1, r, isomap_sample_1, n_neighbors) for idx in idx_list[1:]]

        isomap_other_points = [res['points'] for res in results]
        isomap_matrix = np.vstack([isomap_1_points] + isomap_other_points)

        # Reorder rows to original order.
        order_idx = np.concatenate(idx_list)
        order = np.argsort(order_idx)
        isomap_matrix = isomap_matrix[order, :]

        # Center and rotate the configuration so that coordinates have maximum variance.
        isomap_matrix = isomap_matrix - np.mean(isomap_matrix, axis=0)
        cov_isomap_matrix = np.cov(isomap_matrix, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov_isomap_matrix)
        idx_sort = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx_sort]
        isomap_matrix = isomap_matrix @ eigenvecs

        # Average eigenvalues.
        eigen_sum = np.zeros(r)
        for res in results:
            eigen_sum += res['eigen']
        eigen_sum += isomap_1_eigen
        eigen_final = eigen_sum / num_partitions

        result = {'points': isomap_matrix, 'eigen': eigen_final}

    return result


# Example usage
if __name__ == "__main__":

    import time
    import os
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_swiss_roll

    # Set random seed for reproducibility
    np.random.seed(42)

    X, color = make_swiss_roll(n_samples=30000, noise=0.1, random_state=42)

    print("First few points of high-dimensional data:")
    print(X[:6])

    # Apply divide and conquer isomap and compare to sklearn.manifold.Isomap
    start_time = time.time()
    d_and_c_result = divide_conquer_isomap(
        x=X, l=10000, c_points=4, r=2, n_neighbors=10)
    d_and_c_runtime = time.time() - start_time

    start_time = time.time()
    normal_results = isomap(x=X, r=2, n_neighbors=10)
    normal_runtime = time.time() - start_time


    print(f"D&C runtime: {d_and_c_runtime:.2f} seconds.")
    print(f"sklearn.manifold.Isomap runtime: {normal_runtime:.2f} seconds.")

    # Print results
    print("First few points:")
    print(d_and_c_result['points'][:6])
    print("\nEigenvalues:")
    print(d_and_c_result['eigen'])

    # Original data
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax1.set_title("Original Swiss Roll")

    # D&C Isomap embedding
    ax2 = fig.add_subplot(132)
    ax2.scatter(d_and_c_result['points'][:, 0], d_and_c_result['points']
                [:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_title("D&C Isomap Embedding")

    # sklearn.manifold.Isomap embedding
    ax3 = fig.add_subplot(133)
    ax3.scatter(normal_results['points'][:, 0], normal_results['points']
                [:, 1], c=color, cmap=plt.cm.Spectral)
    ax3.set_title("sklearn.manifold.Isomap Embedding")

    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'isomap_comparison.png'))
    print("Visualization saved as 'isomap_d_and_c.png'")
