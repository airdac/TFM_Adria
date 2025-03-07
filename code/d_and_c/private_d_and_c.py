import numpy as np


def _get_procrustes_parameters(x, target, translation=False):
    """Compute Procrustes rotation to align x to target."""
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
    """Align matrix_to_transform to target using Procrustes transformation."""
    if x.shape != target.shape:
        raise ValueError("x and target do not have the same shape")
    if x.shape[1] != matrix_to_transform.shape[1]:
        raise ValueError(
            "x and matrix_to_transform do not have the same number of columns")

    rotation_matrix, translation_vector = _get_procrustes_parameters(
        x, target, translation)
    return matrix_to_transform @ rotation_matrix + translation_vector


def get_partitions_for_divide_conquer(n, l, c_points, r):
    """Divide indices 0,...,n-1 into partitions."""
    # Input validation
    if l <= c_points:
        raise ValueError("l must be greater than c_points")
    if l - c_points < c_points:
        raise ValueError("l-c_points must be at least c_points")
    if l - c_points < r+2:
        raise ValueError("l-c_points must be at least r+2")

    # Generate random permutation
    permutation = np.random.permutation(n)

    # Handle case when data fits in a single partition
    if n <= l:
        return [permutation]

    # Divide into multiple partitions
    p = np.ceil((n - l) / (l - c_points))
    first_partition = permutation[:l]
    other_partitions = np.array_split(permutation[l:], p)

    return [first_partition] + other_partitions
