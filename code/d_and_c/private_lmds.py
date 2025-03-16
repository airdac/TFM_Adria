from typing import Optional
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
import cvxpy as cp
cp.settings.DCP_CHECKS = False  # Disable DCP validation (use with caution)


def lmds(D: np.ndarray,
         r: int = 2,
         k: int = 10,
         tau: float = 1.,
         max_iter: int = 5000,
         init: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Our own implementation of Local MDS (Chen & Buja (2006)).

    Parameters:
        D (np.ndarray): A symmetric distance matrix.
        r (int, optional): Target dimensionality (default 2).
        k (int, optional): The kth neighbor (default 2).
        tau (float, optional): A scaling parameter (default 1.).
        max_iter (int, optional): Maximum iterations (default 5000).
        init (np.ndarray, optional): Initial configuration; if None, computed via cmds(D).
        verbose (int, optional): Verbosity level (default 0).

    Returns:
        projection (np.ndarray): The low-dimensional embedding of x.

    Raises:
        ValueError: If the D matrix is not symmetric or if ndim > n - 1."""

    # Ensure D is a NumPy array.
    D = np.asarray(D)
    n = D.shape[0]
    # Check symmetry
    if not np.allclose(D, D.T):
        raise ValueError("Delta is not symmetric.")
    if r > (n - 1):
        raise ValueError("Maximum number of dimensions is n-1!")

    # If no initial configuration is provided, compute one via Classical MDS.
    if init is None:
        print("Computing initial configuration...")
        cmds_C = np.eye(n) - np.ones((n, n)) / n
        cmds_B = -0.5 * cmds_C @ (D**2) @ cmds_C
        # Compute eigen-decomposition
        eigen_vals, eigen_vecs = np.linalg.eigh(cmds_B)
        # Sort in descending order
        idx = np.argsort(eigen_vals)[::-1]
        # Use the first r columns.
        idx = idx[:r]
        init = eigen_vecs[:, idx] @ np.diag(eigen_vals[idx])

    print(f"Minimizing stress with tau={tau}, k={k}...")
    # Compute symmetrized k-NN graph
    N = kneighbors_graph(D, n_neighbors=k)
    N = N.maximum(N.T).toarray()
    N = np.triu(N, k=1)

    #  Compute t
    num_N = np.sum(N)
    num_Nc = n*(n-1) - num_N
    median_N_D = np.median(D[N == 1])
    t_val = (num_N / num_Nc) * median_N_D * tau

    # Define the optimization variable and set its initial value
    X = cp.Variable((n, r), value=init)

    # Vectorized computation of the pairwise distances
    # Compute squared norms of rows (n x 1)
    X_norm_sq = X_norm_sq = cp.sum(cp.square(X), axis=1)
    # Build pairwise squared distance matrix using broadcasts and matrix multiplication.
    X_diff_sq = cp.reshape(X_norm_sq, (n, 1)) + \
        cp.reshape(X_norm_sq, (1, n)) - 2 * cp.matmul(X, X.T)
    # Compute pairwise distances (ensure nonnegative values with cp.pos)
    X_diff = cp.sqrt(cp.pos(X_diff_sq))

    # Create an upper-triangular mask
    mask = np.triu(np.ones((n, n)), k=1)

    # Build the vectorized stress objective
    local_stress = cp.sum(cp.multiply((D - X_diff)**2, mask * N))
    repulsion = cp.sum(cp.multiply(X_diff, mask * (1 - N)))
    stress = local_stress - t_val * repulsion

    # # Build the objective from scalar terms over unique pairs (i<j)
    # terms = []
    # for i in range(n):
    #     for j in range(i+1, n):
    #         # Compute the Euclidean distance between x_i and x_j as a scalar CVXPY expression.
    #         dist_ij = cp.norm(X[i, :] - X[j, :])
    #         local_stress = cp.multiply((D[i, j] - dist_ij)**2, N[i, j])
    #         repulsion = cp.multiply(dist_ij, 1 - N[i, j])
    #         terms.append(local_stress - t_val * repulsion)

    # # Sum all terms to form the stress objective.
    # stress = cp.sum(terms)

    # Optimization problem
    prob = cp.Problem(cp.Minimize(stress))
    prob.solve(solver=cp.SCS, max_iter=max_iter, nonconvex=True, validate=False, verbose=True)

    # Get and return the solution
    print(f"Stress minimization status: {prob.status}")
    print(f"Optimal stress: {prob.value}")
    X_solution = X.value
    return X_solution


# Example usage:
if __name__ == "__main__":

    n, r = 100, 2
    np.random.seed(42)
    X = np.random.rand(n, r)
    D = squareform(pdist(X))

    result = lmds(D, r)
