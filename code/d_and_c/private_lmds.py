from typing import Optional
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
import cvxpy as cp
cp.settings.DCP_CHECKS = False  # Disable DCP validation (use with caution)


def lmds_R(delta: np.ndarray,
           d: int = 2,
           k: int = 10,
           tau: float = 1.,
           itmax: int = 5000,
           init: np.ndarray | None = None,
           verbose: int = 0) -> dict:
    """
    A Python version of the Local MDS implementation in the *smacofx* R library (used in the *stops* R library). This function minimizes the Local MDS Stress of Chen & Buja (2006) via gradient descent. This is a ratio metric scaling method.

    Parameters:
        delta (np.ndarray): A symmetric distance matrix.
        k (int, optional): The kth neighbor (default 2).
        tau (float, optional): A scaling parameter (default 1).
        ndim (int, optional): Target dimensionality (default 2).
        itmax (int, optional): Maximum iterations (default 5000).
        init (np.ndarray, optional): Initial configuration; if None, computed via cmds(delta).
        verbose (int, optional): Verbosity level (default 0).

    Returns:
        output (dict): A dictionary containing fields: 'conf', 'confdist', 'stress', 'stress.m', 'stress.r', 'niter'.

    Raises:
        ValueError: If the delta matrix is not symmetric or if ndim > n - 1.
    """
    # Ensure delta is a NumPy array.
    delta = np.asarray(delta)
    n = delta.shape[0]
    # Check symmetry
    if not np.allclose(delta, delta.T):
        raise ValueError("Delta is not symmetric.")
    if verbose > 0:
        print(f"Minimizing lmds with tau={tau}, k={k}")
    # Use delta as Do.
    Do = delta.copy()
    if d > (n - 1):
        raise ValueError("Maximum number of dimensions is n-1!")
    X1 = init  # Initial configuration.

    # Compute symmetrized kNN grah
    # For each column, sort and take the (k+1)th smallest value.
    Daux = np.sort(Do, axis=0)[k, :]  # shape: (n,)
    # Broadcast Daux for comparison along rows.
    Daux_row = Daux.reshape(1, n)
    Inb = np.where(Do > Daux_row, 0, 1)
    Inb1 = np.maximum(Inb, Inb.T)

    # Compute Dnu and Dnulam.
    Dnu = Inb1
    Dnulam = np.where(Inb1 == 1, Do, 0)
    np.fill_diagonal(Dnu, 0)
    np.fill_diagonal(Dnulam, 0)
    # Compute cc.
    cc = ((np.sum(Inb1) - n) / (n * n)) * np.median(Dnulam[Dnulam != 0])
    t_val = tau * cc
    Grad = np.zeros((n, d))
    # If no initial configuration is provided, compute one via Classical MDS.
    if X1 is None:
        cmds_C = np.eye(n) - np.ones((n, n)) / n
        cmds_B = -0.5 * cmds_C @ (Do**2) @ cmds_C
        # Compute eigen-decomposition
        vals, vecs = np.linalg.eigh(cmds_B)
        # Sort in descending order
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        # Use the first d columns.
        X1 = vecs[:, :d] @ np.diag(vals[:d])
        # Add a small random perturbation scaled by norm(Do)/(n*n)
        X1 = X1 + (np.linalg.norm(Do, 'fro') / (n * n)) * \
            0.01 * np.random.randn(n, d)
    # Compute initial configuration distances.
    D1 = squareform(pdist(X1))
    # Rescale X1.
    X1 = X1 * (np.linalg.norm(Do, 'fro') / np.linalg.norm(D1, 'fro'))
    s1 = np.inf
    s0 = 2
    stepsize = 0.1
    i = 0
    # Main optimization loop.
    while stepsize > 1e-5 and i < itmax:
        if s1 >= s0 and i > 1:
            stepsize = 0.5 * stepsize
            X1 = X0 - stepsize * normgrad
        else:
            stepsize = 1.05 * stepsize
            X0 = X1.copy()
            # Compute D1 raised to (mu-2) and (mu+1/lambda - 2)
            D1_temp = D1.copy()
            # Any non-zero value will work since we'll zero it out after
            np.fill_diagonal(D1_temp, 1)
            D1mu2 = D1_temp**(-1)
            np.fill_diagonal(D1mu2, 0)  # Reset diagonal to zeros
            D1mulam2 = np.ones_like(D1)
            np.fill_diagonal(D1mulam2, 0)
            M = Dnu * D1mulam2 - D1mu2 * (Dnulam + t_val * (1 - Inb1))
            E = np.ones((n, d))
            # Elementwise multiply X0 with (M @ E) then subtract M @ X0.
            Grad = X0 * (M @ E) - (M @ X0)
            norm_X0 = np.linalg.norm(X0, 'fro')
            norm_Grad = np.linalg.norm(Grad, 'fro')
            normgrad = (norm_X0 / norm_Grad) * Grad
            X1 = X0 - stepsize * normgrad
        i += 1
        s0 = s1
        D1 = squareform(pdist(X1))
        D1mulam = D1**2
        np.fill_diagonal(D1mulam, 0)
        D1mu = D1
        np.fill_diagonal(D1mu, 0)
        s1 = (np.sum(Dnu * (D1mulam - 1)) / (2) -
              np.sum((D1mu - 1) * Dnulam) / 1 -
              t_val * np.sum((D1mu - 1) * (1 - Inb1)))
        if verbose > 1 and ((i+1) % 100 == 0):
            print(f"niter={i+1} stress={round(s1, 5)}")

    return X1
    # End of loop.
    # Rescale configuration.
    # X1a = X1 * (np.sum(Do * D1) / np.sum(D1**2))
    # D1a = squareform(pdist(X1a))
    # D0 = np.zeros_like(D1a)
    # D1mulama = D1a**2
    # np.fill_diagonal(D1mulama, 0)
    # Domulam = Do**2
    # D0mulam = D0**2  # remains zeros
    # # Set diagonals to zero.
    # np.fill_diagonal(Domulam, 0)
    # np.fill_diagonal(D0mulam, 0)
    # D1mua = D1a
    # Domu = Do
    # D0mu = D0
    # np.fill_diagonal(D1mua, 0)
    # np.fill_diagonal(Domu, 0)
    # np.fill_diagonal(D0mu, 0)
    # s1e = (np.sum(Dnu * D1mulama) / 2 -
    #        np.sum(D1mua * Dnulam) / 1 -
    #        t_val * np.sum(D1mua * (1 - Inb1)))
    # normop = (np.sum(Dnu * Domulam) / 2 -
    #           np.sum(Domu * Dnulam) -
    #           t_val * np.sum(Domu * (1 - Inb1)))
    # s1n = 1 - s1e / normop
    # Build result dictionary.
    # result = {}
    # result["conf"] = X1
    # result["confdist"] = D1
    # result["stress"] = np.sqrt(s1n)
    # result["niter"] = i
    # result["stress.m"] = s1n
    # result["stress.r"] = s1
    # return result


def lmds(D: np.ndarray,
         r: int = 2,
         k: int = 10,
         tau: float = 1.,
         max_iter: int = 5000,
         init: np.ndarray | None = None) -> np.ndarray:
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
    prob.solve(solver=cp.SCS, max_iter=max_iter,
               nonconvex=True, validate=False, verbose=True)

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
