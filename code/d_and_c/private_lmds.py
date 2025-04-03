import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
import numba
import cvxpy as cp
cp.settings.DCP_CHECKS = False  # Disable DCP validation (use with caution)


# Helper functions for lmds_R_optimized
@numba.njit(parallel=True)
def _calculate_gradient(X0, M, E, n, d):
    """Calculate gradient with Numba optimization."""
    Grad = np.empty_like(X0)
    M_dot_E = M @ E
    M_dot_X0 = M @ X0

    for i in numba.prange(n):
        for j in range(d):
            Grad[i, j] = X0[i, j] * M_dot_E[i, j] - M_dot_X0[i, j]

    return Grad


@numba.njit
def _calculate_stress(Dnu, D1mulam, D1mu, Dnulam, t_val, one_minus_Inb1):
    """Calculate stress with Numba optimization."""
    s1 = (np.sum(Dnu * (D1mulam - 1)) / 2 -
          np.sum((D1mu - 1) * Dnulam) -
          t_val * np.sum((D1mu - 1) * one_minus_Inb1))
    return s1


def lmds_R_optimized(delta: np.ndarray,
                     d: int = 2,
                     k: int = 10,
                     tau: float = 1.,
                     itmax: int = 5000,
                     init: np.ndarray | None = None,
                     verbose: int = 0) -> np.ndarray:
    """
    Optimized version of Local MDS implementation.
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
    Do = delta
    if d > (n - 1):
        raise ValueError("Maximum number of dimensions is n-1!")

    # Compute symmetrized kNN graph
    Daux = np.partition(Do, k, axis=0)[k, :]
    Inb = (Do <= Daux.reshape(1, n)).astype(np.float64)
    Inb1 = np.maximum(Inb, Inb.T)
    np.fill_diagonal(Inb1, 0)

    Dnu = Inb1
    Dnulam = Do * Inb1

    cc = ((np.sum(Inb1)) / (n * n)) * np.median(Dnulam[Inb1 > 0])
    t_val = tau * cc

    # If no initial configuration is provided, compute one via Classical MDS.
    if init is None:
        # Optimize CMDS calculation
        cmds_C = np.eye(n) - np.ones((n, n)) / n
        cmds_B = -0.5 * cmds_C @ (Do**2) @ cmds_C
        vals, vecs = np.linalg.eigh(cmds_B)
        idx = np.argsort(vals)[::-1][:d]
        vals_d = vals[idx]
        vecs_d = vecs[:, idx]
        X1 = vecs_d @ np.diag(vals_d)

        # Add small random perturbation
        X1 = X1 + (np.linalg.norm(Do, 'fro') / (n * n)) * \
            0.01 * np.random.randn(n, d)
    else:
        X1 = init.copy()

    # Compute initial configuration distances and rescale
    D1 = squareform(pdist(X1))
    X1 = X1 * (np.linalg.norm(Do, 'fro') / np.linalg.norm(D1, 'fro'))

    # Pre-allocate matrices for optimization loop
    D1_temp = np.empty_like(D1)
    D1mu2 = np.empty_like(D1)
    D1mulam2 = np.empty_like(D1)
    D1mulam = np.empty_like(D1)
    D1mu = np.empty_like(D1)
    M = np.empty_like(D1)
    X0 = np.empty_like(X1)
    E = np.ones((n, d))

    # Constants for the loop
    one_minus_Inb1 = 1 - Inb1
    t_val_times_one_minus_Inb1 = t_val * one_minus_Inb1

    s1 = np.inf
    s0 = 2
    stepsize = 0.1
    i = 0

    # Main optimization loop
    while stepsize > 1e-5 and i < itmax:
        if s1 >= s0 and i > 1:
            # Reduce step size if previous step didn't improve
            stepsize *= 0.5
            X1 = X0 - stepsize * normgrad
        else:
            # Increase step size and calculate new gradient
            stepsize *= 1.05
            np.copyto(X0, X1)

            np.copyto(D1_temp, D1)
            np.fill_diagonal(D1_temp, 1)
            D1mu2 = 1.0 / D1_temp
            np.fill_diagonal(D1mu2, 0)

            D1mulam2.fill(1)
            np.fill_diagonal(D1mulam2, 0)

            M = Dnu * D1mulam2
            M = M - D1mu2 * (Dnulam + t_val_times_one_minus_Inb1)

            # Catch Exception when debugging (debugger can't access Numba compiled code)
            try:
                Grad = _calculate_gradient(X0, M, E, n, d)
            except Exception:
                M_dot_E = M @ E
                M_dot_X0 = M @ X0
                Grad = X0 * M_dot_E - M_dot_X0

            # Normalize gradient
            norm_X0 = np.linalg.norm(X0, 'fro')
            norm_Grad = np.linalg.norm(Grad, 'fro')
            normgrad = (norm_X0 / norm_Grad) * Grad
            X1 = X0 - stepsize * normgrad

        i += 1
        s0 = s1

        # Update distances
        D1 = squareform(pdist(X1))

        # Calculate stress
        D1mulam = D1 * D1
        np.fill_diagonal(D1mulam, 0)
        D1mu = D1.copy()
        np.fill_diagonal(D1mu, 0)

        # Catch Exception when debugging (debugger can't access Numba compiled code)
        try:
            s1 = _calculate_stress(
                Dnu, D1mulam, D1mu, Dnulam, t_val, one_minus_Inb1)
        except Exception:
            s1 = (np.sum(Dnu * (D1mulam - 1)) / 2 -
                  np.sum((D1mu - 1) * Dnulam) -
                  t_val * np.sum((D1mu - 1) * one_minus_Inb1))

        if verbose > 1 and ((i+1) % 100 == 0):
            print(f"niter={i+1} stress={round(s1, 5)}")

    if verbose > 0:
        print(f"Converged after {i} iterations with stress={round(s1, 5)}")

    return X1


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
            D1_temp = D1.copy()
            np.fill_diagonal(D1_temp, 1)
            D1mu2 = D1_temp**(-1)
            np.fill_diagonal(D1mu2, 0)  # Reset diagonal to zeros
            D1mulam2 = np.ones_like(D1)
            np.fill_diagonal(D1mulam2, 0)
            M = Dnu * D1mulam2 - D1mu2 * (Dnulam + t_val * (1 - Inb1))
            E = np.ones((n, d))
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

def lmds_cvxpy(D: np.ndarray,
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
