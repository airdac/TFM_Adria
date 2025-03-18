from enum import Enum
import numpy as np
from sklearn.manifold import Isomap
from scipy.spatial.distance import pdist, squareform
from typing import Callable, Optional

from .private_lmds import lmds


class DRMethod(Enum):
    """Supported Dimensionality Reduction Methods."""
    Isomap = "Isomap"
    LocalMDS = "Local MDS"

    def __str__(self):
        return self.value


def get_method_function(method: DRMethod) -> Callable:
    """Returns the function for the specified DR method.

    Raises:
        ValueError: If the specified method is not supported."""
    method_map = {
        DRMethod.Isomap: isomap,
        DRMethod.LocalMDS: local_mds
    }

    if method not in method_map:
        raise ValueError(
            f"Unsupported method: {method}. Available methods: {list(method_map.keys())}")

    return method_map[method]


def isomap(x: np.ndarray, r: int = 2, **kwargs) -> np.ndarray:
    """
    Perform Isomap on data matrix x.

    Parameters:
        x (np.ndarray): Input data matrix.
        r (int, optional): Target dimensionality (default 2).
        kwargs (Any): Additional parameters for Isomap (e.g. n_neighbors).

    Returns:
        projection (np.ndarray): The low-dimensional embedding of x.
    """
    n_neighbors = kwargs.get('n_neighbors', 5)
    isomap = Isomap(n_neighbors=n_neighbors, n_components=r)
    return isomap.fit_transform(x)


def local_mds(x: np.ndarray, r: int = 2, **kwargs) -> np.ndarray:
    """
        Perform Local MDS on data matrix x.

        Parameters:
            x (np.ndarray): Input data matrix.
            r (int, optional): Target dimensionality (default 2).
            kwargs (Any): Additional parameters for Local MDS (e.g. k, tau).

        Returns:
            projection (np.ndarray): The low-dimensional embedding of x.
        """
    return lmds_R(delta=squareform(pdist(x)), d=r, **kwargs)


def lmds_R(delta: np.ndarray,
          d: int = 2,
          k: int = 10,
          tau: float = 1.,
          itmax: int = 5000,
          init: Optional[np.ndarray] = None,
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


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt
    np.random.seed(42)

    # Generate data
    print("Generating data...")
    X, color = make_swiss_roll(n_samples=10000, random_state=42)

    # csv_filename_X = "swiss_roll_data.csv"
    # np.savetxt(csv_filename_X, X, delimiter=",",
    #         header="Feature 1,Feature 2,Feature 3", comments="")
    # print(f"Input data saved to {csv_filename_X}")

    result = lmds_R(x=squareform(pdist(X)), r=2, k=10, tau=1, verbose=2)

    # Plot the results.
    plt.figure(figsize=(8, 6))
    plt.scatter(result[:, 0], result[:, 1])
    #            , c=color, cmap=plt.cm.Spectral, edgecolor='k', s=50)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Local MDS Configuration. stress = {result['stress']}, niter = {result['niter']}")
    plt.colorbar(label="Color")
    plt.show()
