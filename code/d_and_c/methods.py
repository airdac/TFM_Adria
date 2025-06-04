from enum import Enum
import os
import numpy as np
from sklearn.manifold import Isomap, smacof
from openTSNE import TSNE
from scipy.spatial.distance import pdist, squareform
from typing import Callable

from .private_lmds import lmds_R_optimized
from .utils import apply_principal_components


class DRMethod(Enum):
    """Supported Dimensionality Reduction Methods."""
    Isomap = "Isomap"
    LocalMDS = "Local MDS"
    tSNE = "t-SNE"
    SMACOF = "SMACOF"

    def __str__(self):
        return self.value


def get_method_function(method: DRMethod) -> Callable:
    """Returns the function for the specified DR method.

    Raises:
        ValueError: If the specified method is not supported."""
    method_map = {
        DRMethod.Isomap: isomap,
        DRMethod.LocalMDS: local_mds,
        DRMethod.tSNE: tsne,
        DRMethod.SMACOF: mds_smacof
    }

    if method not in method_map:
        raise ValueError(
            f"Unsupported method: {method}. Available methods: {list(method_map.keys())}")

    return method_map[method]


def isomap(x: np.ndarray, r: int = 2,
           principal_components: bool | None = True,
           n_jobs: int | None = -1,
           **kwargs) -> np.ndarray:
    """
    Perform Isomap on data matrix x.

    Parameters:
        x (np.ndarray): Input data matrix.
        r (int, optional): Target dimensionality (default 2).
            principal_components (bool, optional): Whether to apply principal compoenents to output embedding.
        kwargs (Any): Additional parameters for Isomap (e.g. n_neighbors).

    Returns:
        projection (np.ndarray): The low-dimensional embedding of x.
    """
    n_neighbors = kwargs.get('n_neighbors', 5)
    isomap = Isomap(n_neighbors=n_neighbors, n_components=r, n_jobs=n_jobs)
    embedding = isomap.fit_transform(x)
    if principal_components:
        return apply_principal_components(embedding)
    return embedding


def mds_smacof(x: np.ndarray, r: int = 2,
               principal_components: bool | None = True,
               n_jobs: int | None = -1,
               **kwargs) -> np.ndarray:
    """
        Compute multidimensional scaling on x using the SMACOF algorithm.

        Parameters:
            x (np.ndarray): Input data matrix.
            r (int, optional): Target dimensionality (default 2).
            principal_components (bool, optional): Whether to apply principal compoenents to output embedding.
            n_jobs (int, optional): number of jobs to run in parallel. -1 is interpreted as using all cores.
            kwargs (Any): Additional parameters for SMACOF(e.g. k, max_iter).

        Returns:
            projection (np.ndarray): The low-dimensional embedding of x.
        """
    embedding, _ = smacof(squareform(pdist(x)), n_components=r, n_jobs=n_jobs, **kwargs)
    if principal_components:
        return apply_principal_components(embedding)
    return embedding


def local_mds(x: np.ndarray, r: int = 2,
              principal_components: bool | None = True,
              **kwargs) -> np.ndarray:
    """
        Perform LMDS on data matrix x.

        Parameters:
            x (np.ndarray): Input data matrix.
            r (int, optional): Target dimensionality (default 2).
            principal_components (bool, optional): Whether to apply principal compoenents to output embedding.
            kwargs (Any): Additional parameters for Local MDS (e.g. k, tau).

        Returns:
            projection (np.ndarray): The low-dimensional embedding of x.
        """
    embedding = lmds_R_optimized(delta=squareform(pdist(x)), d=r, **kwargs)
    if principal_components:
        return apply_principal_components(embedding)
    return embedding


def tsne(x: np.ndarray, r: int = 2,
         principal_components: bool | None = True,
         n_jobs: int | None = -1,
         **kwargs) -> np.ndarray:
    """
        Perform t-SNE on data matrix x.

        Parameters:
            x (np.ndarray): Input data matrix.
            r (int, optional): Target dimensionality (default 2).
            principal_components (bool, optional): Whether to apply principal compoenents to output embedding.
            n_jobs (int, optional): number of jobs to run in parallel. -1 is interpreted as using all cores.
            kwargs (Any): Additional parameters for t-SNE.

        Returns:
            projection (np.ndarray): The low-dimensional embedding of x.
        """
    tsne = TSNE(n_components=r, n_jobs=n_jobs, **kwargs)
    embedding = np.array(tsne.fit(x))
    if principal_components:
        return apply_principal_components(embedding)
    return embedding


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
    plt.title(
        f"Local MDS Configuration. stress = {result['stress']}, niter = {result['niter']}")
    plt.colorbar(label="Color")
    plt.show()
