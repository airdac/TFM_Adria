# ChatGPT translation of the LMDS implementation in the stops R library

import numpy as np
from scipy.spatial.distance import pdist, squareform

import os
import shutil
import matplotlib.pyplot as plt
def plot_3D_to_2D(color, x, projection, method, path=None, new_directory=None):
    """
    Plot a 3D dataset and its 2D projection.

    Parameters:
        color : np.ndarray
            Colors of data points in x.
        x: np.ndarray
            Data matrix.
        projection: np.ndarray
            Data matrix of the projection.
        path: str
            Plot's path. If empty, return plot instead of saving it.
        directory: str
            Directory where plot is stored.
    """
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(234, projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2],
                c=color, cmap=plt.cm.Spectral)
    ax1.set_title("Original Data")

    ax2 = fig.add_subplot(231)
    ax2.scatter(x[:, 0], x[:, 1],
                c=color, cmap=plt.cm.Spectral)
    ax2.set_title("Original Data (dims 1,2)")

    ax3 = fig.add_subplot(232)
    ax3.scatter(x[:, 0], x[:, 2],
                c=color, cmap=plt.cm.Spectral)
    ax3.set_title("Original Data (dims 1,3)")

    ax4 = fig.add_subplot(233)
    ax4.scatter(x[:, 1], x[:, 2],
                c=color, cmap=plt.cm.Spectral)
    ax4.set_title("Original Data (dims 2,3)")

    ax5 = fig.add_subplot(235)
    ax5.scatter(projection[:, 0], projection
                [:, 1], c=color, cmap=plt.cm.Spectral)
    ax5.set_title(f"D&C {method} Subset Embedding")

    plt.tight_layout()

    if new_directory:
        # Empty directory
        if os.path.exists(new_directory):
            shutil.rmtree(new_directory)
        os.makedirs(new_directory)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        return fig


# Helper: Frobenius norm
def enorm(X):
    return np.linalg.norm(X, 'fro')

# (Placeholder) Helper: Classical MDS via eigen-decomposition.
def cmds(delta):
    # Assume delta is a distance matrix.
    # Compute double-centered matrix B = -0.5 * C * (delta**2) * C, where C = I - 1/n * ones.
    n = delta.shape[0]
    C = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * C @ (delta**2) @ C
    # Compute eigen-decomposition
    vals, vecs = np.linalg.eigh(B)
    # Sort in descending order
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return {"val": vals, "vec": vecs}

# (Placeholder) Helper: spp; assume it returns a dict with keys "resmat" and "spp".
def spp(delta, confdist, weightmat):
    # For example purposes, return dummy arrays with correct shape.
    n = delta.shape[0]
    resmat = np.zeros((n, n))
    spp_val = 0
    return {"resmat": resmat, "spp": spp_val}


def lmds(delta, k=2, tau=1, type="ratio", ndim=2, weightmat=None, itmax=5000,
         init=None, verbose=0, principal=False, normconf=False):
    """
    A Python version of the local multidimensional scaling function.
    
    Parameters
    ----------
    delta : array-like
        A symmetric distance matrix.
    k : int, optional
        The kth neighbor (default 2).
    tau : float, optional
        A scaling parameter (default 1).
    type : str, optional
        Scaling type (default "ratio").
    ndim : int, optional
        Target dimensionality (default 2).
    weightmat : array-like, optional
        A weight matrix; default is 1 - identity matrix.
    itmax : int, optional
        Maximum iterations (default 5000).
    init : array-like, optional
        Initial configuration; if None, it is computed via cmds(delta).
    verbose : int, optional
        Verbosity level (default 0).
    principal : bool, optional
        If True, project to principal components (default False).
    normconf : bool, optional
        If True, normalize final configuration (default False).
    
    Returns
    -------
    result : dict
        A dictionary containing fields:
          - 'delta', 'dhat', 'confdist', 'conf'
          - 'stress', 'stress.m', 'stress.r'
          - 'spp', 'ndim', 'weightmat', 'resmat', 'rss'
          - 'init', 'model', 'niter', 'nobj', 'type'
          - 'parameters', 'pars', 'theta', 'k', 'tau'
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
    # Create labels if not provided.
    labos = np.array([str(i) for i in range(1, n+1)])
    if ndim > (n - 1):
        raise ValueError("Maximum number of dimensions is n-1!")
    # Set weightmat to default if needed.
    if weightmat is None:
        weightmat = 1 - np.eye(n)
    X1 = init  # Initial configuration.
    lambda_ = 1.0
    mu = 1.0
    nu = 0.0
    niter = itmax
    d = ndim
    # For each column, sort and take the (k+1)th smallest value.
    # In R, indexing is 1-based so we take element at index k (0-based indexing).
    Daux = np.sort(Do, axis=0)[k, :]  # shape: (n,)
    # Broadcast Daux for comparison along rows.
    Daux_row = Daux.reshape(1, n)
    Inb = np.where(Do > Daux_row, 0, 1)
    Inb1 = np.maximum(Inb, Inb.T)
    # Compute Dnu and Dnulam.
    Dnu = np.where(Inb1 == 1, Do**nu, 0)
    Dnulam = np.where(Inb1 == 1, Do**(nu + 1/lambda_), 0)
    np.fill_diagonal(Dnu, 0)
    np.fill_diagonal(Dnulam, 0)
    # Compute cc.
    cc = ((np.sum(Inb1) - n) / (n * n)) * np.median(Dnulam[Dnulam != 0])
    t_val = tau * cc
    Grad = np.zeros((n, d))
    # If no initial configuration is provided, compute one via cmds.
    if X1 is None:
        cmd = cmds(Do)
        # Use the first d columns.
        X1 = cmd["vec"][:, :d] @ np.diag(cmd["val"][:d])
        # Add a small random perturbation scaled by norm(Do)/(n*n)
        X1 = X1 + (enorm(Do) / (n * n)) * 0.01 * np.random.randn(n, d)
    xstart = X1.copy()
    # Compute initial configuration distances.
    D1 = squareform(pdist(X1))
    # Rescale X1.
    X1 = X1 * (enorm(Do) / enorm(D1))
    s1 = np.inf
    s0 = 2
    stepsize = 0.1
    i = 0
    # Main optimization loop.
    while stepsize > 1e-5 and i < niter:
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
            D1mu2 = D1_temp**(mu - 2)
            np.fill_diagonal(D1mu2, 0)  # Reset diagonal to zeros
            D1mulam2 = D1**(mu + 1/lambda_ - 2)
            np.fill_diagonal(D1mulam2, 0)
            # M = Dnu * D1mulam2 - D1mu2 * (Dnulam + t * (not Inb1))
            # In R, !Inb1 gives 1 where Inb1==0; in Python, (1 - Inb1) does that.
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
        D1mulam = D1**(mu + 1/lambda_)
        np.fill_diagonal(D1mulam, 0)
        D1mu = D1**mu
        np.fill_diagonal(D1mu, 0)
        s1 = (np.sum(Dnu * (D1mulam - 1)) / (mu + 1/lambda_) -
              np.sum((D1mu - 1) * Dnulam) / mu -
              t_val * np.sum((D1mu - 1) * (1 - Inb1)) / mu)
        if verbose > 1 and ((i+1) % 100 == 0):
            print(f"niter={i+1} stress={round(s1, 5)}")
    # End of loop.
    # Rescale configuration.
    X1a = X1 * (np.sum(Do * D1) / np.sum(D1**2))
    D1a = squareform(pdist(X1a))
    D0 = np.zeros_like(D1a)
    D1mulama = D1a**(mu + 1/lambda_)
    np.fill_diagonal(D1mulama, 0)
    Domulam = Do**(mu + 1/lambda_)
    D0mulam = D0**(mu + 1/lambda_)  # remains zeros
    # Set diagonals to zero.
    np.fill_diagonal(Domulam, 0)
    np.fill_diagonal(D0mulam, 0)
    D1mua = D1a**mu
    Domu = Do**mu
    D0mu = D0**mu
    np.fill_diagonal(D1mua, 0)
    np.fill_diagonal(Domu, 0)
    np.fill_diagonal(D0mu, 0)
    s1e = (np.sum(Dnu * D1mulama) / (mu + 1/lambda_) -
           np.sum(D1mua * Dnulam) / mu -
           t_val * np.sum(D1mua * (1 - Inb1)) / mu)
    normop = (np.sum(Dnu * Domulam) / (mu + 1/lambda_) -
              np.sum(Domu * Dnulam) / mu -
              t_val * np.sum(Domu * (1 - Inb1)) / mu)
    s1n = 1 - s1e / normop
    # Build result dictionary.
    result = {}
    result["delta"] = Do
    result["dhat"] = Do
    if normconf:
        X1 = X1 / enorm(X1)
    if principal:
        # Project to principal components via SVD.
        U, s_vals, Vt = np.linalg.svd(X1, full_matrices=False)
        X1 = X1 @ Vt.T
    result["iord"] = np.argsort(Do.flatten())
    result["confdist"] = D1
    result["conf"] = X1
    result["stress"] = np.sqrt(s1n)
    # Create default weightmat as 1 - identity (here as full matrix)
    result["weightmat"] = 1 - np.eye(n)
    # Call spp to get resmat and spp (assume spp returns a dict)
    spoint = spp(result["delta"], result["confdist"], result["weightmat"])
    result["resmat"] = spoint["resmat"]
    # Compute rss from lower-triangular part.
    result["rss"] = np.sum(np.tril(spoint["resmat"], k=-1))
    result["spp"] = spoint["spp"]
    result["ndim"] = ndim
    result["niter"] = i
    result["nobj"] = n
    result["type"] = "ratio"
    result["stress.m"] = s1n
    result["stress.r"] = s1
    result["tdelta"] = Do
    result["parameters"] = {"k": k, "tau": tau}
    result["pars"] = {"k": k, "tau": tau}
    result["theta"] = {"k": k, "tau": tau}
    result["k"] = k
    result["tau"] = tau
    result["model"] = "Local MDS"
    # Optionally, store initial configuration and the call parameters.
    result["init"] = xstart
    result["call"] = {"delta": delta, "k": k, "tau": tau, "ndim": ndim, "itmax": itmax,
                      "init": init, "verbose": verbose, "principal": principal, "normconf": normconf}
    # In R the result is given specific classes; here we simply return the dict.
    return result


def stoploss(obj, stressweight=1, structures=None, strucweight=None,
             strucpars=None, stoptype="additive", verbose=0):
    """
    A Python version of the stoploss function.
    
    Parameters
    ----------
    obj : dict
        A dictionary output from lmds containing at least 'stress.m' and 'conf' and 'pars'.
    stressweight : float, optional
        Weight for the stress term (default 1).
    structures : list of str
        List of structure names.
    strucweight : list of float, optional
        Weights for each structure. If None, defaults to a list of -1/len(structures).
    strucpars : list, optional
        List of parameters (dictionaries) for each structure; if None, defaults to a list of None.
    stoptype : str, optional
        Either "additive" or "multiplicative" (default "additive").
    verbose : int, optional
        Verbosity level (default 0).
    
    Returns
    -------
    out : dict
        A dictionary with keys 'stoploss', 'strucindices', 'parameters', and 'theta'.
    """
    if structures is None:
        raise ValueError("structures must be provided")
    # Set default strucpars if missing.
    if strucpars is None:
        strucpars = [None] * len(structures)
    # Set default strucweight if missing.
    if strucweight is None:
        strucweight = [-1/len(structures)] * len(structures)
    stressi = obj["stress.m"]
    pars = obj["pars"]
    # Compute structure indices by calling each registry entry's index function.
    # NB: The registry argument has been removed in this Python code.
    struc = []
    # Compute the combined index.
    if stoptype == "additive":
        ic = stressi * stressweight + \
            sum(s * w for s, w in zip(struc, strucweight))
    elif stoptype == "multiplicative":
        # Use product; note: math.prod is available in Python 3.8+
        import math
        prod_val = math.prod(s**w for s, w in zip(struc, strucweight))
        ic = (stressi**stressweight) * prod_val
    else:
        raise ValueError(
            "stoptype must be either 'additive' or 'multiplicative'")
    if verbose > 0:
        print(
            f"stoploss = {ic}, mdsloss = {stressi}, structuredness = {struc}, parameters = {pars}")
    out = {
        "stoploss": ic,
        "strucindices": struc,
        "parameters": pars,
        "theta": pars
    }
    return out


def local_mds(
    dis,
    theta=[10, 0.5],
    type="ratio",
    weightmat=None,
    init=None,
    ndim=2,
    itmaxi=5000,
    stressweight=1,
    structures=None,
    strucweight=None,
    strucpars=None,
    verbose=0,
    stoptype="additive",
    **kwargs
):
    """
    A Python version of an R function for local MDS.
    
    Parameters
    ----------
    dis : array-like
        A distance matrix (or an object that can be converted to a matrix).
    theta : list of two numbers, optional
        Parameters [k, tau]. Default is [2, 0.5]. If a single value is provided,
        it will be replicated to length 2.
    type : str, optional
        The type of scaling; default is "ratio".
    weightmat : array-like, optional
        A weight matrix (default: None).
    init : array-like, optional
        Initial configuration (default: None).
    ndim : int, optional
        Target dimensionality (default: 2).
    itmaxi : int, optional
        Maximum number of iterations (default: 5000).
    stressweight : numeric, optional
        Weight for stress in the stopping objective (default: 1).
    structures : list of str, optional
        List of structure names; if None a default list is used.
    strucweight : list of numeric, optional
        Weights for the structures; if None each is set to 1/len(structures).
    strucpars : any, optional
        Additional structure parameters.
    verbose : int, optional
        Verbosity level (default: 0).
    stoptype : str, optional
        Either "additive" or "multiplicative"; default is "additive".
    **kwargs :
        Additional keyword arguments passed to lmds.
    
    Returns
    -------
    out : dict
        Dictionary with keys: 'stress', 'stress.m', 'stoploss', 'strucindices',
        'parameters', 'fit', and 'stopobj'.
    """
    if len(theta) > 3:
        raise ValueError(
            "There are too many parameters in the theta argument.")
    if len(theta) < 2:
        # If only one value is provided, replicate it to create two parameters
        theta = theta * 2

    k = theta[0]
    tau = theta[1]

    # Increase verbosity by 2
    verbose += 2

    # If dis is a distance object or a DataFrame, convert it to a numpy array.
    # (This assumes dis can be converted via numpy.asarray.)
    import numpy as np
    dis = np.asarray(dis)

    # Call the lmds function (which you must implement or import)
    fit = lmds(delta=dis, k=k, tau=tau, init=init, ndim=ndim,
               verbose=verbose, itmax=itmaxi, **kwargs)

    # In R the original call was substituted; in Python you can store the parameters if desired.
    fit["k"] = k
    fit["tau"] = tau
    fit["parameters"] = fit["theta"] = fit["pars"] = {
        "k": fit["k"], "tau": fit["tau"]}

    # Call stoploss to obtain additional diagnostic measures
    stopobj = stoploss(
        fit,
        stressweight=stressweight,
        structures=structures if structures is not None else [
            "cclusteredness", "clinearity", "cdependence", "cmanifoldness",
            "cassociation", "cnonmonotonicity", "cfunctionality", "ccomplexity",
            "cfaithfulness", "cregularity", "chierarchy", "cconvexity",
            "cstriatedness", "coutlying", "cskinniness", "csparsity",
            "cstringiness", "cclumpiness", "cinequality"
        ],
        strucweight=strucweight if strucweight is not None else None,
        strucpars=strucpars,
        verbose=(verbose > 1),
        stoptype=stoptype
    )

    # Build the output dictionary
    out = {
        "stress": fit.get("stress"),
        "stress.m": fit.get("stress.m"),
        "stoploss": stopobj.get("stoploss"),
        "strucindices": stopobj.get("strucindices"),
        "parameters": stopobj.get("parameters"),
        "fit": fit,
        "stopobj": stopobj,
    }

    return out


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    import json
    from numpy.core.numeric import array_repr
    import time

    # Set random seed for reproducibility
    np.random.seed(42)

    n = 1000

    X, color = make_swiss_roll(n_samples=n, random_state=42)

    start_time = time.time()
    delta = squareform(pdist(X, metric="euclidean"))
    result = local_mds(delta)
    lmds_time = time.time() - start_time

    # Plot result
    plot_3D_to_2D(color, X, result['fit']['conf'], "LMDS")
    plot_filename = f'dc_LMDS-n{n}'
    plt.savefig(os.path.join(
        'figures', plot_filename))
    plt.close()
    print(
        f"Visualization saved as '{os.path.join("figures", plot_filename)}.png'")

    # Save results to a text file
    with open('LMDS_result.txt', 'w') as f:
        # Write key metrics and parameters
        f.write("Local MDS Results\n")
        f.write("================\n\n")
        f.write(f"Stress: {result['stress']}\n")
        f.write(f"Stress (normalized): {result['stress.m']}\n")
        f.write(
            f"Parameters: k={result['parameters'].get('k')}, tau={result['parameters'].get('tau')}\n\n")
        f.write(f"Runtime: {lmds_time:.2f} seconds.\n")

        # Write configuration summary
        if 'fit' in result and 'conf' in result['fit']:
            conf = result['fit']['conf']
            f.write(f"Configuration shape: {conf.shape}\n")
            f.write(
                f"Configuration first 5 points:\n{array_repr(conf[:5], precision=4)}\n\n")

        # Write structure indices if available
        if 'strucindices' in result:
            f.write("Structure indices:\n")
            for i, val in enumerate(result['strucindices']):
                f.write(f"  Index {i}: {val}\n")

    print(f"Results saved to LMDS_result.txt")
