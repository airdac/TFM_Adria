# ChatGPT translation of the LMDS implementation in the stops R library

import numpy as np

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
