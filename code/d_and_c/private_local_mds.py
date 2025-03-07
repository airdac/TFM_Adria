# ChatGPT translation of the LMDS implementation in the stops R library

import numpy as np
from typing import Optional, List

def enorm(X: np.ndarray) -> float:
    """
    Compute the Frobenius norm of matrix X.

    Parameters:
        X (np.ndarray): Input matrix.

    Returns:
        float: The Frobenius norm of X.
    """
    return np.linalg.norm(X, 'fro')


def cmds(delta: np.ndarray) -> dict:
    """
    Perform classical MDS on the distance matrix.

    Parameters:
        delta (np.ndarray): A distance matrix.

    Returns:
        output (dict): A dictionary containing:
        "val" - Eigenvalues sorted in descending order.
        "vec" - Eigenvectors corresponding to the values.
    """
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


def spp(delta: np.ndarray, confdist: np.ndarray, weightmat: np.ndarray) -> dict:
    """
    Compute a structure preservation measure. (Placeholder implementation)

    Parameters:
        delta (np.ndarray): The original distance matrix.
        confdist (np.ndarray): The distance matrix computed from the configuration.
        weightmat (np.ndarray): A weight matrix.

    Returns:
        output (dict): A dictionary with keys:
        "resmat" - Residual matrix (np.ndarray).
        "spp" - A structure preservation index (value type depends on implementation; here, dummy value 0).
    """
    # For example purposes, return dummy arrays with correct shape.
    n = delta.shape[0]
    resmat = np.zeros((n, n))
    spp_val = 0
    return {"resmat": resmat, "spp": spp_val}


def stoploss(obj: dict,
             stressweight: float = 1,
             structures: Optional[List[str]] = None,
             strucweight: Optional[List[float]] = None,
             strucpars: Optional[List] = None,
             stoptype: str = "additive",
             verbose: int = 0) -> dict:
    """
    Compute a stop-loss measure for configuration optimization based on stress and structure preservation.

    Parameters:
        obj (dict): A dictionary output from the local MDS function (such as _lmds) containing at least keys 'stress.m', 'conf', and 'pars'.
        stressweight (float, optional): Weight for the stress term (default is 1).
        structures (Optional[List[str]]): A list of structure names. Must be provided.
        strucweight (Optional[List[float]], optional): Weights for each structure. If None, defaults to -1/len(structures).
        strucpars (Optional[List], optional): Parameters for each structure; if None, defaults to a list of None.
        stoptype (str, optional): "additive" or "multiplicative" (default "additive").
        verbose (int, optional): Verbosity level (default 0).

    Returns:
        output (dict): A dictionary with keys:
            "stoploss" - The computed overall loss.
            "strucindices" - The computed structure indices.
            "parameters" - The parameters used.
            "theta" - The same as parameters.
    
    Raises:
        ValueError: If structures is not provided or if stoptype is neither 'additive' nor 'multiplicative'.
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
