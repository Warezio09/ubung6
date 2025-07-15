# Automated check that you use the correct Python version #
from sys import version_info
if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################

import numpy as np


def power_method(A: np.ndarray[complex], v0: np.ndarray[complex], tol: float=1e-8, max_iter: int=1000) -> tuple:
    """
        This method shall implement the power method,
        that calculates the eigenvalue with the greatest absolute value
    Input:
        A: np.ndarray  -> Matrix of which the greatest (in absolute value) eigenvalue shall be calculated
        v0: np.ndarray -> Start vector for the power method
        tol: float     -> Tolerance for the stop criterion (optional)
        max_iter : int -> Maximal number of iterations
    Output:
        tuple -> Array of the iterates lambda^(i) for the eigenvalue and the approximated eigenvector v
    """
    v = v0.astype(complex) # Guarantee that the initial vector is complex
    lambdas = []
    lambdas.append((np.vdot(v,A.dot(v)))/np.vdot(v,v))
    v = v / np.linalg.norm(v)  # Anfangsvektor normieren
    for i in range(1,max_iter+1):
        produkt = A.dot(v)
        v = produkt / np.linalg.norm(produkt) #Iterationsvorschrift von Z mta3 zebi
        lambdas.append((np.vdot(v,A.dot(v)))/np.vdot(v,v))
        if abs(lambdas[i] - lambdas[i - 1]) < tol:
            break

    return np.array(lambdas, dtype=complex), v
