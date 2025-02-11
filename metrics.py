import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

class LinearCKA:
    """
    Note: This function differs from the one outlined in
    Kornblith et al. (2019). It introduces an arccos(.)
    into the final calculation so that the result satisfies
    the conditions of a metric.
    """

    def __init__(self, center_columns=True):
        self.center_columns = center_columns

    def fit(self, X, Y):
        pass

    def score(self, X, Y):
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        dist : float
            Distance between X and Y.
        """

        if self.center_columns:
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)

        # Compute angular distance between (sample x sample) covariance matrices.
        KX = X@np.transpose(X)
        KY = Y@np.transpose(Y)    
        cka_score = np.trace(KX @ KY)/(np.linalg.norm(KX,ord='fro')*np.linalg.norm(KY,ord='fro'))

        return cka_score



def sq_bures_metric(A: npt.NDArray, B: npt.NDArray) -> float:
    """Slow way to compute the square of the Bures metric between two
     positive-definite matrices.
    """
    va, ua = np.linalg.eigh(A)
    vb, ub = np.linalg.eigh(B)
    sva = np.sqrt(np.maximum(va, 0.0))
    svb = np.sqrt(np.maximum(vb, 0.0))
    return (
        np.sum(va) + np.sum(vb) - 2 * np.sum(
            np.linalg.svd(
                (sva[:, None] * ua.T) @ (ub * svb[None, :]),
                compute_uv=False
            )
        )
    )