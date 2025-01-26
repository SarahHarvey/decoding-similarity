import numpy as np
import numpy.typing as npt


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