import numpy as np

__all__ = ["euclidean_rows", "HAS_NUMBA"]

# ------------------------------------------------------------
# Fast Euclidean distance between corresponding rows of two
# (N,3) float32 arrays.  Falls back to NumPy if Numba missing.
# ------------------------------------------------------------

try:
    from numba import njit, prange  # type: ignore

    @njit(fastmath=True, nogil=True, parallel=True)
    def _euclid_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pragma: no cover
        n = a.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in prange(n):
            dx = a[i, 0] - b[i, 0]
            dy = a[i, 1] - b[i, 1]
            dz = a[i, 2] - b[i, 2]
            out[i] = (dx * dx + dy * dy + dz * dz) ** 0.5
        return out

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _euclid_rows = None  # type: ignore
    HAS_NUMBA = False


def euclidean_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance of two equal-length (N,3) arrays."""
    if HAS_NUMBA:
        return _euclid_rows(a.astype(np.float32), b.astype(np.float32))
    return np.linalg.norm(a - b, axis=1).astype(np.float32) 
