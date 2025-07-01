import numpy as np
import numba as nb
from scipy.sparse import diags, identity,csr_matrix,coo_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve as dense_solve

# === GROUPED SUM ===
def matrix_grouped_sum_sparse(group_matrix: csr_matrix, values: np.ndarray) -> np.ndarray:
    return group_matrix @ values

def matrix_grouped_sum_dense(group_matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    return group_matrix @ values



# === GROUPED COUNT ===
def matrix_grouped_count_sparse(group_matrix: csr_matrix) -> np.ndarray:
    """
    Grouped count using sparse membership matrix.
    """
    return np.asarray(group_matrix.sum(axis=1)).flatten()


def matrix_grouped_count_dense(group_matrix: np.ndarray) -> np.ndarray:
    """
    Grouped count using dense membership matrix.
    """
    return group_matrix.sum(axis=1)


# === MASKED SUM ===
def matrix_masked_sum_sparse(mask_matrix: csr_matrix, data: np.ndarray) -> np.ndarray:
    return mask_matrix @ data

def matrix_masked_sum_dense(mask_matrix: np.ndarray, data: np.ndarray) -> np.ndarray:
    return mask_matrix @ data



# === UTILITY ===
def encode_sparse_groups(group_ids: np.ndarray, n_groups: int) -> csr_matrix:
    """
    Encode group ID vector → sparse binary group membership matrix (G × N).
    """
    N = group_ids.shape[0]
    row = group_ids
    col = np.arange(N)
    data = np.ones(N, dtype=np.float32)
    return coo_matrix((data, (row, col)), shape=(n_groups, N)).tocsr()


def encode_dense_groups(group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Encode group ID vector → dense group matrix (G × N).
    """
    N = group_ids.shape[0]
    G = np.zeros((n_groups, N), dtype=np.float32)
    G[group_ids, np.arange(N)] = 1.0
    return G


# === Smoothing ===
def smooth(X, alpha=0.02, out=None):
    smoothed = (1 - alpha) * X + alpha * X.mean(axis=-1, keepdims=True)
    if out is not None:
        np.copyto(out, smoothed)
        return out
    return smoothed
