import numpy as np
from tabularepimdl.matrixops_utils import (
    matrix_grouped_count_sparse,
    matrix_grouped_count_dense,
    matrix_masked_sum_dense,
    matrix_masked_sum_sparse,
    matrix_grouped_sum_dense,
    matrix_grouped_sum_sparse
)

# === Internal autodispatch thresholds (based on benchmarking) ===
_DISPATCH_THRESHOLD = 350

# === Grouped count autodispatch ===
def matrix_grouped_count(group_matrix):
    """
    Auto-dispatch grouped count using size threshold.
    Use dense for small matrices, sparse for large ones.
    """
    n = group_matrix.shape[0]
    if n < _DISPATCH_THRESHOLD:
        dense_matrix = group_matrix.toarray() if hasattr(group_matrix, "toarray") else group_matrix
        return matrix_grouped_count_dense(dense_matrix)
    else:
        return matrix_grouped_count_sparse(group_matrix)

# === Masked sum autodispatch ===
def matrix_masked_sum(mask_matrix, data):
    """
    Auto-dispatch masked sum using size threshold.
    Uses dense for small matrices, sparse for large ones.
    """
    n = mask_matrix.shape[0]
    is_sparse = hasattr(mask_matrix, "toarray")

    if n < _DISPATCH_THRESHOLD:
        dense_matrix = mask_matrix.toarray() if is_sparse else mask_matrix
        return matrix_masked_sum_dense(dense_matrix, data)
    else:
        return matrix_masked_sum_sparse(mask_matrix, data)

# === Grouped sum autodispatch ===
def matrix_grouped_sum(group_matrix, values):
    """
    Auto-dispatch grouped sum using size threshold.
    Uses dense for small matrices, sparse for large ones.
    """
    n = group_matrix.shape[0]
    is_sparse = hasattr(group_matrix, "toarray")

    if n < _DISPATCH_THRESHOLD:
        dense_matrix = group_matrix.toarray() if is_sparse else group_matrix
        return matrix_grouped_sum_dense(dense_matrix, values)
    else:
        return matrix_grouped_sum_sparse(group_matrix, values)
