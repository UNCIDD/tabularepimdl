import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity as sparse_identity
from tabularepimdl.matrixops_utils import (
    matrix_grouped_sum_dense,
    matrix_grouped_sum_sparse,
    matrix_grouped_count_dense,
    matrix_grouped_count_sparse,
    matrix_masked_sum_dense,
    matrix_masked_sum_sparse,
)

def test_matrix_grouped_sum():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    group_ids = np.array([0, 1, 0, 1])
    G_dense = encode_dense_groups(group_ids, 2)
    G_sparse = encode_sparse_groups(group_ids, 2)

    expected = np.array([4.0, 6.0])
    np.testing.assert_allclose(matrix_grouped_sum_dense(values, G_dense), expected)
    np.testing.assert_allclose(matrix_grouped_sum_sparse(values, G_sparse), expected)

def test_matrix_grouped_count():
    group_ids = np.array([0, 1, 0, 1])
    G_dense = encode_dense_groups(group_ids, 2)
    G_sparse = encode_sparse_groups(group_ids, 2)

    expected = np.array([2.0, 2.0])
    np.testing.assert_allclose(matrix_grouped_count_dense(G_dense), expected)
    np.testing.assert_allclose(matrix_grouped_count_sparse(G_sparse), expected)

def test_matrix_masked_sum():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    mask_sparse = csr_matrix(mask)

    expected = np.array([4.0, 6.0])
    np.testing.assert_allclose(matrix_masked_sum_dense(values, mask), expected)
    np.testing.assert_allclose(matrix_masked_sum_sparse(values, mask_sparse), expected)

