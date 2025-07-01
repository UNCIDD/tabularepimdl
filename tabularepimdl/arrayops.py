# file: tabularepimdl/arrayops.py
from .arrayops_utils import (
    grouped_sum_vectorized,
    grouped_sum_serial,
    grouped_sum_parallel,
    grouped_count_vectorized,
    grouped_count_serial,
    grouped_count_parallel,
    masked_sum_vectorized,
    masked_sum_parallel,
    masked_sum_serial,
)
from typing import Tuple, Dict
import numpy as np
import numba as nb

# === 1. Grouped sum ===
def grouped_sum(values, group_ids, n_groups):
    """
    Public grouped_sum API — auto-dispatches to optimal implementation.
    """
    N = values.shape[0]

    if N >= 1_000_000:
        return grouped_sum_parallel(values, group_ids, n_groups)
    elif N >= 100_000:
        return grouped_sum_serial(values, group_ids, n_groups)
    else:
        return grouped_sum_vectorized(values, group_ids, n_groups)


def grouped_count(values, group_ids, n_groups):
    N = values.shape[0]

    group_ids_int = group_ids.astype(np.int32)

    if N >= 1_000_000:
        return grouped_count_parallel(group_ids_int, n_groups)
    elif N >= 100_000:
        return grouped_count_serial(group_ids_int, n_groups)
    else:
        return grouped_count_vectorized(group_ids_int, n_groups)



# === 3. Masked sum ===
def masked_sum(data, mask, T):
    """
    Public masked_sum API — auto-dispatches to optimal implementation.
    """
    N = data.shape[0]
    workload = N * T

    if workload >= 1e9:
        return masked_sum_parallel_numba(data, mask, N, T)
    elif workload >= 1e7:
        return masked_sum_numba(data, mask, N, T)
    else:
        return masked_sum_vectorized(data, mask)  # Removed N, T here



# === 6. Filter index ===
@nb.njit
def get_indices(mask: np.ndarray) -> np.ndarray:
    return np.nonzero(mask)[0]


# === 7. Categorical encoding ===
def encode_categories(categories: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    unique, inv = np.unique(categories, return_inverse=True)
    mapping = {name: i for i, name in enumerate(unique)}
    return inv, mapping
