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

    group_ids_int = group_ids.astype(np.int64)

    if N >= 1_000_000:
        return grouped_count_parallel(group_ids_int, n_groups)
    elif N >= 100_000:
        return grouped_count_serial(group_ids_int, n_groups)
    else:
        return grouped_count_vectorized(group_ids_int, n_groups)



# === 3. Masked sum ===
def masked_sum(values, mask):
    """
    Public masked_sum API — auto-dispatches to optimal implementation
    based on input size.

    Parameters
    ----------
    values : np.ndarray
        Array of values to sum.
    mask : np.ndarray
        Boolean mask indicating which entries to include.

    Returns
    -------
    float
        The masked sum of values.
    """
    N = values.shape[0]
    workload = N  # Only depends on N now

    if workload >= 1e9:
        return masked_sum_parallel(values, mask)
    elif workload >= 1e7:
        return masked_sum_serial(values, mask)
    else:
        return masked_sum_vectorized(values, mask)





