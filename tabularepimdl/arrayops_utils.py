import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def grouped_sum_serial(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Numba JIT grouped sum (serial).
    """
    result = np.zeros(n_groups, dtype=values.dtype)
    for i in range(values.shape[0]):
        result[group_ids[i]] += values[i]
    return result

@nb.njit(parallel=True, fastmath=True)
def grouped_sum_parallel(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Numba JIT grouped sum (parallel safe with thread-local accumulation).
    """
    tmp = np.zeros((nb.get_num_threads(), n_groups), dtype=values.dtype)
    for i in nb.prange(values.shape[0]):
        thread_id = nb.get_thread_id()
        tmp[thread_id, group_ids[i]] += values[i]
    result = np.zeros(n_groups, dtype=values.dtype)
    for t in range(tmp.shape[0]):
        for g in range(n_groups):
            result[g] += tmp[t, g]
    return result

def grouped_sum_vectorized(values: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Pure NumPy vectorized grouped sum.
    """
    result = np.zeros(n_groups, dtype=values.dtype)
    np.add.at(result, group_ids, values)
    return result



@nb.njit(fastmath=True)
def grouped_count_serial(group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Numba JIT grouped count (serial).
    """
    result = np.zeros(n_groups, dtype=np.int32)
    for i in range(group_ids.shape[0]):
        result[group_ids[i]] += 1
    return result

@nb.njit(parallel=True, fastmath=True)
def grouped_count_parallel(group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Numba JIT grouped count (parallel safe with thread-local accumulation).
    """
    tmp = np.zeros((nb.get_num_threads(), n_groups), dtype=np.int32)
    for i in nb.prange(group_ids.shape[0]):
        thread_id = nb.get_thread_id()
        tmp[thread_id, group_ids[i]] += 1
    result = np.zeros(n_groups, dtype=np.int32)
    for t in range(tmp.shape[0]):
        for g in range(n_groups):
            result[g] += tmp[t, g]
    return result

def grouped_count_vectorized(group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Pure NumPy vectorized grouped count.
    """
    result = np.zeros(n_groups, dtype=np.int32)
    np.add.at(result, group_ids, 1)
    return result

@nb.njit(fastmath=True)
def masked_sum_serial(values: np.ndarray, mask: np.ndarray) -> float:
    """
    Numba JIT masked sum (serial).
    """
    total = 0.0
    for i in range(values.shape[0]):
        if mask[i]:
            total += values[i]
    return total

@nb.njit(parallel=True, fastmath=True)
def masked_sum_parallel(values: np.ndarray, mask: np.ndarray) -> float:
    """
    Numba JIT masked sum (parallel safe with thread-local accumulation).
    """
    tmp = np.zeros(nb.get_num_threads(), dtype=values.dtype)
    for i in nb.prange(values.shape[0]):
        if mask[i]:
            thread_id = nb.get_thread_id()
            tmp[thread_id] += values[i]
    return tmp.sum()

def masked_sum_vectorized(values: np.ndarray, mask: np.ndarray) -> float:
    """
    Pure NumPy vectorized masked sum.
    """
    return np.sum(values[mask])
