from tabularepimdl.arrayops import grouped_sum as array_grouped_sum
from tabularepimdl.arrayops import grouped_count as array_grouped_count
from tabularepimdl.arrayops import masked_sum as array_masked_sum
from tabularepimdl.matrixops import matrix_grouped_sum, matrix_grouped_count, matrix_masked_sum

# ===  Deterministic transition ===
@nb.njit(parallel=False, fastmath=True)
def apply_deterministic_transition(counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return counts * probs

# ===  Stochastic transition ===
@nb.njit(parallel=False, fastmath=True)
def apply_stochastic_transition(counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    out = np.empty(counts.shape[0], dtype=np.int32)
    for i in range(counts.shape[0]):
        out[i] = np.random.binomial(counts[i], probs[i])
    return out

# === Filter index ===
@nb.njit
def get_indices(mask: np.ndarray) -> np.ndarray:
    return np.nonzero(mask)[0]


# === Categorical encoding ===
def encode_categories(categories: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    unique, inv = np.unique(categories, return_inverse=True)
    mapping = {name: i for i, name in enumerate(unique)}
    return inv, mapping



def grouped_sum_meta(values, group_ids=None, group_matrix=None):
    """
    Meta-dispatch for grouped sum operation based on problem size.

    Parameters
    ----------
    values : np.ndarray
        1D array of values to be grouped.
    group_ids : np.ndarray or None
        Group labels for array-based implementation (required if using array backend).
    group_matrix : np.ndarray or None
        Binary indicator matrix for matrix-based implementation (required if using matrix backend).

    Returns
    -------
    np.ndarray
        Grouped sum result.
    """
    N = values.shape[0]
    if group_ids is not None:
        G = int(group_ids.max()) + 1
    elif group_matrix is not None:
        G = group_matrix.shape[0]
    else:
        raise ValueError("Either group_ids or group_matrix must be provided.")

    if (N <= 1e5 and G <= 100) or (N <= 1e6 and G <= 25):
        if group_matrix is None:
            raise ValueError("group_matrix required for matrix implementation")
        return matrix_grouped_sum(group_matrix, values)
    else:
        if group_ids is None:
            raise ValueError("group_ids required for array implementation")
        return array_grouped_sum(values, group_ids, G)

def grouped_count_meta(values, group_ids=None, group_matrix=None):
    """
    Meta-dispatch for grouped count operation based on problem size.

    Parameters
    ----------
    values : np.ndarray
        1D array of values to be grouped.
    group_ids : np.ndarray or None
        Group labels for array-based implementation (required if using array backend).
    group_matrix : np.ndarray or None
        Binary indicator matrix for matrix-based implementation (required if using matrix backend).

    Returns
    -------
    np.ndarray
        Grouped sum result.
    """
    N = values.shape[0]
    if group_ids is not None:
        G = int(group_ids.max()) + 1
    elif group_matrix is not None:
        G = group_matrix.shape[0]
    else:
        raise ValueError("Either group_ids or group_matrix must be provided.")

    if (N <= 1e5 and G <= 100) or (N <= 1e6 and G <= 25):
        if group_matrix is None:
            raise ValueError("group_matrix required for matrix implementation")
        return matrix_grouped_count(group_matrix)
    else:
        if group_ids is None:
            raise ValueError("group_ids required for array implementation")
        return array_grouped_count(values, group_ids, G)


