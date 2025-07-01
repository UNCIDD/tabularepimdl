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
