# TO DO: MOVE THESE TO A SEPERATE UTIL FILE
# === 4. Deterministic transition ===
@nb.njit(parallel=False, fastmath=True)
def apply_deterministic_transition(counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return counts * probs

# === 5. Stochastic transition ===
@nb.njit(parallel=False, fastmath=True)
def apply_stochastic_transition(counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    out = np.empty(counts.shape[0], dtype=np.int32)
    for i in range(counts.shape[0]):
        out[i] = np.random.binomial(counts[i], probs[i])
    return out
# TO DO: MOVE THESE TO A SEPERATE UTIL FILE
