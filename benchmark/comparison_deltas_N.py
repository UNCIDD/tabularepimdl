"""
Cross-structure correctness check for the benchmark *Runner.py scripts.

Notes:
    - Deterministic runs only. Under a stochastic run (stochastic=True), each structure draws its own
      independent random sample, so their outputs are not expected to match. The comparison is skipped in
    that case.
    - Structures whose last deltas were None or empty are skipped (nothing to compare).
    - Values are compared *sorted*, not by row position.
"""
import numpy as np
import pandas as pd


def _extract_n_values(deltas, col_idx_map: dict, n_col: str = "N") -> np.ndarray | None:
    """Return column N as a 1-D float array from a numpy-array or DataFrame deltas result.

    Returns:
        None if deltas is None or empty (nothing to compare for certain data structure).
    """
    if deltas is None:
        return None
    if isinstance(deltas, np.ndarray):
        if deltas.size == 0:
            return None
        return deltas[:, col_idx_map[n_col]].astype(float)
    if isinstance(deltas, pd.DataFrame):
        if deltas.empty:
            return None
        return deltas[n_col].to_numpy(dtype=float)
    raise TypeError(f"Unsupported deltas type for comparison: {type(deltas)}")


def compare_structure_deltas(
    last_deltas_by_struct: dict,
    col_idx_map: dict,
    rule_name: str,
    stochastic: bool = False,
    n_col: str = "N",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> None:
    """
    Compare each data structure's last-iteration deltas' column N values for a deterministic run.

    Args:
        last_deltas_by_struct: mapping of structure name (e.g. 'Pandas', 'Numpy_Encode') to its
        last get_deltas() result for the data size just benchmarked.
        
        col_idx_map: mapping of column name to column index, used to read N out of numpy-array
        deltas. Ignored for DataFrame deltas, which are indexed by column name directly.

        rule_name: name of the rule being benchmarked, used in the printed/raised message.
        
        stochastic: whether this benchmark run used stochastic=True. When True, comparison is
        skipped.
        
        n_col: name of the population-count column to compare. Defaults to 'N'.
        
        rtol, atol: tolerance passed to np.allclose, to absorb floating-point noise between
        independently-implemented structures rather than requiring bit-exact equality.

    Raises:
        ValueError: if two or more structures have non-empty deltas whose sorted N values disagree.
    """
    if stochastic:
        print(f"{rule_name}: skipping cross-structure comparison (stochastic run -- draws are expected to differ).")
        return

    extracted = {}
    for struct, deltas in last_deltas_by_struct.items():
        n_values = _extract_n_values(deltas, col_idx_map, n_col)
        if n_values is None:
            continue
        extracted[struct] = np.sort(n_values)

    if len(extracted) < 2:
        return  # nothing to compare against

    baseline_struct, baseline_values = next(iter(extracted.items()))
    mismatches = []
    for struct, values in extracted.items():
        if struct == baseline_struct:
            continue
        if values.shape != baseline_values.shape or not np.allclose(values, baseline_values, rtol=rtol, atol=atol):
            mismatches.append(struct)

    if mismatches:
        raise ValueError(
            f"{rule_name}: deltas' column '{n_col}' values do not match between data structures. "
            f"Baseline structure '{baseline_struct}' disagrees with: {mismatches}."
        )

    print(
        f"{rule_name}: results of {list(extracted.keys())} match -- deltas column '{n_col}' values "
        f"are consistent across data structures."
    )
