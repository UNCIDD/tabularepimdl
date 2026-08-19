"""Shared time/memory measurement helper for the benchmark/*Runner.py scripts.

Every Runner's run() loop repeated the same gc.collect() / tracemalloc / perf_counter
block once per structure branch. This factors that out so each Runner only supplies
the structure-specific call to make.
"""
import gc
import time
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd


def measure(struct: str, get_deltas_fn: Callable[[], pd.DataFrame | np.ndarray], iters: int, col_idx_map: dict) -> tuple:
    """Run get_deltas_fn iters times, tracking elapsed time and peak memory.

    Args:
        struct: structure name (e.g. 'Pandas', 'Numpy_Encode'), used in the debug print.
        get_deltas_fn: zero-arg callable that returns one iteration's deltas.
        iters: number of times to call get_deltas_fn.
        col_idx_map: column-name-to-index map, used to read column N out of the last
        iteration's deltas for the debug print (ignored for DataFrame deltas).

    Returns:
        (deltas, time_sec, peak_memory_MB), note deltas is the last iteration's result.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        deltas = get_deltas_fn()
    t1 = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    non_zero = np.count_nonzero(
        deltas[:, col_idx_map["N"]] if isinstance(deltas, np.ndarray) else deltas.loc[:, "N"]
    )
    print(
        f"Sample deltas for {struct}:\n{deltas}\n, data length: {len(deltas)}\n, "
        f"non-zero counts: {non_zero}"
    )
    return deltas, round(t1 - t0, 3), round(peak / 1024**2, 2)
