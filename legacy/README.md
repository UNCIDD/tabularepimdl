# legacy

This directory contains inital pandas-structured rules and the model engine, evolved model engines that are now expired or superseded, and initial examples that is **not part of the installable `tabularepimdl` package**. These files are retained for reference and future development.

## Directory layout

| Directory | Contents |
|-----------|----------|
| `pandas_reference/` | The original pandas-structured rules and engine that predate the NumPy rewrite. Not a prototype -- this is a deliberately-maintained reference baseline: `benchmark/` and `tests/test_engine_parity.py` run it alongside the NumPy engine to validate that the rewrite produces the same results. |
| `experimental_engines/` | Iterative NumPy engine prototypes. None are production-ready. |
| `experimental_rules/` | Rule variants that are superseded, project-specific, or personal explorations. |
| `experimental_ops/` | Auto-dispatching array/matrix operation modules with Numba JIT backends. |
| `experimental_examples/` | Notebooks for benchmarking, Kronecker product exploration, and EpiRunner prototyping. Not user-facing examples. |
| `experimental_docs/` | Planning notebooks (e.g., future user-interface steps). |
