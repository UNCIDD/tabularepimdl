# legacy/

Experimental, superseded, and benchmarking code that is **not part of the
installable `tabularepimdl` package**. These files are retained for reference
and future development.

## Directory layout

| Directory | Contents |
|-----------|----------|
| `pandas_reference/` | The original pandas-structured rules and engine (`SimpleTransition`, `EpiModel`, etc.) that predate the NumPy rewrite. Not a prototype -- this is a deliberately-maintained reference baseline: `benchmark/` and `tests/test_engine_parity.py` run it alongside the NumPy engine to validate that the rewrite produces the same results. The canonical, shipped engine is in `src/tabularepimdl/`. |
| `experimental_engines/` | Iterative NumPy engine prototypes (`EpiModel_Vec_Encode1` through `1_5`, `2`, and the pre-refactor `EpiModel_orig`). None are production-ready. |
| `experimental_rules/` | Rule variants that are superseded, project-specific, or personal explorations (e.g., `WAIFWTransmission_Mpox`, `SI_Josh_Encode_Vec`, nobuffer variants). |
| `experimental_ops/` | Auto-dispatching array/matrix operation modules (`arrayops`, `matrixops`, `operations`) with Numba JIT backends. Used only by experimental rules. |
| `experimental_examples/` | Notebooks for benchmarking, Kronecker product exploration, and EpiRunner prototyping. Not user-facing examples. |
| `experimental_docs/` | Planning notebooks (e.g., future user-interface steps). |

## Promoting code to production

If an experimental module matures to production quality:

1. Open an issue describing what it does and why it should be promoted.
2. Move it into `src/tabularepimdl/` and add it to `__init__.py`.
3. Add or update tests in `tests/`.
4. Submit a PR following the workflow in `CONTRIBUTING.md`.
