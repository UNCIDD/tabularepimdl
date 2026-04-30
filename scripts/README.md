# scripts/

Experimental, superseded, and benchmarking code that is **not part of the
installable `tabularepimdl` package**. These files are retained for reference
and future development.

## Directory layout

| Directory | Contents |
|-----------|----------|
| `experimental_engines/` | Iterative engine prototypes (`EpiModel_Vec_Encode1` through `1_5`, `2`, and the pre-refactor `EpiModel_orig`). None are production-ready; the canonical engine remains `tabularepimdl/EpiModel.py`. |
| `experimental_rules/` | Rule variants that are superseded, project-specific, or personal explorations (e.g., `WAIFWTransmission_Mpox`, `SI_Josh_Encode_Vec`, nobuffer variants). |
| `experimental_ops/` | Auto-dispatching array/matrix operation modules (`arrayops`, `matrixops`, `operations`) with Numba JIT backends. Used only by experimental rules. |
| `experimental_examples/` | Notebooks for benchmarking, Kronecker product exploration, and EpiRunner prototyping. Not user-facing examples. |
| `experimental_docs/` | Planning notebooks (e.g., future user-interface steps). |

## Promoting code to production

If an experimental module matures to production quality:

1. Open an issue describing what it does and why it should be promoted.
2. Move it into `tabularepimdl/` and add it to `__init__.py`.
3. Add or update tests in `epitest/`.
4. Submit a PR following the workflow in `CONTRIBUTING.md`.
