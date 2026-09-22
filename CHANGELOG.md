# Changelog

All notable changes to `tabularepimdl` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-27

A comprehensive engineering pass turning the NumPy rewrite into a tested, maintainable,
release-track package.

### Added

- Automated test suite for the NumPy rule/engine family, now at 90% statement coverage
  (`pytest --cov`): dedicated unit tests for every previously-untested `_Vec_Encode` rule
  (`HospRule_Vec_Encode`, `EnvironmentalTransmission_Vec_Encode`,
  `MultiStrainInfectiousProcess_Vec_Encode`, `SharedTraitInfection_Vec_Encode`, and the rest of
  the rule library), an `EpiModel_Vec_Encode_1_5` engine-level test suite, and a pandas-vs-NumPy
  engine parity test that continuously validates the NumPy engine against the original pandas
  implementation ([#137](https://github.com/UNCIDD/tabularepimdl/pull/137),
  [#142](https://github.com/UNCIDD/tabularepimdl/pull/142),
  [#143](https://github.com/UNCIDD/tabularepimdl/pull/143)).
- Cross-structure correctness check (`benchmark/comparison_deltas_N.py`) run by every
  `benchmark/*Runner.py` script, comparing each data structure's computed deltas for agreement
  and raising if any disagree ([#139](https://github.com/UNCIDD/tabularepimdl/pull/139)).
- GitHub Actions CI running the test suite, `ruff`, and `mypy` on every push and pull request
  targeting `main` ([#141](https://github.com/UNCIDD/tabularepimdl/pull/141)); `ruff` and `mypy`
  made fully blocking after a dedicated lint/type-debt cleanup
  ([#151](https://github.com/UNCIDD/tabularepimdl/pull/151)).
- Opt-in debug logging (`tabularepimdl.configure_logging()`) across the package, replacing ad hoc
  `print()` calls; silent by default, matching standard library-vs-application logging convention
  ([#146](https://github.com/UNCIDD/tabularepimdl/pull/146)).
- `legacy/pandas_reference/`: the original pandas rule/engine implementation, retained as a
  deliberately-maintained internal baseline to validate the NumPy rewrite, no longer part of the
  installable package's public API ([#148](https://github.com/UNCIDD/tabularepimdl/pull/148)).
- `benchmark/timing.py`, a shared time/memory-measurement helper removing ~10 lines of duplicated
  `gc`/`tracemalloc`/`perf_counter` boilerplate from each of the 8 `benchmark/*Runner.py` scripts
  ([#149](https://github.com/UNCIDD/tabularepimdl/pull/149)).
- `Rule` now formally declares `expansion_factor` and `_encode_categorical_states()` as part of
  its interface, so every NumPy rule's contract with the model engine is explicit rather than
  implicit ([#151](https://github.com/UNCIDD/tabularepimdl/pull/151)).
- `Rule.get_deltas()`'s `col_idx_map` and `result_buffer` parameters are now required rather than
  optional -- any external `Rule` subclass or caller relying on the old `None`-able defaults will
  need to pass them explicitly.

### Changed

- Moved the package to a standard `src/` layout
  ([#144](https://github.com/UNCIDD/tabularepimdl/pull/144)).
- Consolidated dependency management onto `pyproject.toml`; removed `setup.py` and
  `requirements.txt` ([#142](https://github.com/UNCIDD/tabularepimdl/pull/142),
  [#143](https://github.com/UNCIDD/tabularepimdl/pull/143)).
- Renamed `epitest/` to `tests/` and `*_pytest.py` files to `test_*.py` to match standard Python
  convention ([#142](https://github.com/UNCIDD/tabularepimdl/pull/142)).
- Simplified `Rule.get_deltas()`'s abstract signature (`col_idx_map`/`result_buffer` are now
  required, not optional) now that NumPy is the only production rule family
  ([#148](https://github.com/UNCIDD/tabularepimdl/pull/148)).

### Fixed

- Fixed a combinatorial-explosion bug in `EpiModel_Vec_Encode_1_5`'s delta-buffer sizing
  (multiplying, rather than summing, every rule's expansion factor across a ruleset), which could
  cause the engine to attempt allocating an astronomically large buffer and crash for realistic
  rulesets containing several high-fan-out rules
  ([#150](https://github.com/UNCIDD/tabularepimdl/pull/150)).
- Fixed a CWD-dependent test failure, several stale test fixtures, and a missing
  `.reset_index(drop=True)` correctness bug in `MultiStrainInfectiousProcess.get_deltas()` that
  could return deltas with a duplicated, non-unique index
  ([#137](https://github.com/UNCIDD/tabularepimdl/pull/137)).
- Fixed `HospRule_Vec_Encode`'s `expansion_factor` and `_encode_categorical_states()`, both broken
  for any model using the engine-driven (rather than rule-level self-encoding) categorical-state
  encoding path.
- Fixed broken import paths across `benchmark/` and `legacy/pandas_reference/` left over from the
  `src/` layout move and the pandas-rules relocation
  ([#148](https://github.com/UNCIDD/tabularepimdl/pull/148),
  [#149](https://github.com/UNCIDD/tabularepimdl/pull/149)).

### Removed

- Removed the pandas-structured rules and engine (`EpiModel`, `BirthProcess`, `SimpleInfection`,
  etc.) from the installable package's public API; relocated to `legacy/pandas_reference/` as an
  internal-only comparison baseline, invisible to anyone installing the package from PyPI (future plan)
  ([#148](https://github.com/UNCIDD/tabularepimdl/pull/148)).
- Removed several superseded/duplicate rule variants (`SimpleTransition_Vec`, `ST_Josh_Encode_Vec`,
  `SI_Josh_Encode_Vec`, `SimpleObservationProcess_Vec_Encode_nobuffer`,
  `MultiStrainInfectiousProcess_Vec_Encode_2`) in favor of their maintained counterparts.

## [0.2.0] - 2025-12-12

### Added

- NumPy-vectorized rewrite of the rule library and model engine, addressing pandas performance
  limitations on large simulations: `SimpleInfection_Vec_Encode`, `SimpleTransition_Vec_Encode`,
  `SimpleObservationProcess_Vec_Encode`, `StateBasedDeathProcess_Vec_Encode`,
  `SharedTraitInfection_Vec_Encode`, `WAIFWTransmission_Vec_Encode_Bincount`, and the supporting
  `_types`/`_validators` modules ([#70](https://github.com/UNCIDD/tabularepimdl/pull/70)).


Release 0.2.0 Note: The 0.2.0 release was never formally implemented. During development, the codebase was reorganized across different directories, with some code moved or removed, making it impractical to reconstruct the codebase as it existed at the intended 0.2.0 release. This note serves as a record of the development work and changes associated with the 0.2.0 milestone.

## [0.1.0] - 2025-09-04

### Added

- Initial pandas-structured rule library and `EpiModel` engine: `SimpleInfection`,
  `SimpleTransition`, `SimpleObservationProcess`, `StateBasedDeathProcess`,
  `MultiStrainInfectiousProcess`, `SharedTraitInfection`, `WAIFWTransmission`, `BirthProcess`, and
  `Rule`'s YAML-based rule configuration loading.
- Example notebooks demonstrating aging-population, household-structure, and multi-strain models
  ([#21](https://github.com/UNCIDD/tabularepimdl/pull/21),
  [#22](https://github.com/UNCIDD/tabularepimdl/pull/22)).
- Trait-sharing and value-filtering rule variants (`WithFilters`, `FilteredSimpleInfection`,
  `FilteredSimpleTransition`, `SharedTraitInfection_MultiBeta`)
  ([#39](https://github.com/UNCIDD/tabularepimdl/pull/39)).

Release 0.1.0 Note: The 0.1.0 release was never formally implemented. During development, the codebase was reorganized across different directories, with some code moved or removed, making it impractical to reconstruct the codebase as it existed at the intended 0.1.0 release. This note serves as a record of the development work and changes associated with the 0.1.0 milestone.