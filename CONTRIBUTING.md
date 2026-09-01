# Contributing
This document provides guidelines and instructions for contributing to `tabularepimdl`, including development conventions and tips for best practices.


## Development Setup

### Prerequisites
- Python 3.11 or 3.12.
- [`uv`](https://docs.astral.sh/uv/) - Python package manager.
- [`just`](https://just.systems/) - Command runner (optional -- see below).

### Initial setup

1. Clone the repository:

```shell
git clone https://github.com/UNCIDD/tabularepimdl.git
cd tabularepimdl
```

2. Install dependencies:

```shell
uv sync --all-extras
```

`uv sync` creates and manages `.venv` for users (no separate `python -m venv` step needed) and
installs the package itself in editable mode, `--all-extras` pulls in both the `test` and `dev` dependency
groups.

To run a command inside the environment without activating it, prefix it with `uv run` (e.g.
`uv run pytest`) -- this is what the `justfile` recipes below do. To activate the environment
directly in users' shell instead:

```shell
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Using `just` to verify the setup

This repo has a `justfile` with shortcuts for the commands below. Run `just --list` to see all of
them, or `just` (no argument) for the same. A few examples:

```shell
just test       # pytest tests/
just check      # ruff check .
just typecheck  # mypy
just ci         # check + typecheck + test, in CI's order
```

`just` is entirely optional -- every recipe is a thin wrapper around a `uv run ...` command users
can run directly instead, and both are shown throughout this document.

## Code Standards

### Code Style

[`ruff`](https://docs.astral.sh/ruff/) is used for both formatting and linting python:

- **Formatting**: `ruff format` follows the Black code style.
- **Linting**: `ruff check` enforces code quality rules.

### Docstring Style

[google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
comments and docstrings are used to document code.

## Workflow overview

Every change to this repository follows the **issue → branch → PR → review → merge** cycle. No work should land on `main` without a pull request.

```
1. Open an issue describing the work
2. Create a feature branch from main
3. Make focused commits on that branch
4. Open a PR linking the issue
5. Get review, iterate
6. Merge (squash or rebase) into main
7. Delete the feature branch
```

## Issues

- Open an issue before starting work.
- Use descriptive titles.
- Label with `bug`, `enhancement`, `feature`, `question`, `documentation`, etc. as appropriate.

## Branches

- Branch from `main`, not from other feature branches.
- Name branches descriptively.
- Please use a feature branch instead of committing directly to `main`, as `main` is protected.
- Delete branches after their PR is merged.

## Commits

- Keep commits thematic. Each commit should represent one logical change.
- Keep commits focused on a single, logical change. Avoid combining unrelated feature, formatting, or bug-fix work in the same commit.
- Before pushing, take a quick pass to remove any leftover debug or print statements.
- Write commit messages in imperative mood: *"add X"* instead of *"added X"*.

## Pull requests

### Size limits

- **A single PR should not exceed ~1,000 lines of diff.** Ideally keep PRs shorter (200–500 lines).
- If an issue requires more than 1,000 lines, break it into multiple PRs:
  - PR 1: core logic
  - PR 2: tests
  - PR 3: examples / documentation
- Large PRs are hard to review, slow to merge, and risky to revert. Smaller PRs get faster, better reviews.

### Linking issues

- Reference the issue in the PR body: `Closes #48` or `Relates to #48`.
- Use `Closes` for issues fully resolved by the PR. Use `Relates to` for partial progress.
- A single issue may have multiple PRs (that's fine and encouraged for large work).

### PR description

Include:
1. **What** changed and **why**.
2. Which issue(s) it addresses.
3. How to test or verify the change.
4. Any follow-up work remaining.

### Review

- All PRs require at least one approving review before merge.
- Address review comments with new commits (don't force-push during review unless asked).
- Use the GitHub "Resolve conversation" button to mark addressed feedback.

### Merging

- Prefer **squash merge** for single-purpose PRs (keeps `main` history clean).
- Use **rebase merge** if the individual commits are meaningful and well-structured.
- Never use merge commits.

## Codebase organization

| Directory | Purpose |
|-----------|---------|
| `src/tabularepimdl/` | Installable package — production rules, engine, and utilities |
| `tests/` | Pytest test suite |
| `examples/` | User-facing Jupyter notebook examples |
| `benchmark/` | Performance benchmarking scripts |
| `legacy/` | Experimental, superseded, and prototype code (see `legacy/README.md`) |
| `docs/` | Specification documents |

### Directory `src/tabularepimdl/` vs `legacy/`

- **`src/tabularepimdl/`**: Code that is part of the public API, imported via `__init__.py`, tested, and documented.
- **`legacy/`**: Prototypes, superseded versions, project-specific variants, personal experiments, and benchmarking explorations. These are kept for reference but are not importable from the package.

## Testing

- Add or update tests in `tests/` for any code change to `src/tabularepimdl/`.
- Run the test suite before opening a PR:
```shell
  just test          # or: uv run pytest tests/
```
- `legacy/pandas_reference/tests/` is a separate suite validating the internal pandas reference
  baseline against the shipped NumPy engine (see `legacy/README.md`). It isn't part of CI's default
  scope, but run it too if your change touches `Rule.py`, the model engine, or anything the parity
  test depends on:
```shell
  just test-legacy    # or: uv run pytest legacy/pandas_reference/tests
```
- CI must pass before merge.

## Packaging

Before a release, verify the package actually builds and installs correctly from a clean
environment:

```shell
just build           # or: uv build
```

This produces an sdist and wheel under `dist/`. If you're changing `pyproject.toml`'s
`[project]` metadata or `[build-system]` configuration, run this locally to confirm the build
still succeeds.

## Style

- Follow existing code conventions in the repo.
- Use type hints for function signatures.
- Use docstrings for public classes and methods.
- Run lint and type checks before pushing -- both are blocking in CI:
 ```shell
  just check          # or: uv run ruff check .
  just typecheck       # or: uv run mypy
  just ci              # runs check + typecheck + test together
```
- `legacy/` and `examples/` are excluded from `ruff`/`mypy`'s default scope (see
  `pyproject.toml`'s `[tool.ruff] exclude`) -- they're reference/experimental code, not held to
  the same lint bar as `src/tabularepimdl/`, `tests/`, and `benchmark/`.

## Changelog

User-facing changes (new rules, engine behavior changes, bug fixes, breaking changes) should get
an entry in `CHANGELOG.md` under `[Unreleased]`, in the appropriate `Added`/`Changed`/`Fixed`/
`Removed` section. Internal-only changes (refactors with no behavior change, test-only changes,
CI tweaks) don't need one.
