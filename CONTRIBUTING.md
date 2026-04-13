# Contributing to tabularepimdl

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

- **Open an issue before starting work.** Even small fixes benefit from a recorded rationale.
- Use descriptive titles: prefer *"Refactor SimpleTransition: replace pandas with NumPy"* over *"update code"*.
- Assign a milestone if one applies (this repo uses thematic milestones, not version-based).
- Label with `bug`, `enhancement`, `feature`, `question`, or `documentation` as appropriate.

## Branches

- Branch from `main`, not from other feature branches.
- Name branches descriptively:  
  `issue-48-simpletransition-vectorize`  
  `fix-42-yaml-key-check`
- Never commit directly to `main`. The `main` branch is protected.
- Delete branches after their PR is merged.

## Commits

- **Keep commits thematic.** Each commit should represent one logical change.
  - Good: *"add NumPy vectorized get_deltas() for SimpleTransition"*
  - Bad: *"updates"*, *"debug code"*, *"fix"*
- Do not mix unrelated changes (e.g., a new feature + docstring reformatting + a bugfix) in one commit.
- Do not commit debug/print statements. Remove them before pushing.
- Write commit messages in imperative mood: *"add X"* not *"added X"*.

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

## Code organization

| Directory | Purpose |
|-----------|---------|
| `tabularepimdl/` | Installable package — production rules, engine, and utilities |
| `epitest/` | Pytest test suite |
| `examples/` | User-facing Jupyter notebook examples |
| `benchmark/` | Performance benchmarking scripts |
| `scripts/` | Experimental, superseded, and prototype code (see `scripts/README.md`) |
| `docs/` | Specification documents |

### What goes in `tabularepimdl/` vs `scripts/`

- **`tabularepimdl/`**: Code that is part of the public API, imported via `__init__.py`, tested, and documented.
- **`scripts/`**: Prototypes, superseded versions, project-specific variants, personal experiments, and benchmarking explorations. These are kept for reference but are not importable from the package.

To promote experimental code to production, see `scripts/README.md`.

## Testing

- Add or update tests in `epitest/` for any code change to `tabularepimdl/`.
- Run the test suite before opening a PR:
  ```bash
  pytest epitest/
  ```
- CI must pass before merge.

## Style

- Follow existing code conventions in the repo.
- Use type hints for function signatures.
- Use docstrings for public classes and methods.
- Run `ruff check` and `mypy` before pushing (these are already in the dev dependencies).
