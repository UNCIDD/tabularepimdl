# Run `just --list` to see all available commands.
# `just` is optional -- every recipe here is a thin wrapper around a `uv run ...` command
# users can always run directly instead. See CONTRIBUTING.md.

set windows-shell := ["cmd.exe", "/c"]

default:
    just --list

# Run the main test suite (same scope as CI)
test:
    uv run pytest tests/

# Run the main test suite with a coverage report
cov:
    uv run pytest --cov=tabularepimdl --cov-report=term-missing tests/

# Run the legacy pandas-reference test suite (not part of CI's default scope)
test-legacy:
    uv run pytest legacy/pandas_reference/tests

# Lint using `ruff` (same as CI; add --fix locally to auto-fix what's fixable)
check:
    uv run ruff check .

# Auto-fix lint issues and apply formatting using `ruff`
format:
    uv run ruff check --fix .
    uv run ruff format .

# Static type checking (same as CI) using `mypy`
typecheck:
    uv run mypy

# Run everything CI runs, in the same order CI runs it
ci: check typecheck test

# Build the distributable package (sdist + wheel)
build:
    uv build
