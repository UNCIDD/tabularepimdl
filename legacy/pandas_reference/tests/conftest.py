import sys
from pathlib import Path

# from legacy.pandas_reference.X import Y needs the repo root on sys.path. Walk up from this
# file's own location (not cwd -- pytest may be invoked from anywhere) to find it.
root = Path(__file__).resolve().parent
while not (root / "pyproject.toml").exists():
    if root.parent == root:
        raise RuntimeError("Could not locate repo root (no pyproject.toml found in any parent)")
    root = root.parent

sys.path.insert(0, str(root))
