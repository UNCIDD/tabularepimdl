from pathlib import Path

from .yaml_exceptions import YAMLFileReadError


def read_yaml_file(path: Path) -> str:
    """
    Read a YAML file as UTF-8 text.

    Checks:
    - File accesibility
    - Reading the file
    - UTF-8 validation

    Args:
        path: Path to the YAML file.

    Returns:
        Contents of the file as a string.

    Raises:
        YAMLFileReadError:
            If the file cannot be read or decoded as UTF-8.
    """

    try:
        return path.read_text(encoding="utf-8")

    except UnicodeDecodeError as exc:
        raise YAMLFileReadError(
            f"YAML file is not valid UTF-8: {path}"
        ) from exc

    except PermissionError as exc:
        raise YAMLFileReadError(
            f"Permission denied while reading YAML file: {path}"
        ) from exc

    except OSError as exc:
        raise YAMLFileReadError(
            f"Failed to read YAML file: {path}: {exc}"
        ) from exc
