from pathlib import Path

from .yaml_exceptions import (
    YAMLFileExtensionError,
    YAMLFileNotFoundError,
    YAMLFileTypeError,
)

SUPPORTED_YAML_EXTENSIONS = {".yaml", ".yml"}

def validate_yaml_path(filename: str | Path) -> Path:
    """
    Validate and normalize a YAML file path.

    Checks:
    - Path can be represented as a pathlib.Path.
    - File exists.
    - Path refers to a regular file.
    - File has a supported YAML extension.

    Args:
        filename: name or path of the YAML file being loaded.

    Returns:
        A normalized Path object.

    Raises:
        YAMLFileNotFoundError:
            If the file does not exist.

        YAMLFileTypeError:
            If the path exists but is not a regular file.

        YAMLFileExtensionError:
            If the file extension is not .yaml or .yml.
    """

    path = Path(filename).expanduser()

    if not path.exists():
        raise YAMLFileNotFoundError(
            f"YAML file does not exist: {path}"
        )

    if not path.is_file():
        raise YAMLFileTypeError(
            f"Path is not a regular file: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_YAML_EXTENSIONS:
        raise YAMLFileExtensionError(
            f"Unsupported YAML file extension: {path.suffix!r}. "
            f"Expected one of: "
            f"{', '.join(sorted(SUPPORTED_YAML_EXTENSIONS))}. "
            f"Path: {path}"
        )

    return path