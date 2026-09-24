# Support the overall YAML file loading error classification

from __future__ import annotations

class EpiModelValidationError(Exception):
    """Base exception for tabularepimdl input/validation errors."""

# ---------------------------------------------------------------------------
# File-related errors
# ---------------------------------------------------------------------------


class YAMLFileError(EpiModelValidationError):
    """Base exception for filesystem-related YAML errors."""


class YAMLFileNotFoundError(YAMLFileError):
    """Raised when the YAML file does not exist."""


class YAMLFileTypeError(YAMLFileError):
    """Raised when the path is not a regular file."""


class YAMLFileExtensionError(YAMLFileError):
    """Raised when the file does not have a supported YAML extension."""


class YAMLFileReadError(YAMLFileError):
    """Raised when the YAML file cannot be read."""


# ---------------------------------------------------------------------------
# YAML parsing errors
# ---------------------------------------------------------------------------


class YAMLParseError(EpiModelValidationError):
    """Raised when the file contains invalid YAML syntax."""


class DuplicateYAMLKeyError(YAMLParseError):
    """
    Raised when a YAML mapping contains a duplicate key.
    """

    def __init__(
        self,
        *,
        filename: str,
        key: object,
        path: str,
        original_line: int,
        original_column: int,
        duplicate_line: int,
        duplicate_column: int,
    ) -> None:
        self.filename = filename
        self.key = key
        self.path = path
        self.original_line = original_line
        self.original_column = original_column
        self.duplicate_line = duplicate_line
        self.duplicate_column = duplicate_column

        message = (
            f"Duplicate key {key!r}\n"
            f"File: {filename}\n"
            f"Path: {path}\n"
            f"Original occurrence: "
            f"Line {original_line}, Column {original_column}\n"
            f"Duplicate occurrence: "
            f"Line {duplicate_line}, Column {duplicate_column}"
        )

        super().__init__(message)


# ---------------------------------------------------------------------------
# YAML structure errors
# ---------------------------------------------------------------------------
class YAMLStructureError(EpiModelValidationError):
    """Raised when valid YAML has an unexpected structure."""


# ---------------------------------------------------------------------------
# EpiModel/domain errors
# --------------------------------------------------------------------------
#this class may not be used
class EpiModelDataError(EpiModelValidationError):
    """Raised when valid YAML does not satisfy the EpiModel schema."""
