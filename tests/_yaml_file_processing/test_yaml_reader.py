from pathlib import Path
from unittest.mock import patch

import pytest

from tabularepimdl._validators.yaml_file_processing.yaml_reader import read_yaml_file

from tabularepimdl._validators.yaml_file_processing.yaml_exceptions import (
    YAMLFileReadError,
)


def test_read_yaml_file_success(tmp_path: Path) -> None:
    """Test that a valid YAML file is read as UTF-8 text."""

    # Create a temporary YAML file.
    yaml_content = "column: InfState\nfrom_st: S\nto_st: I\n"
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Read the YAML file.
    result = read_yaml_file(yaml_file)

    # The returned text matches the file contents.
    assert result == yaml_content


def test_read_yaml_file_invalid_utf8(tmp_path: Path) -> None:
    """Test that invalid UTF-8 content raises YAMLFileReadError."""

    # Write bytes that are not valid UTF-8.
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_bytes(b"name: Alice\nvalue: \xff")

    # Verify the custom exception is raised.
    with pytest.raises(YAMLFileReadError,
                        match="YAML file is not valid UTF-8",
    ):
        read_yaml_file(yaml_file)


def test_read_yaml_file_permission_error(tmp_path: Path) -> None:
    """Test that a permission error is wrapped in YAMLFileReadError."""

    # Create a temporary file path.
    yaml_file = tmp_path / "restricted.yaml"

    # Mock Path.read_text so the test behaves consistently across
    # operating systems and does not depend on actual file permissions.
    with patch.object(
        Path,
        "read_text",
        side_effect=PermissionError("Access denied"),
    ):
        with pytest.raises(
            YAMLFileReadError,
            match="Permission denied while reading YAML file",
        ) as exc_info:
            read_yaml_file(yaml_file)

    # Verify that the original exception is preserved as the cause.
    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_read_yaml_file_missing_file(tmp_path: Path) -> None:
    """Test that a missing file raises YAMLFileReadError."""

    # Create a path that does not exist.
    yaml_file = tmp_path / "missing.yaml"

    with pytest.raises(YAMLFileReadError,
                       match="Failed to read YAML file",
    ):
        read_yaml_file(yaml_file)


def test_read_yaml_file_generic_os_error(tmp_path: Path) -> None:
    """Test that other OSErrors are wrapped in YAMLFileReadError."""

    #  Create a temporary file path.
    yaml_file = tmp_path / "config.yaml"

    with patch.object(
        Path,
        "read_text",
        side_effect=OSError("Disk read failure"),
    ):        
        with pytest.raises(
            YAMLFileReadError,
            match="Failed to read YAML file",
        ):
            read_yaml_file(yaml_file)