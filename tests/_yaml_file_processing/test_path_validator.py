from pathlib import Path

import pytest

from tabularepimdl._validators.yaml_file_processing.path_validator import validate_yaml_path
from tabularepimdl._validators.yaml_file_processing.yaml_exceptions import YAMLFileExtensionError, YAMLFileNotFoundError, YAMLFileTypeError


def test_valid_yaml_file_returns_path(tmp_path):
    """
    Create a yaml file within the temporary directory and
    validate it is a Path.
    """
    # create a valid YAML file
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("rules: SimpleTransition\n")
    
    result = validate_yaml_path(yaml_file)
    
    assert result == yaml_file
    assert isinstance(result, Path)


def test_valid_yaml_file_string_input(tmp_path):
    """
    Create a .yaml file within the temporary directory and convert its path to a string,
    validate the returned result is a Path.
    """
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("rules: SimpleTransition\n")

    result = validate_yaml_path(str(yaml_file))

    assert result == yaml_file
    assert isinstance(result, Path)


def test_missing_file_raises_not_found_error(tmp_path):
    """Create a path to a file, but not actually create the file itself. Expect YAMLFileNotFoundError."""
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(YAMLFileNotFoundError,
                       match="YAML file does not exist"):
        validate_yaml_path(missing_file)


def test_directory_raises_file_type_error(tmp_path):
    """Create a directory instead of a file. Expect YAMLFileTypeError."""
    directory = tmp_path / "configs"
    directory.mkdir()

    with pytest.raises(YAMLFileTypeError,
                       match="Path is not a regular file"):
        validate_yaml_path(directory)


def test_unsupported_extension_raises_extension_error(tmp_path):
    """Create a .txt file within the temporary directory. Expect YAMLFileExtensionError."""
    text_file = tmp_path / "config.txt"
    text_file.write_text("It is text file.\n")

    with pytest.raises(YAMLFileExtensionError,
                       match="Unsupported YAML file extension"):
        validate_yaml_path(text_file)


@pytest.mark.parametrize(
    "extension",
    [".yaml", ".yml", ".YAML", ".YML", ".Yaml"],
)
def test_supported_extensions(tmp_path, extension):
    """Check YAML file extensions regard to case."""
    yaml_file = tmp_path / f"config{extension}"
    yaml_file.write_text("extension: yaml\n")

    result = validate_yaml_path(yaml_file)

    assert result == yaml_file
