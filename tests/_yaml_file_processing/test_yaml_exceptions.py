import pytest

from tabularepimdl._validators.yaml_file_processing.yaml_exceptions import DuplicateYAMLKeyError, YAMLParseError


def test_duplicate_yaml_key_stores_attributes():
    """Test that the attributes are stored correctly in DuplicateYAMLKeyError."""
    error = DuplicateYAMLKeyError(
        filename="epidemic.yaml",
        key="rate",
        path="rules.BirthProcess_Vec_Encode",
        original_line=4,
        original_column=3,
        duplicate_line=8,
        duplicate_column=3,
    )

    assert error.filename == "epidemic.yaml"
    assert error.key == "rate"
    assert error.path == "rules.BirthProcess_Vec_Encode"
    assert error.original_line == 4
    assert error.original_column == 3
    assert error.duplicate_line == 8
    assert error.duplicate_column == 3


def test_duplicate_key_error_can_be_caught_as_parse_error():
    """DuplicateYAMLKeyError inherits from YAMLParseError, 
    an application should catch it using the parent exception YAMLParseError"""
    with pytest.raises(YAMLParseError) as exc_info:
        raise DuplicateYAMLKeyError(
            filename="epidemic.yaml",
            key="rate",
            path="rules.BirthProcess_Vec_Encode",
            original_line=4,
            original_column=3,
            duplicate_line=8,
            duplicate_column=3,
        )

    assert isinstance(exc_info.value, DuplicateYAMLKeyError)
    assert exc_info.value.key == "rate"