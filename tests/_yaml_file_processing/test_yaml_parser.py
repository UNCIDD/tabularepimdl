
import pytest

from tabularepimdl._validators.yaml_file_processing.yaml_exceptions import DuplicateYAMLKeyError, YAMLParseError
from tabularepimdl._validators.yaml_file_processing.yaml_parser import parse_yaml


# 1. Successful YAML parsing
def test_parse_simple_yaml() -> None:
    """A simple nested YAML mapping should become a Python dictionary."""

    content = """
    SimpleTransition:
        column: InfState
        from_st: S
        to_st: I
        compartments:
            - S
            - I
            - R
    """

    # Parse the YAML text.
    result = parse_yaml(content)

    # Check the resulting Python object.
    assert result == {
    "SimpleTransition": {
            "column": "InfState",
            "from_st": "S",
            "to_st": "I",
            "compartments": ["S", "I", "R"]
        }
    }


def test_parse_empty_yaml() -> None:
    """An empty YAML document should return None."""

    result = parse_yaml("")

    assert result is None


# 2. Duplicate-key detection
def test_duplicate_top_level_key() -> None:
    """Duplicate keys at the root should raise a custom exception."""

    content = """
    SimpleTransition:
        column: InfState
        from_st: S
        to_st: I
        rate: 0.1
        rate: 0.2
    """

    with pytest.raises(DuplicateYAMLKeyError) as exc_info:
        parse_yaml(content, filename="config.yaml")

    # The exception message should identify the file and key.
    error_message = str(exc_info.value)

    assert "config.yaml" in error_message
    assert "rate" in error_message


def test_duplicate_key_inside_list() -> None:
    """
    Duplicate keys inside a list item should include its index.
    The error path should identify the correct list index.
    """

    content = """
    rules:
      - a_rule: first
        rate: 0.1
      - b_rule: second
        rate: 0.2
        rate: 0.3
    """

    with pytest.raises(DuplicateYAMLKeyError) as exc_info:
        parse_yaml(content, filename="rules.yaml")

    error_message = str(exc_info.value)

    assert "rules.yaml" in error_message
    assert "rules[1].rate" in error_message


# 3. Error location details
def test_duplicate_key_reports_line_numbers() -> None:
    """The error should identify the original and duplicate lines."""

    content = "rate: 0.1\nrate: 0.2\n"

    with pytest.raises(DuplicateYAMLKeyError) as exc_info:
        parse_yaml(content, filename="config.yaml")

    error = exc_info.value

    # These assertions assume the custom exception exposes
    # attributes matching the constructor arguments in yaml_parser.py.
    assert error.original_line == 1
    assert error.duplicate_line == 2

    # Both keys start in column 1.
    assert error.original_column == 1
    assert error.duplicate_column == 1


def test_duplicate_key_inside__different_list() -> None:
    """Duplicate keys inside different lists should NOT raise an Error."""

    content = """
    rules:
      - a_rule: first
        rate: 0.1
      - b_rule: second
        rate: 0.2
    """

    # Parse the YAML text.
    result = parse_yaml(content)
    
    # Check the resulting Python object.
    assert result == {
        "rules": [
            {"a_rule": "first", "rate": 0.1},
            {"b_rule": "second", "rate": 0.2},
        ]
    }



def test_invalid_yaml_syntax() -> None:
    """Malformed YAML should raise YAMLParseError."""

    # The indentation makes this mapping invalid.
    content = """
    from_st: S
      to_st: I
    """

    with pytest.raises(
        YAMLParseError,
        match="Invalid YAML syntax",
    ) as exc_info:
        parse_yaml(content, filename="invalid.yaml")

    # Verify that the custom filename appears in the message.
    assert "invalid.yaml" in str(exc_info.value)