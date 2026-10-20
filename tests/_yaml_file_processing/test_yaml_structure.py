import pytest

from tabularepimdl._validators.yaml_file_processing.yaml_structure import validate_yaml_structure

from tabularepimdl._validators.yaml_file_processing.yaml_exceptions import (
    YAMLStructureError,
)


# Valid input tests
def test_valid_yaml_structure():
    """Accept a dictionary containing only the top-level 'rules' key."""
    data = {
        "rules": {
            "SimpleTransition": {"from_st": "S", "stochastic": False}
        }
    }

    result = validate_yaml_structure(data)

    # validate_yaml_structure should return the original dictionary.
    assert result == data
    assert result is data


# -------------------
# Invalid input tests
# -------------------
# empty document
def test_empty_yaml_document_raises_error():
    """Reject None, which represents an empty YAML document."""
    data = None

    with pytest.raises(YAMLStructureError,
                       match="YAML document is empty",
    ):
        validate_yaml_structure(data)


# top-level is not a dictionary
@pytest.mark.parametrize(
    "data",
    [
        "rules",          # String
        ["rules"],        # List
        123,              # number
        True,             # Boolean
    ],
)
def test_non_mapping_top_level_raises_error(data):
    """Reject top-level values that are not dictionaries."""
    with pytest.raises(YAMLStructureError,
                       match="Expected the top-level YAML document to be a mapping",
    ):
        validate_yaml_structure(data)


# number of top-level keys != 1
@pytest.mark.parametrize(
    "data",
    [
        {},  # No top-level keys
        {"rules": [], "metadata": {}},  # Two top-level keys

    ],
)
def test_incorrect_number_of_top_level_keys_raises_error(data):
    """Reject yamls that do not have exactly one top-level key."""
    with pytest.raises(YAMLStructureError,
                       match="must contain exactly one",
    ):
        validate_yaml_structure(data)


# required top-level key must be 'rules'
@pytest.mark.parametrize(
    "data",
    [
        {"metadata": {}},       # Wrong key
        {"config": []},         # Wrong key and value type
    ],
)
def test_missing_rules_key_raises_value_error(data):
    """Reject a single-key dictionary without the required 'rules' key."""
    with pytest.raises(ValueError,
                       match="Missing required top-level keys",
    ):
        validate_yaml_structure(data)