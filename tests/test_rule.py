"""
Unit test for Rule.py. Pytest package is used.
The Rule class defines a class that represents a transition rule.
The unit tests ensure that the class behaves as expected.
"""

from pathlib import Path

import pytest
import yaml

from my_module import MyRule


@pytest.fixture
def rule_yaml_setUp():
    """
    Load a yaml file from the tests directory (independent of the current working directory).
    Returns: content from the loaded yaml file.
    """
    # Load the YAML file
    yaml_path = Path(__file__).parent / "yaml_input.yml"
    with open(yaml_path) as file:
        rule_yaml = yaml.safe_load(file)
    return rule_yaml


@pytest.fixture
def myrule(rule_yaml_setUp):
    """
    Initialize the Rule object by using from_yaml classmethod with loaded yaml content.
    Returns: Initialized Rule object/instance.
    """
    rule = MyRule.from_yaml(rule_yaml_setUp)
    return rule


def test_from_yaml(myrule):
    """
    Test the from_yaml() defined in Rule module.
    Args: myrule object.
    """
    assert not myrule.stochastic
    assert myrule.rate == 0.1
    assert myrule.column == "InfState"
    assert myrule.from_st == "S"
    assert myrule.to_st == "I"


def test_to_dict(myrule):
    """
    Test the to_dict() defined in SimpleTransition_Vec_Encode, reached via Rule.from_yaml()'s dispatch.
    Args: myrule object.
    """
    expected_dict = {
        "tabularepimdl.SimpleTransition_Vec_Encode": {
            "column": "InfState",
            "from_st": "S",
            "to_st": "I",
            "rate": 0.1,
            "stochastic": False,
            "column_categories": ["placeholder"],
            "infstate_compartments": ["S", "I", "R"],
        }
    }
    returned_dict = myrule.to_dict()
    assert returned_dict == expected_dict
