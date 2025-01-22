"""
Unit test for Rule.py. Pytest package is used.
The Rule class defines a class that represents a transition rule. 
The unit tests ensure that the class behaves as expected.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.append('../')
from my_module import MyRule

@pytest.fixture
def rule_yaml_setUp():
        """
        Load a yaml file from local working directory.
        Returns: content from the loaded yaml file.
        """
        # Load the YAML file
        with open("yaml_input.yml", "r") as file:
            rule_yaml = yaml.safe_load(file)
        return(rule_yaml)
        
@pytest.fixture
def myrule(rule_yaml_setUp):
     """
     Initialize the Rule object by using from_yaml classmethod with loaded yaml content.
     Returns: Initialized Rule object/instance.
     """
     rule = MyRule.from_yaml(rule_yaml_setUp)
     return(rule)

def test_from_yaml(myrule):
    """
    Test the from_yaml() defined in Rule module.
    Args: myrule object.
    """
    assert isinstance(myrule, MyRule)
    assert myrule.param1 == 1.0
    assert myrule.param2 == 2.0

def test_get_deltas(myrule):
    """
    Test the get_deltas() defined in MyRule module.
    Args: myrule object.
    """
    current_state = 10
    dt = 1.0
    expected_deltas = {"state": 11.0}
    returned_deltas = myrule.get_deltas(current_state, dt, stochastic=False)
    assert returned_deltas == expected_deltas
    
def test_to_yaml(myrule):
    """
    Test the to_yaml() defined in MyRule module.
    Args: myrule object.
    """
    expected_yaml = {
            "my_module.MyRule": {
                "param1": 1.0,
                "param2": 2.0
            }
        }
    returned_to_yaml = myrule.to_yaml()
    assert returned_to_yaml == expected_yaml
    

