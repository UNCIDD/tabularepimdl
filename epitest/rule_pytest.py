"""
Unit test for Rule.py. Pytest package is used.
The Rule class defines a class that represents a transition rule. 
The unit tests ensure that the class behaves as expected.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
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
    assert myrule.stochastic == False
    assert myrule.rate == 0.1
    pd.testing.assert_frame_equal(myrule.start_state_sig, pd.DataFrame([{'age': 10, 'health': 'good'}]))

def test_to_yaml(myrule):
    """
    Test the to_yaml() defined in MyRule module.
    Args: myrule object.
    """
    expected_yaml = {
            "tabularepimdl.BirthProcess": {
                "start_state_sig": {'age': 10, 'health': 'good'},
                "rate": 0.1,
                'stochastic': False
            }
        }
    returned_to_yaml = myrule.to_yaml()
    assert returned_to_yaml == expected_yaml
    

