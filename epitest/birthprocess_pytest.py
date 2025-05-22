"""
Unit test for BirthProcess.py. Pytest package is used.
The BirthProcess class models a birth process where new individuals are added to a population 
based on a specified birth rate, which can be deterministic or stochastic. 
The unit tests ensure that the class behaves as expected under various conditions, including both 
deterministic and stochastic scenarios. 
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from tabularepimdl.BirthProcess import BirthProcess

from unittest import mock #used for mocking poisson distribution

@pytest.fixture
def start_state_dict():
    """Fixture to create a dictionary representing the start state."""
    return {'age': 10, 'health': 'good'}

@pytest.fixture
def start_state_df(start_state_dict):
    """Fixture to create a DataFrame from the start state dictionary."""
    return pd.DataFrame([start_state_dict])

@pytest.fixture
def birthprocess_non_stochastic(start_state_dict):
    """Fixture to create a non-stochastic BirthProcess instance."""
    return BirthProcess(rate=0.1, start_state_sig=start_state_dict, stochastic=False)

@pytest.fixture
def birthprocess_stochastic(start_state_df):
    """Fixture to create a stochastic BirthProcess instance."""
    return BirthProcess(rate=0.1, start_state_sig=start_state_df, stochastic=True)

@pytest.fixture
def current_state():
    """Fixture to create a DataFrame representing the current state."""
    return pd.DataFrame({'N': [1000]})

def test_birthprocess_initialization(start_state_dict, start_state_df):
    """Test the initialization of BirthProcess instances.
    Args: start state dictionary and dataframe
    """
    bp = BirthProcess(rate=0.1, start_state_sig=start_state_dict, stochastic=False)
    assert bp.rate == 0.1
    assert bp.start_state_sig.equals(pd.DataFrame([start_state_dict]))
    assert bp.stochastic == False

    bp = BirthProcess(rate=0.1, start_state_sig=start_state_df, stochastic=True)
    assert bp.rate == 0.1
    assert bp.start_state_sig.equals(start_state_df)
    assert bp.stochastic == True

    with pytest.raises(ValueError) as excinfo:
        BirthProcess(rate=0.1, start_state_sig='invalid', stochastic=True)
    
def test_birthprocess_get_deltas_non_stochastic(birthprocess_non_stochastic, current_state):
    """
    Test the get_deltas method for non-stochastic BirthProcess.
    Args: birthprocess non-stochastic object and current state
    """
    deltas = birthprocess_non_stochastic.get_deltas(current_state, dt=1.0)
    expected_births = pd.DataFrame([{'age': 10, 'health': 'good', 'N': 1000 * (1 - np.exp(-0.1))}])
    pd.testing.assert_frame_equal(deltas, expected_births)

def test_birthprocess_get_deltas_stochastic(birthprocess_stochastic, current_state):
    """
    Test the get_deltas method for stochastic BirthProcess. Mock poisson distribution's sample value
    Args: birthprocess non-stochastic object and current state

    """
    with mock.patch('numpy.random.poisson', return_value=5):
        deltas = birthprocess_stochastic.get_deltas(current_state, dt=1.0)
        expected_births = pd.DataFrame([{'age': 10, 'health': 'good', 'N': 5}])
        pd.testing.assert_frame_equal(deltas, expected_births)

def test_birthprocess_to_yaml(birthprocess_non_stochastic, start_state_dict):
    """Test the to_yaml method for BirthProcess."""
    expected_yaml = {
        'tabularepimdl.BirthProcess': {
            'rate': 0.1,
            'start_state_sig': pd.DataFrame([start_state_dict]),
            'stochastic': False
        }
    }
    result_yaml = birthprocess_non_stochastic.to_yaml()
    assert result_yaml['tabularepimdl.BirthProcess']['rate'] == expected_yaml['tabularepimdl.BirthProcess']['rate']
    assert result_yaml['tabularepimdl.BirthProcess']['stochastic'] == expected_yaml['tabularepimdl.BirthProcess']['stochastic']
    
    #Extract the DataFrame from the result_yaml and expected_yaml for comparison
    pd.testing.assert_frame_equal(result_yaml['tabularepimdl.BirthProcess']['start_state_sig'], expected_yaml['tabularepimdl.BirthProcess']['start_state_sig'])
