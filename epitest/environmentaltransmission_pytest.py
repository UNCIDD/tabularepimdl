"""
Unit test for EnvironmentalTransmission.py. Pytest package is used.
The EnvironmentalTransmission class models a natural environment disease transmission process.
The unit tests ensure that the class behaves as expected under various conditions, including both 
deterministic and stochastic scenarios. 
"""
import pytest
import pandas as pd
import numpy as np
from unittest import mock #used for mocking binomial distribution

import sys
sys.path.append('../')
from tabularepimdl.EnvironmentalTransmission import EnvironmentalTransmission

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts, their infection states and trait groups
    """
    data = {
    'InfState':pd.Categorical(["S"]*4+["I"],  categories=['S','I','R']),
    'HH_Number': list(range(4))+[1],
    'N': [4, 3, 2, 5, 1],
    }
    current_state = pd.DataFrame(data)
    current_state['N'] = current_state['N'].astype(np.float64) #converting column N to float type
    current_state = current_state.groupby(['HH_Number', 'InfState'], observed=True)['N'].sum().reset_index()
    return (current_state)

@pytest.fixture()
def environmental_transmission():
    """
    Initialize the EnvironmentalTransmission object with specified parameters.
    Returns: Initialized EnvironmentalTransmission object/instance.
    """
    return(EnvironmentalTransmission(beta=0.2/5, inf_col='InfState', trait_col='HH_Number'))

def test_initialization(environmental_transmission):
    """
    Test the initialization of the EnvironmentalTransmission object.
    Args: environmental_transmission object.
    """
    assert environmental_transmission.beta == 0.2/5
    assert environmental_transmission.inf_col == 'InfState'
    assert environmental_transmission.trait_col == 'HH_Number'
    assert environmental_transmission.s_st == 'S'
    assert environmental_transmission.i_st == 'I'
    assert environmental_transmission.inf_to == 'I'
    assert environmental_transmission.stochastic == False

def test_get_deltas_deterministic(environmental_transmission, dummy_state):
    """
    #Test the get_deltas method for the deterministic scenario.
    #Args: environmental_transmission object, dummy dataframe.
    """
    returned_rc = environmental_transmission.get_deltas(dummy_state)

    expected_rc = pd.DataFrame({
        'HH_Number':       [0, 1, 2, 3, 0, 1, 2, 3],
        'InfState':        ['S', 'S',  'S', 'S',  'I',  'I',  'I',  'I'],
        'N':               [-0.156842, -0.117632, -0.078421, -0.196053, 0.156842, 0.117632, 0.078421, 0.196053]
        })

    pd.testing.assert_frame_equal(returned_rc, expected_rc)

def test_get_deltas_stochastic(environmental_transmission, dummy_state):
    """
    Test the get_deltas method for the stochastic scenario.
    #Args: environmental_transmission object, dummy dataframe.
    """
    with mock.patch("numpy.random.binomial", return_value=2):
        returned_rc = environmental_transmission.get_deltas(dummy_state, stochastic=True)
        returned_rc['InfState'] = returned_rc['InfState'].astype('category') #convert InfState back to category given mock method changed its initial dtype

        expected_deltas = pd.DataFrame({
        'HH_Number': [0, 1, 2, 3 ],
        'InfState': pd.Categorical(['S',  'S', 'S',  'S'], categories=['I', 'S']),
        'N':        [-2, -2, -2, -2]
        })

        expected_deltas_add = pd.DataFrame({
        'HH_Number': [0, 1, 2, 3],
        'InfState': pd.Categorical(['I',  'I', 'I',  'I'], categories=['I', 'S']),
        'N':        [2, 2, 2, 2]
        })

        expected_rc = pd.concat([expected_deltas, expected_deltas_add])
        expected_rc = expected_rc.loc[expected_rc.N!=0].reset_index(drop=True)

        pd.testing.assert_frame_equal(returned_rc, expected_rc)

def test_to_yaml(environmental_transmission):
    """
    Test the to_yaml method of the WAIFWTransmission object.
    Args: environmental_transmission object.
    """
    returned_yaml = environmental_transmission.to_yaml()

    expected_yaml = {
            'tabularepimdl.EnvironmentalTransmission': {
                'beta': 0.2/5,
                'inf_col': 'InfState',
                'trait_col': 'HH_Number',
                's_st': 'S',
                'i_st': 'I',
                'inf_to':'I',
                'stochastic': False
            }
        }
    
    assert returned_yaml == expected_yaml
