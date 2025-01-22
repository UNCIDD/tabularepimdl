"""
Unit test for SimpleTransition.py. Pytest package is used.
The SimpleTransition class models a simple transition process where individuals in a population 
can transition from a susceptible state to an infectious state based on a transmission probability.
The unit tests ensure that the class behaves as expected under various conditions, including both 
deterministic and stochastic scenarios. 
"""

import pytest
import pandas as pd
import numpy as np
from unittest import mock #used for mocking binomial distribution

import os
import sys
sys.path.append('../')
from tabularepimdl.SimpleTransition import SimpleTransition

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts and their infection states.
    """
    data = {
        "N": [10, 20, 30, 40], #population counts
        "Infection_State": ['S', 'I', 'S', 'I'] #infection state

    }
    return pd.DataFrame(data)

@pytest.fixture
def simple_transition():
    """
    Initialize the SimpleTransition object with specified parameters.
    Returns: Initialized SimpleTransition object/instance.
    """
    return SimpleTransition(column='Infection_State', from_st='S', to_st='I', rate=0.3)

def test_initialization(simple_transition):
    """
    Test the initialization of the SimpleTransition object.
    Args: simple_tansition object.
    """
    assert simple_transition.column == 'Infection_State'
    assert simple_transition.from_st == 'S'
    assert simple_transition.to_st == 'I'
    assert simple_transition.rate == 0.3
    assert simple_transition.stochastic == False

def test_deltas_calculation(simple_transition, dummy_state):
    """
    Test the number of from_st individuals (deltas).
    Args: simple_transition object and dummy dataframe.
    """
    deltas = dummy_state.loc[dummy_state[simple_transition.column]==simple_transition.from_st]
    expected_deltas = pd.DataFrame({
        'N': [10,30],
        'Infection_State':['S', 'S']
    })
    pd.testing.assert_frame_equal(deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True)) #reset df index to default integer index

def test_get_deltas_deterministic(simple_transition, dummy_state):
    """
    Test the get_deltas method for the deterministic scenario.
    Args: simple_transition object and dummy dataframe.
    """
    returned_deltas_and_tmp = simple_transition.get_deltas(current_state=dummy_state, stochastic=False)

    expected_deltas_of_subtractions = pd.DataFrame({
        'N': [-10 * (1 - np.exp(-1.0*0.3)), -30 * (1 - np.exp(-1.0*0.3))],
        'Infection_State': ['S', 'S']
    })

    expected_tmp_of_additions = pd.DataFrame({
        'N': [10 * (1 - np.exp(-1.0*0.3)), 30 * (1 - np.exp(-1.0*0.3))],
        'Infection_State': ['I', 'I']
    })

    expected_result = pd.concat([expected_deltas_of_subtractions, expected_tmp_of_additions])
    pd.testing.assert_frame_equal(returned_deltas_and_tmp.reset_index(drop=True), expected_result.reset_index(drop=True))

def test_get_deltas_stochastic(simple_transition, dummy_state):
    """
    Test the get_deltas method for the stochastic scenario. Mock binomial distribution's sample value
    Args: simple_transition object and dummy dataframe.
    """
    
    with mock.patch("numpy.random.binomial", return_value=10):
        returned_deltas_and_tmp = simple_transition.get_deltas(current_state=dummy_state, stochastic=True)

        expected_deltas_of_subtractions = pd.DataFrame({
            'N': [-10, -10],
            'Infection_State': ['S', 'S']
        })

        expected_tmp_of_additions = pd.DataFrame({
            'N': [10, 10],
            'Infection_State': ['I', 'I']
        })

        expected_result = pd.concat([expected_deltas_of_subtractions, expected_tmp_of_additions])
        pd.testing.assert_frame_equal(returned_deltas_and_tmp.reset_index(drop=True), expected_result.reset_index(drop=True))

def test_str(simple_transition):
    """
    Test the __str__ method of SimpleTransition object.
    Args: simple_transition object.
    """
    expected_str = simple_transition.__str__()
    assert expected_str == "S --> I at rate 0.3"


def test_to_yaml(simple_transition):
    """
    Test the to_yaml method of the SimpleTransition object.
    Args: simple_transition object.
    """
    expected_yaml = {
            'tabularepimdl.SimpleTransition': {
                'column': 'Infection_State',
                'from_st': 'S',
                'to_st': 'I',
                'rate': 0.3, 
                'stochastic': False
            }
        }
    
    assert simple_transition.to_yaml() == expected_yaml