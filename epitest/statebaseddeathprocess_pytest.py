"""
Unit test for StateBasedDeathProcess.py. Pytest package is used.
The StateBasedDeathProcess class models a death process that takes people out of a state
defined by one or more state columns at some rate.
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
from tabularepimdl.StateBasedDeathProcess import StateBasedDeathProcess

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts, their infection states and hospitalization states
    """
    data = {
        'N': [10, 20, 30, 40, 50],
        'Infection_State': ['S1', 'S2', 'I1', 'R1', 'R2'], #column1
        'Hosp':            ['I1', 'I2', 'P1', 'U1', 'I3'] #column2
    }
    return (pd.DataFrame(data))

@pytest.fixture()
def statebased_deathprocess():
    """
    Initialize the StateBasedDeathProcess object with specified parameters.
    Returns: Initialized StateBasedDeathProcess object/instance.
    """
    return(StateBasedDeathProcess(columns=['Infection_State', 'Hosp', 'Infection_State', 'Hosp'], states=['S2', 'U1', 'I1', 'I3'], rate=0.05))

def test_initialization(statebased_deathprocess):
    """
    Test the initialization of the StateBasedDeathProcess object.
    Args: statebased_deathprocess object.
    """
    assert statebased_deathprocess.columns == ['Infection_State', 'Hosp', 'Infection_State', 'Hosp'] #a list of different headers
    assert statebased_deathprocess.states == ['S2', 'U1', 'I1', 'I3'] #a list of different states used in each header
    assert statebased_deathprocess.rate == 0.05
    assert statebased_deathprocess.stochastic == False

def test_deltas_calculation(statebased_deathprocess, dummy_state):
    """
    Test the number of incident observations (deltas).
    Args: statebased_deathprocess object and dummy dataframe.
    """
    delta_initial = dummy_state.copy()
    returned_deltas = pd.DataFrame()
    for column, state in zip(statebased_deathprocess.columns, statebased_deathprocess.states):
        returned_deltas = pd.concat([returned_deltas, delta_initial.loc[delta_initial[column]==state]])
        
    
    expected_deltas = pd.DataFrame({
        'N':               [20,    40,   30,   50],
        'Infection_State': ['S2', 'R1', 'I1', 'R2'],
        'Hosp':            ['I2', 'U1', 'P1', 'I3']
    })
    
    pd.testing.assert_frame_equal(returned_deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True))

def test_get_deltas_deterministic(statebased_deathprocess, dummy_state):
    """
    Test the get_deltas method for the deterministic scenario.
    Args: statebased_deathprocess object and dummy dataframe.
    """
    returned_deltas = statebased_deathprocess.get_deltas(dummy_state)

    expected_deltas = pd.DataFrame({
    'N':               [-20*(1-np.exp(-1.0*0.05)), -40*(1-np.exp(-1.0*0.05)), -30*(1-np.exp(-1.0*0.05)), -50*(1-np.exp(-1.0*0.05))],
    'Infection_State': ['S2', 'R1', 'I1', 'R2'],
    'Hosp':            ['I2', 'U1', 'P1', 'I3']
    })

    pd.testing.assert_frame_equal(returned_deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True))

def test_get_deltas_stochastic(statebased_deathprocess, dummy_state):
        """
        Test the get_deltas method for the stochastic scenario.
        Args: statebased_deathprocess object and dummy dataframe.
        """
        with mock.patch("numpy.random.binomial", return_value=20):
            returned_deltas = statebased_deathprocess.get_deltas(dummy_state, stochastic=True)
             
            expected_deltas = pd.DataFrame({
                 'N':               [-20, -20, -20, -20],
                 'Infection_State': ['S2', 'R1', 'I1', 'R2'],
                 'Hosp':            ['I2', 'U1', 'P1', 'I3']
            })

            pd.testing.assert_frame_equal(returned_deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True))

def test_to_yaml(statebased_deathprocess):
    """
    Test the to_yaml method of the StateBasedDeathProcess object.
    Args: statebased_deathprocess object.
    """
    expected_yaml = {
        'tabularepimdl.StateBasedDeathProcess' : {
            'columns' : ['Infection_State', 'Hosp', 'Infection_State', 'Hosp'],
            'states' : ['S2', 'U1', 'I1', 'I3'],
            'rate' : 0.05,
            'stochastic' : False
            }
    }

    assert statebased_deathprocess.to_yaml() == expected_yaml
    