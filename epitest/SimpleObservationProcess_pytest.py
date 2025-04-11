"""
Unit test for SimpleObservationProcess.py. Pytest package is used.
The SimpleObservationProcess class models a simple generic observation process 
where people from a particular state are observed to move into another state at some constant rate.
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
from tabularepimdl.SimpleObservationProcess import SimpleObservationProcess

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts, their infection states and hospitalization states
    """
    data = {
        'N': [10, 20, 30, 40, 50],
        'Infection_State': ['S', 'I', 'S', 'I', 'S'], #links to source_col
        'Hosp': ['U', 'U', 'P', 'U', 'I'] #links to obs_col
    }
    return(pd.DataFrame(data))

@pytest.fixture
def simple_observation():
    """
    Initialize the SimpleObservationProcess object with specified parameters.
    Returns: Initialized SimpleObservationProcess object/instance.
    """
    #this initialization prepares the rule with infected individuals and they will be unobserved people for the following use.
    return(SimpleObservationProcess(source_col='Infection_State', source_state='I', obs_col='Hosp', rate=0.05))

def test_initialization(simple_observation):
    """
    Test the initialization of the SimpObservationProcess object.
    Args: simple_observation object.
    """
    assert simple_observation.source_col == 'Infection_State'
    assert simple_observation.source_state == 'I'
    assert simple_observation.obs_col == 'Hosp'
    assert simple_observation.rate == 0.05
    assert simple_observation.unobs_state == 'U'
    assert simple_observation.incobs_state == 'I'
    assert simple_observation.prevobs_state == 'P'
    assert simple_observation.stochastic == False

def test_deltas_calculation(simple_observation, dummy_state):
    """
    Test the number of incident observations (deltas).
    Args: simple_observation object and dummy dataframe.
    """
    #infected but unobserved individuals
    delta_incobs = dummy_state.loc[(dummy_state[simple_observation.source_col]==simple_observation.source_state) & (dummy_state[simple_observation.obs_col]==simple_observation.unobs_state)]

    expected_delta = pd.DataFrame({
        'N': [20, 40],
        'Infection_State': ['I', 'I'], #source_col is I
        'Hosp': ['U', 'U'] #obs_col is U
    })

    pd.testing.assert_frame_equal(delta_incobs.reset_index(drop=True), expected_delta.reset_index(drop=True))

def test_get_deltas_deterministic(simple_observation, dummy_state):
    """
    Test the get_deltas method for the deterministic scenario.
    Args: simple_observation object and dummy dataframe.
    """
    returned_deltas_and_tmp = simple_observation.get_deltas(current_state=dummy_state, stochastic=False)

    expected_deltas_incobs_of_subtractions = pd.DataFrame({
        'N': [-20 * (1-np.exp(-1.0*0.05)), -40 * (1-np.exp(-1.0*0.05))],
        'Infection_State': ['I', 'I'], #source_col is I
        'Hosp': ['U', 'U'] #obs_col is U
    })

    expected_tmp_of_additions = pd.DataFrame({
        'N': [20 * (1-np.exp(-1.0*0.05)), 40 * (1-np.exp(-1.0*0.05))],
        'Infection_State': ['I', 'I'], #source_col is I
        'Hosp': ['I', 'I'] #obs_col is U
    })

    expected_delta_toprev = pd.DataFrame({
        'N': [50],
        'Infection_State': ['S'], #source_col is I
        'Hosp': ['P'] #obs_col is U
    })

    expected_tmp2 = pd.DataFrame({
        'N': [-50],
        'Infection_State': ['S'], #source_col is I
        'Hosp': ['I'] #obs_col is U
    })

    expected_result = pd.concat([expected_deltas_incobs_of_subtractions, expected_tmp_of_additions, expected_tmp2, expected_delta_toprev])
    pd.testing.assert_frame_equal(returned_deltas_and_tmp.reset_index(drop=True), expected_result.reset_index(drop=True))

def test_get_deltas_stochastic(simple_observation, dummy_state):
        """
        Test the get_deltas method for the stochastic scenario.
        Args: simple_observation object and dummy dataframe.
        """
        with mock.patch("numpy.random.binomial", return_value=10):
            returned_deltas_and_tmp = simple_observation.get_deltas(current_state=dummy_state, stochastic=True)
            
            expected_deltas_incobs_of_subtractions = pd.DataFrame({
            'N': [-10, -10],
            'Infection_State': ['I', 'I'], #source_col is I
            'Hosp': ['U', 'U'] #obs_col is U
            })

            expected_tmp_of_additions = pd.DataFrame({
                'N': [10, 10],
                'Infection_State': ['I', 'I'], #source_col is I
                'Hosp': ['I', 'I'] #obs_col is U
            })

            expected_delta_toprev = pd.DataFrame({
                'N': [50],
                'Infection_State': ['S'], #source_col is I
                'Hosp': ['P'] #obs_col is U
            })

            expected_tmp2 = pd.DataFrame({
                'N': [-50],
                'Infection_State': ['S'], #source_col is I
                'Hosp': ['I'] #obs_col is U
            })

            expected_result = pd.concat([expected_deltas_incobs_of_subtractions, expected_tmp_of_additions, expected_tmp2, expected_delta_toprev])
            pd.testing.assert_frame_equal(returned_deltas_and_tmp.reset_index(drop=True), expected_result.reset_index(drop=True))

def test_to_yaml(simple_observation):
    """
    Test the to_yaml method of the SimpleTransition object.
    Args: simple_transition object.
    """
    expected_yaml = {
         'tabularepimdl.SimpleObservationProcess': {
                'source_col': 'Infection_State',
                'source_state': 'I',
                'obs_col': 'Hosp',
                'rate': 0.05,
                'unobs_state': 'U',
                'incobs_state': 'I',
                'prevobs_state': 'P',
                'stochastic': False
            }
        }
    
    assert simple_observation.to_yaml() == expected_yaml