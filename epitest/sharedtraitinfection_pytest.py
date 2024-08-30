"""
Unit test for SharedTraitInfection.py. Pytest package is used.
The SharedTraitInfection.py class models a shared trait infection process
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
from tabularepimdl.SharedTraitInfection import SharedTraitInfection

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts, their infection states and trait groups
    """
    data = {
        'N':               [10,   20,   30,   40,   50,   60],
        'Infection_State': ['I',  'I',  'S',  'S',  'I',  'S'],
        'Trait':           ['T1', 'T2', 'T1', 'T2', 'T1', 'T2']
    }
    return (pd.DataFrame(data))

@pytest.fixture()
def sharedtrait_infection():
    """
    Initialize the SharedTraitInfection object with specified parameters.
    Returns: Initialized SharedTraitInfection object/instance.
    """
    return(SharedTraitInfection(in_beta=0.02, out_beta=0.04, inf_col='Infection_State', trait_col='Trait'))

def test_initialization(sharedtrait_infection):
    """
    Test the initialization of the SharedTraitInfection object.
    Args: sharedtrait_infection object.
    """
    assert sharedtrait_infection.in_beta == 0.02
    assert sharedtrait_infection.out_beta == 0.04
    assert sharedtrait_infection.inf_col == 'Infection_State'
    assert sharedtrait_infection.trait_col == 'Trait'
    assert sharedtrait_infection.s_st == 'S'
    assert sharedtrait_infection.i_st == 'I'
    assert sharedtrait_infection.inf_to == 'I'
    assert sharedtrait_infection.stochastic == False

def test_detlas_calculation(sharedtrait_infection, dummy_state):
    """
    Test the number of incident observations (deltas).
    Args: sharedtrait_infection object and dummy dataframe.
    """
    returned_deltas = dummy_state.loc[dummy_state[sharedtrait_infection.inf_col]==sharedtrait_infection.s_st].copy()

    expected_deltas = pd.DataFrame({
        'N':               [30,   40,   60],
        'Infection_State': ['S',  'S',  'S'],
        'Trait':           ['T1', 'T2', 'T2']
    })

    pd.testing.assert_frame_equal(returned_deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True))

def test_inI_outI_prI_deltas(sharedtrait_infection, dummy_state):
    """
    Test the values of parameters inI, outI and prI, and N population values in deltas.
    Args: sharedtrait_infection object and dummy dataframe.
    """
    dummy_state['N'] = dummy_state['N'].astype(np.float64) # converting column N to float type
    
    # Create deltas DataFrame for rows where Infection_State is 'S'
    deltas = dummy_state.loc[dummy_state[sharedtrait_infection.inf_col] == sharedtrait_infection.s_st].copy()
    deltas_add = deltas.copy()

    # Precompute inI and outI using groupby and sum
    inI  = dummy_state[dummy_state[sharedtrait_infection.inf_col] == sharedtrait_infection.i_st].groupby(sharedtrait_infection.trait_col)['N'].sum() #filter all records where infection state is I and sum individuals based on their trait groups
    outI = dummy_state[dummy_state[sharedtrait_infection.inf_col] == sharedtrait_infection.i_st].groupby(sharedtrait_infection.trait_col)['N'].sum().sum() - inI #sum all infected individuals regardless of trait groups, then minus individuals in inI to get individuals that are not in each inI's trait groups

    # Map inI and outI values to the deltas DataFrame
    deltas['inI']  = deltas[sharedtrait_infection.trait_col].map(inI).fillna(0) #map inI's each trait group's value to each correponding deltas' trait group, fill NA values with 0
    deltas['outI'] = deltas[sharedtrait_infection.trait_col].map(outI).fillna(0) #map outI's each trait group's value to each correponding deltas' trait group, fill NA values with 0

    # Vectorized calculation of prI
    deltas['prI'] = 1 - np.power(np.exp(-1.0 * sharedtrait_infection.in_beta), deltas['inI']) * np.power(np.exp(-1.0 * sharedtrait_infection.out_beta), deltas['outI'])

    # Update N values based on prI
    deltas['N'] = deltas['N'] * deltas['prI']

    expected_deltas_with_inI_outI_prI = pd.DataFrame({
        'N':               [25.939941502901636, 37.56759749499129, 56.35139624248693],
        'Infection_State': ['S',  'S',  'S'],
        'Trait':           ['T1', 'T2', 'T2'],
        'inI':             [60.0,  20.0, 20.0],
        'outI':            [20.0,  60.0, 60.0],
        'prI':             [0.8646647167633879, 0.9391899373747822, 0.9391899373747822]

    })

    pd.testing.assert_frame_equal(deltas.reset_index(drop=True), expected_deltas_with_inI_outI_prI.reset_index(drop=True))


def test_get_deltas_deterministic(sharedtrait_infection, dummy_state):
    """
    #Test the get_deltas method for the deterministic scenario.
    #Args: sharedtrait_infection object, dummy dataframe.
    """
    returned_rc = sharedtrait_infection.get_deltas(dummy_state)

    deltas_with_temp_parameters = pd.DataFrame({
        'N':               [25.939941502901636, 37.56759749499129, 56.35139624248693],
        'Infection_State': ['S',  'S',  'S'],
        'Trait':           ['T1', 'T2', 'T2'],
        'inI':             [60.0,  20.0, 20.0],
        'outI':            [20.0,  60.0, 60.0],
        'prI':             [0.8646647167633879, 0.9391899373747822, 0.9391899373747822]

    })

    #drop temporary columns inI, outI and prI
    deltas_with_temp_parameters.drop(['inI', 'outI', 'prI'], axis=1, inplace=True)

    # Update deltas and deltas_add DataFrames
    deltas_add = deltas_with_temp_parameters.copy()
    deltas = deltas_with_temp_parameters.assign(N=-deltas_with_temp_parameters.N)
        
    deltas_add['Infection_State']='I'

    expected_rc = pd.concat([deltas, deltas_add])
    expected_rc = expected_rc.loc[expected_rc.N!=0].reset_index(drop=True)

    pd.testing.assert_frame_equal(returned_rc, expected_rc)

def test_get_deltas_stochastic(sharedtrait_infection, dummy_state):
    """
    Test the get_deltas method for the stochastic scenario.
    #Args: sharedtrait_infection object, dummy dataframe.
    """
    with mock.patch("numpy.random.binomial", return_value=20):
        returned_rc = sharedtrait_infection.get_deltas(dummy_state, stochastic=True)
            
        expected_deltas = pd.DataFrame({
        'N':               [-20, -20, -20],
        'Infection_State': ['S',  'S',  'S'],
        'Trait':           ['T1', 'T2', 'T2'],
        })

        expected_deltas_add = pd.DataFrame({
        'N':               [20, 20, 20],
        'Infection_State': ['I',  'I',  'I'],
        'Trait':           ['T1', 'T2', 'T2'],
        })

        expected_rc = pd.concat([expected_deltas, expected_deltas_add])
        expected_rc = expected_rc.loc[expected_rc.N!=0].reset_index(drop=True)

        pd.testing.assert_frame_equal(returned_rc, expected_rc)

def test_to_yaml(sharedtrait_infection):
    """
    Test the to_yaml method of the WAIFWTransmission object.
    Args: waifw_transmission object.
    """
    returned_yaml = sharedtrait_infection.to_yaml()

    expected_yaml = {
            'tabularepimdl.SharedTraitInfection': {
                'in_beta': 0.02,
                'out_beta': 0.04,
                'inf_col': 'Infection_State',
                'trait_col': 'Trait',
                's_st': 'S',
                'i_st': 'I',
                'inf_to':'I',
                'stochastic': False
            }
        }
    
    assert returned_yaml == expected_yaml