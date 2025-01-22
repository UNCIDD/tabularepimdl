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
    np.random.seed(3)
    data = {
    'InfState':pd.Categorical(["S"]*5+["I"]*2 +["I", "S"] +["R"],  categories=['S','I','R']),
    'HH_Number': list(range(5))      +[3, 4]  +[3, 1]     +[3], 
    'N': list(np.random.poisson(2,5))+[5, 5]  +[2, 3]     +[1], #lambda in poisson is 2 since avg.household size is 2, the additional 1 is for HH=499
    }
    current_state = pd.DataFrame(data)
    current_state['N'] = current_state['N'].astype(np.float64) #converting column N to float type
    current_state = current_state.groupby(['HH_Number', 'InfState'], observed=True)['N'].sum().reset_index()
    return (current_state)

@pytest.fixture()
def sharedtrait_infection():
    """
    Initialize the SharedTraitInfection object with specified parameters.
    Returns: Initialized SharedTraitInfection object/instance.
    """
    return(SharedTraitInfection(in_beta=0.2/5, out_beta=0.002/5, inf_col='InfState', trait_col='HH_Number'))

def test_initialization(sharedtrait_infection):
    """
    Test the initialization of the SharedTraitInfection object.
    Args: sharedtrait_infection object.
    """
    assert sharedtrait_infection.in_beta == 0.2/5
    assert sharedtrait_infection.out_beta == 0.002/5
    assert sharedtrait_infection.inf_col == 'InfState'
    assert sharedtrait_infection.trait_col == 'HH_Number'
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
        'HH_Number': [0, 1, 2, 3, 4],
        'InfState':  pd.Categorical(['S', 'S', 'S', 'S', 'S'], categories=['S', 'I', 'R']),
        'N':         [2.0, 6.0, 1.0, 1.0, 2.0]
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
    #deltas_add = deltas.copy()

    #infected people only
    i_only = dummy_state.loc[dummy_state[sharedtrait_infection.inf_col]==sharedtrait_infection.i_st].copy(deep=True)
    sum_i = i_only['N'].sum()

    #map HH_Number between infected people only and deltas
    hh_n_map = i_only.set_index('HH_Number')['N'] #set HH_number as index of a series, N is the value of the index
    
    # Precompute inI and outI
    deltas['inI'] = deltas['HH_Number'].map(hh_n_map).fillna(0)
    deltas['outI'] = sum_i - deltas['inI']
    
    # Vectorized calculation of prI
    deltas['prI'] = 1 - np.power(np.exp(-1.0 * sharedtrait_infection.in_beta), deltas['inI']) * np.power(np.exp(-1.0 * sharedtrait_infection.out_beta), deltas['outI'])

    # Update N values based on prI
    deltas['N'] = deltas['N'] * deltas['prI']

    expected_deltas_with_inI_outI_prI = pd.DataFrame({
        'HH_Number':       [0, 1, 2, 3, 4],
        'InfState':        pd.Categorical(['S',  'S',  'S', 'S',  'S'], categories=['S', 'I', 'R']),
        'N':               [0.009576996819805172, 0.028730990459415517, 0.004788498409902586, 0.24572631546691115, 0.36711697319890346],
        'inI':             [0.0, 0.0, 0.0, 7.0, 5.0],
        'outI':            [12.0, 12.0, 12.0, 5.0, 7.0],
        'prI':             [0.004788498409902586, 0.004788498409902586, 0.004788498409902586, 0.24572631546691115, 0.18355848659945173]
    })

    pd.testing.assert_frame_equal(deltas.reset_index(drop=True), expected_deltas_with_inI_outI_prI.reset_index(drop=True))

def test_get_deltas_deterministic(sharedtrait_infection, dummy_state):
    """
    #Test the get_deltas method for the deterministic scenario.
    #Args: sharedtrait_infection object, dummy dataframe.
    """
    returned_rc = sharedtrait_infection.get_deltas(dummy_state)

    deltas_with_temp_parameters = pd.DataFrame({
        'HH_Number':       [0, 1, 2, 3, 4],
        'InfState':        pd.Categorical(['S',  'S',  'S', 'S',  'S'], categories=['S', 'I', 'R']),
        'N':               [0.009576996819805172, 0.028730990459415517, 0.004788498409902586, 0.24572631546691115, 0.36711697319890346],
        'inI':             [0.0, 0.0, 0.0, 7.0, 5.0],
        'outI':            [12.0, 12.0, 12.0, 5.0, 7.0],
        'prI':             [0.004788498409902586, 0.004788498409902586, 0.004788498409902586, 0.24572631546691115, 0.18355848659945173]

    })

    #drop temporary columns inI, outI and prI
    deltas_with_temp_parameters.drop(['inI', 'outI', 'prI'], axis=1, inplace=True)

    # Update deltas and deltas_add DataFrames
    deltas_add = deltas_with_temp_parameters.copy()
    deltas = deltas_with_temp_parameters.assign(N=-deltas_with_temp_parameters.N)
        
    deltas_add['InfState']='I'

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
        returned_rc['InfState'] = returned_rc['InfState'].astype('category') #convert InfState back to category given mock method changed its initial dtype

        expected_deltas = pd.DataFrame({
        'HH_Number': [0, 1, 2, 3, 4],
        'InfState': pd.Categorical(['S',  'S',  'S', 'S',  'S'], categories=['I', 'S']),
        'N':        [-20, -20, -20, -20, -20]
        })

        expected_deltas_add = pd.DataFrame({
        'HH_Number': [0, 1, 2, 3, 4],
        'InfState': pd.Categorical(['I',  'I',  'I', 'I',  'I'], categories=['I', 'S']),
        'N':        [20, 20, 20, 20, 20]
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
                'in_beta': 0.2/5,
                'out_beta': 0.002/5,
                'inf_col': 'InfState',
                'trait_col': 'HH_Number',
                's_st': 'S',
                'i_st': 'I',
                'inf_to':'I',
                'stochastic': False
            }
        }
    
    assert returned_yaml == expected_yaml
