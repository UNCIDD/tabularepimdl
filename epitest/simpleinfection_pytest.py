"""
Unit test for SimpleInfection.py. Pytest package is used.
The SimpleInfection class models a simple infectious disease process where individuals in a population 
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
from tabularepimdl.SimpleInfection import SimpleInfection

@pytest.fixture
def dummy_state(): #Creat a dummy DataFrame
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts and their infection states.
    """
    data = {
    'N': [10, 20, 30, 40], #population counts
    'Infection_State': ['S', 'I', 'S', 'I'] #infection state
    }
    return pd.DataFrame(data)

@pytest.fixture
def simple_infection(): #Initialize the SimpleInfection object
    """
    Initialize the SimpleInfection object with specified parameters.
    Returns: Initialized SimpleInfection object/instance.
    """
    return SimpleInfection(beta=0.3, column='Infection_State')
    
def test_initialization(simple_infection): #test simple_infection object initialization
    """
    Test the initialization of the SimpleInfection object.
    Args: simple_infection object.
    """
    assert simple_infection.beta == 0.3
    assert simple_infection.column == 'Infection_State'
    assert simple_infection.s_st == 'S'
    assert simple_infection.i_st == 'I'
    assert simple_infection.inf_to == 'I'
    assert simple_infection.freq_dep == True
    assert simple_infection.stochastic == False

def test_beta_freq_true(dummy_state): #test beta value when freq_dep is True
    """
    Test the beta attribute value when freq_dep is True.
    Args: beta's initial value, column name and freq_dep initial value.
    """
    si = SimpleInfection(beta=0.3, column='Infection_State', freq_dep=True)
    beta = si.beta/(dummy_state['N'].sum())
    assert si.freq_dep == True
    assert beta == 0.003 # =0.3/100

def test_beta_freq_flase(): #test beta value when freq_dep is False
    """
    Test the beta attribute value when freq_dep is False.    
    Args: beta's initial value, column name and freq_dep initial value.
    """
    si = SimpleInfection(beta=0.3, column='Infection_State', freq_dep=False)
    beta = si.beta
    assert si.freq_dep == False
    assert beta == 0.3

def test_infectious_calculation(simple_infection, dummy_state): #test infectious calculation
    """
    Test the calculation of the total number of infectious individuals.
    Args: simple_infection object and dummy dataframe.
    """
    infectious = dummy_state.loc[dummy_state[simple_infection.column]=='I', 'N'].sum()
    assert infectious == 60

def test_deltas_calculation(simple_infection, dummy_state): #test the deltas calculation
    """
    Test the number of susceptible individuals (deltas).
    Args: simple_infection object and dummy dataframe.
    """
    deltas = dummy_state.loc[dummy_state[simple_infection.column]=='S']
    expected_deltas = pd.DataFrame({
        'N': [10,30],
        'Infection_State':['S', 'S']
    })
    pd.testing.assert_frame_equal(deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True)) #reset df index to default integer index

def test_get_deltas_deterministic(simple_infection, dummy_state): #test get_deltas deterministic process
    """
    Test the get_deltas method for the deterministic scenario.
    Args: simple_infection object and dummy dataframe.
    """
    #freq_dep=True, stochastic=False
    returned_deltas_and_tmp = simple_infection.get_deltas(dummy_state)
    
    expected_deltas_of_subtractions = pd.DataFrame({
        'N': [-10 * (1 - np.power(np.exp(-1*0.003), 60)), -30 * (1 - np.power(np.exp(-1*0.003), 60))],
        'Infection_State': ['S', 'S']
    })

    expected_tmp_of_additions = pd.DataFrame({
        'N': [10 * (1 - np.power(np.exp(-1*0.003), 60)), 30 * (1 - np.power(np.exp(-1*0.003), 60))],
        'Infection_State': ['I', 'I']
    })

    expected_result = pd.concat([expected_deltas_of_subtractions, expected_tmp_of_additions])
    pd.testing.assert_frame_equal(returned_deltas_and_tmp.reset_index(drop=True), expected_result.reset_index(drop=True))

def test_get_deltas_stochastic(simple_infection, dummy_state): #test get_deltas stochastic process
    """
    Test the get_deltas method for the stochastic scenario. Mock binomial distribution's sample value
    Args: simple_infection object and dummy dataframe.
    """
    #freq_dep=True, stochastic=True
    with mock.patch('numpy.random.binomial', return_value=10):
        simple_infection.stochastic = True
        returned_deltas_and_tmp = simple_infection.get_deltas(dummy_state)
        
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

def test_to_yaml(simple_infection): #test to_yaml Method
    """
    Test the to_yaml method of the SimpleInfection object.
    Args: simple_infection object.
    """
    expected_yaml = {
            'tabularepimdl.SimpleInfection': {
            'beta': 0.3,
            'column': 'Infection_State',
            's_st': 'S',
            'i_st': 'I',
            'inf_to': 'I',
            'freq_dep': True,
            'stochastic': False
            }
        }
    assert simple_infection.to_yaml() == expected_yaml