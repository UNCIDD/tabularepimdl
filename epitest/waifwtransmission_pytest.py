"""
Unit test for WAIFWTransmission.py. Pytest package is used.
The WAIFWTransmission.py class models a transmission process based on 
a simple WAIFW (who acuqire infection from whom) transmission matrix
The unit tests ensure that the class behaves as expected under various conditions, including both 
deterministic and stochastic scenarios. 
"""

import pytest
import pandas as pd
import numpy as np
from unittest import mock #used for mocking binomial distribution

import sys
sys.path.append('../')
from tabularepimdl.WAIFWTransmission import WAIFWTransmission

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population.
    Returns: DataFrame containing population counts, their infection states and hospitalization states
    """
    data = {
        'N': [50, 5, 40, 10],
        'Infection_State': ['S', 'I', 'S', 'I'], #links to inf_col
        'Age_Group': pd.Categorical(['youth', 'youth', 'adult', 'adult'], categories=['youth', 'adult']) #links to group_col, need to designate the order of categories
    }
    return (pd.DataFrame(data))

@pytest.fixture
def dummy_waifw_matrix():
    """
    Create a dummy WAIFW matrix to simulate the transmission rate.
    Returns: matrix with two groups - group 1 and 2, transmission rates are beta1,2, beta2,2, beta2,1, beta2,2
    """
    waifw_matrix = np.array([[0.1, 0.2],
                             [0.3, 0.4]])
    return(waifw_matrix)

@pytest.fixture()
def waifw_transmission(dummy_waifw_matrix):
    """
    Initialize the WAIFWTransmission object with specified parameters.
    Returns: Initialized WAIFWTransmission object/instance.
    """
    return(WAIFWTransmission(waifw_matrix=dummy_waifw_matrix, inf_col='Infection_State', group_col='Age_Group'))

def test_initialization(waifw_transmission):
    """
    Test the initialization of the WAIFWTransmission object.
    Args: waifw_transmission object.
    """
    assert (waifw_transmission.waifw_matrix == np.array([[0.1, 0.3], [0.2, 0.4]])).all() #initial waifw_matrix is transposed within WAIFW rule
    assert waifw_transmission.inf_col == 'Infection_State'
    assert waifw_transmission.group_col == 'Age_Group'
    assert waifw_transmission.s_st == 'S'
    assert waifw_transmission.i_st == 'I'
    assert waifw_transmission.inf_to == 'I'
    assert waifw_transmission.stochastic == False

def test_intialization_error_handling():
    """
    Test the error handling for input matrix parameters. Set the following parameters with incorrect values or format in the way to trigger ValueError.
    If setting the correct values and format with commented out data, the test will fail with message "Failed: DID NOT RAISE <class 'ValueError'>"
    """
    check_matrix = np.array([0.1, -0.2]) #np.array([0.1, 0.05])
    check_inf_col = 3 #"Infection_State"
    check_group_col = ['A', 'B', 'C'] #"Host"

    with pytest.raises(ValueError) as excinfo:
        WAIFWTransmission(waifw_matrix=check_matrix, inf_col=check_inf_col, group_col=check_group_col)

def test_len_of_categories_match_len_of_waifw(waifw_transmission):
    """
    Test the number of unique categories in current_state's group_col match the waifw matrix's length.
    Args: waifw_transmission object.
    """
    state_with_more_categories = pd.DataFrame(
        {
        'N': [50, 5, 40, 10],
        'Infection_State': ['S', 'I', 'S', 'I'], #links to inf_col
        'Age_Group': ['youth', 'child', 'senior', 'adult'] #links to group_col, reserved to test error handling in get_deltas() method
        #'Age_Group': ['youth', 'youth', 'adult', 'adult'] #links to group_col, reserved to trigger ValueError in get_deltas() method
        }
    )
    
    with pytest.raises(ValueError) as excinfo:
        waifw_transmission.get_deltas(state_with_more_categories, stochastic=True)

def test_inf_array_slicing(waifw_transmission, dummy_state):
    """
    Test slicing and aggregation of current_state.
    Args: waifw_transmission object and dummy dataframe.
    """
    #two different approaches for calculating infection array
    inf_array_groupby = dummy_state.loc[dummy_state[waifw_transmission.inf_col]==waifw_transmission.i_st].groupby(waifw_transmission.group_col, observed=False)['N'].sum(numeric_only=True).values
    inf_array_bincount = np.bincount(dummy_state.loc[dummy_state[waifw_transmission.inf_col]==waifw_transmission.i_st, waifw_transmission.group_col].cat.codes, #array input
                                     dummy_state.loc[dummy_state[waifw_transmission.inf_col]==waifw_transmission.i_st, 'N'], #weights
                                     minlength=len(dummy_state[waifw_transmission.group_col].cat.categories)) #minimum length

    expected_inf_array = np.array([5, 10]) #in each age group, sum the number of individuals whose infection_state is I

    assert (inf_array_groupby == expected_inf_array).all()
    assert (inf_array_bincount == expected_inf_array).all()

def test_categorical_type(waifw_transmission, dummy_state):
    """
    Test the parameter data type defined in current_state.
    Args: waifw_transmission object and dummy dataframe.
    """
    assert isinstance(dummy_state[waifw_transmission.group_col].dtype, pd.CategoricalDtype)


def test_probablity_of_infection_calculation(waifw_transmission, dummy_state, dummy_waifw_matrix):
    """
    Test infection probability for individuals from filtered groups.
    Args: waifw_transmission object, dummy dataframe and dummy waifw_matrix.
    """
    inf_array = dummy_state.loc[dummy_state[waifw_transmission.inf_col]==waifw_transmission.i_st].groupby(waifw_transmission.group_col, observed=False)['N'].sum(numeric_only=True).values
    prI = np.power(np.exp(-1.0*waifw_transmission.waifw_matrix), inf_array)
    prI = 1-prI.prod(axis=1) #axis=1 makes column-elements multiply

    expected_inf_array = np.array([5, 10])
    expected_prI = np.power(np.exp(-1.0*dummy_waifw_matrix.T), expected_inf_array) #transpose dummy_waifw_matrix as waifw_matrix is transposed within WAIFW rule
    expected_prI = 1-expected_prI.prod(axis=1)

    assert (prI == expected_prI).all() #array([0.96980262, 0.99326205]), these output values are rounded floating numbers
    

def test_deltas_calculation(waifw_transmission, dummy_state):
    """
    Test the number of incident observations (deltas).
    Args: waifw_transmission object and dummy dataframe.
    """
    deltas = dummy_state.loc[dummy_state[waifw_transmission.inf_col]==waifw_transmission.s_st]

    expected_deltas = pd.DataFrame({
        'N': [50, 40],
        'Infection_State': ['S', 'S'], #links to inf_col
        'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult']) #links to group_col
    })

    pd.testing.assert_frame_equal(deltas.reset_index(drop=True), expected_deltas.reset_index(drop=True))

def test_get_deltas_deterministic(waifw_transmission, dummy_state, dummy_waifw_matrix):
    """
    #Test the get_deltas method for the deterministic scenario.
    #Args: waifw_transmission object, dummy dataframe and dummy matrix.
    """
    returned_rc = waifw_transmission.get_deltas(dummy_state)

    filtered_deltas = pd.DataFrame({
        'N': [50, 40],
        'Infection_State': ['S', 'S'], #links to inf_col
        'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult']) #links to group_col
    })

    expected_inf_array = np.array([5, 10])
    prI_calc = np.power(np.exp(-1.0*dummy_waifw_matrix.T), expected_inf_array)#transpose dummy_waifw_matrix as waifw_matrix is transposed within WAIFW rule
    prI_calc = 1-prI_calc.prod(axis=1)
    prI_rounded = np.array([0.96980262, 0.99326205])#rounded result from test_probablity_of_infection_calculation
    
    #check prI calculated values
    assert(abs(prI_calc - prI_rounded) < 1e-8).all() #method1: if diff is less than 1e-8 magnitude, then pass

    #check prI indexed values
    prI_indexed = prI_calc[filtered_deltas['Age_Group'].cat.codes] #codes should be 0 and 1 now
    #assert(abs(prI_indexed - prI_rounded) < 1e-8).all() #if diff is less than 1e-8 magnitude, then pass
    assert (prI_indexed == np.array([0.9698026165776815, 0.9932620530009145])).all() #method2: use actual values to compare with programmatic values
    
    #generate expected deltas value with filtered deltas value and actual prI values
    expected_deltas = pd.DataFrame({
        'N': [-50, -40]*np.array([0.9698026165776815, 0.9932620530009145]),
        'Infection_State': ['S', 'S'], #links to inf_col
        'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult']) #links to group_col
    })

    expected_deltas_add = pd.DataFrame({
        'N': [50, 40]*np.array([0.9698026165776815, 0.9932620530009145]),
        'Infection_State': ['I', 'I'], #links to inf_col
        'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult']) #links to group_col
    })
    
    #concatenate expected deltas value and expected deltas addition value
    expected_rc = pd.concat([expected_deltas, expected_deltas_add])
    expected_rc = expected_rc.loc[expected_rc['N']!=0].reset_index(drop=True)

    """
    The expected_rc should be 
    {
        'N': [-x, -y, x, y], #x, y are numerical values
        'Infection_State': ['S', 'S', 'I', 'I'],
        'Age_Group': pd.Categorical(['youth', 'adult', 'youth', 'adult'])
    }
    """

    pd.testing.assert_frame_equal(returned_rc, expected_rc) #index has been reseted in the original class code          

def test_get_deltas_stochastic(waifw_transmission, dummy_state):
        """
        Test the get_deltas method for the stochastic scenario.
        #Args: waifw_transmission object, dummy dataframe.
        """
        with mock.patch("numpy.random.binomial", return_value=20):
            returned_rc = waifw_transmission.get_deltas(dummy_state, stochastic=True)
             
            expected_deltas = pd.DataFrame({
                 'N':               [-20, -20],
                 'Infection_State': ['S', 'S'],
                 'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult'])
            })

            expected_deltas_add = pd.DataFrame({
                 'N':               [20, 20],
                 'Infection_State': ['I', 'I'],
                 'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult'])
            })

            expected_rc = pd.concat([expected_deltas, expected_deltas_add])
            expected_rc = expected_rc.loc[expected_rc['N']!=0].reset_index(drop=True)

            pd.testing.assert_frame_equal(returned_rc, expected_rc) #index has been reseted in the original class code          

def test_to_yaml(waifw_transmission):
    """
    Test the to_yaml method of the WAIFWTransmission object.
    Args: waifw_transmission object.
    """
    returned_yaml = waifw_transmission.to_yaml()

    expected_yaml = {
        'tabularepimdl.WAIFWTransmission' : {
                'waifw_matrix' : np.array([[0.1, 0.2], [0.3, 0.4]]),
                'inf_col' : 'Infection_State',
                'group_col' : 'Age_Group',
                's_st': 'S',
                'i_st': 'I',
                'inf_to': 'I',
                'stochastic': False
        }
    }
    
    
    #extract matrix from yaml for comparison
    expected_yaml_matrix_only = {'waifw_matrix': expected_yaml['tabularepimdl.WAIFWTransmission'].pop('waifw_matrix')}
    returned_yaml_matrix_only = {'waifw_matrix': returned_yaml['tabularepimdl.WAIFWTransmission'].pop('waifw_matrix')}
    np.testing.assert_array_equal(returned_yaml_matrix_only['waifw_matrix'],  expected_yaml_matrix_only['waifw_matrix'])

    #compare the remaining parts of two yaml data
    assert expected_yaml == returned_yaml