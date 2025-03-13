"""
Unit test for MultiStrainInfectiousProcess.py. Pytest package is used.
The MultiStrainInfectiousProcess.py class models a multi-strain infection process with cross-protect involved
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
from tabularepimdl.MultiStrainInfectiousProcess import MultiStrainInfectiousProcess

@pytest.fixture
def dummy_state():
    """
    Create a dummy DataFrame to simulate the state of a population aginst different strains.
    Returns: DataFrame containing population counts and strain states.
    """
    data = {
        'N':       [100,  200,  150],
        'Strain1': ['S',  'I',  'R'],
        'Strain2': ['S',  'S',  'I']
    }
    return (pd.DataFrame(data))

@pytest.fixture
def betas():
    """
    Create an array to contain transmission rates for different strains.
    Returns: Array containing tranmission rates.
    """
    betas = np.array([0.1, 0.05])
    return(betas)
    
@pytest.fixture
def columns():
    """
    Create a list to contain different strain names. 
    The list length and order should be the same as betas
    Returns: A column containing strain names.
    """
    columns = ["Strain1", "Strain2"]
    return(columns)

@pytest.fixture
def cross_protect():
    """
    Create a matrix to contain protection rate for intra- and inter-groups. 
    It should be a N(strain)* N(strain) matrix.
    Returns: A sqaure matrix containg protection rates.
    """
    cross_protect = np.array([[1.0, 0.5], [0.5, 1.0]])
    return(cross_protect)

@pytest.fixture
def multistrain_infectiousprocess(betas, columns, cross_protect): #pass above defined fixtures into initialization function
    """
    Initialize the MultiStrainInfectiousProcess object with specified parameters.
    Returns: Initialized MultiStrainInfectiousProcess object/instance.
    """
    return(MultiStrainInfectiousProcess(betas=betas, columns=columns, cross_protect=cross_protect))

def test_intialization(multistrain_infectiousprocess):
    """
    Test the initialization of the MultiStrainInfectiousProcess object.
    Args: multistrain_infectiousprocess object.
    """
    assert (multistrain_infectiousprocess.betas == np.array([0.1, 0.05])).all()
    assert multistrain_infectiousprocess.columns == ["Strain1", "Strain2"]
    assert (multistrain_infectiousprocess.cross_protect == np.array([[1.0, 0.5], [0.5, 1.0]])).all()
    assert multistrain_infectiousprocess.s_st == 'S'
    assert multistrain_infectiousprocess.i_st == 'I'
    assert multistrain_infectiousprocess.r_st == 'R'
    assert multistrain_infectiousprocess.inf_to == 'I'
    assert multistrain_infectiousprocess.stochastic == False
    assert multistrain_infectiousprocess.freq_dep == True
    assert len(multistrain_infectiousprocess.columns) == len(multistrain_infectiousprocess.betas) #Check that columns and betas have the same length
    assert multistrain_infectiousprocess.cross_protect.shape[0] == multistrain_infectiousprocess.cross_protect.shape[1] #Check that cross_protect is a square matrix 
    assert multistrain_infectiousprocess.cross_protect.shape[0] == len(multistrain_infectiousprocess.betas) #Check that cross_protect matrix dimension equals to the length of betas


def test_betas_on_freq_dep(dummy_state, betas):
    """
    Test the beta calculation if freq_dep is True.
    Args: dummy dataframe, betas array.
    """
    returned_betas = betas/(dummy_state['N'].sum())
    expectedc_betas = np.array([0.1, 0.05])/(100+200+150)

    assert (returned_betas == expectedc_betas).all()

def test_infectious_of_strain(multistrain_infectiousprocess, dummy_state, columns):
    """
    Test infectious calculation of each strain type.
    Args: multistrain_infectiousprocess object, dummy dataframe.
    """
    #for strain columns, check where the column values are equal to 'I', then multiply the N values element-wise, sums the result for each column 
    #return values use equation from source code
    returned_infectious = ((dummy_state[columns] == multistrain_infectiousprocess.i_st).multiply(dummy_state['N'], axis=0)).sum(axis=0)
    returned_infectious = np.array(returned_infectious)

    #expected values use known calculated data
    expected_infectious = pd.DataFrame({'Strain1': [False, True, False], 'Strain2': [False, False, True]})
    expected_infectious = expected_infectious.mul(np.array([100, 200, 150]), axis=0).sum(axis=0)
    expected_infectious = np.array(expected_infectious)

    assert(returned_infectious == expected_infectious).all() #infecous result is [200 150]


def test_cross_protection_multiplier(multistrain_infectiousprocess, dummy_state, columns, cross_protect):
    """
    Test beta values calculation based on cross-protect matrix.
    Args: multistrain_infectiousprocess object, dummy dataframe, columns, cross_protect.
    """
    #return values done with original loop-like equation
    #Convert the boolean result of x == r_st into an array of True (1) and False (0).
    #Multiply the boolean array with cross_protect, effectively zeroing out the values in cross_protect where the condition is False (0).
    #Computes the maximum value along each row of the resulting matrix.
    #returned_row_beta_mult = dummy_state[columns].apply(lambda x: ((x==multistrain_infectiousprocess.r_st).array* cross_protect).max(axis=1), 
    #                                                    axis=1, #apply lambda function to each row
    #                                                    result_type='expand' #expand the result into a DataFrame
    #                                                    )
    #returned_row_beta_mult = 1-returned_row_beta_mult

    #broadcasts recovered_mask so that each row of self.cross_protect gets multiplied by each corresponding element of recovered_mask
    recovered_mask = (dummy_state[columns] == multistrain_infectiousprocess.r_st).values
    returned_row_beta_mult = 1 - np.max(recovered_mask[:, np.newaxis] * cross_protect, axis=2)
    returned_row_beta_mult = pd.DataFrame(returned_row_beta_mult)


    #expected values use known calculated data
    boolean_df = pd.DataFrame({'Strain1': [False, False, True], 'Strain2': [False, False, False]})
    boolean_df['Strain1'] = boolean_df['Strain1'].map({True:1, False:0})
    boolean_df['Strain2'] = boolean_df['Strain2'].map({True:1, False:0})
    boolean_df0 = boolean_df*cross_protect[0]
    df0_max = boolean_df0.max(axis=1) #pick max value from each row
    boolean_df1 = boolean_df*cross_protect[1]
    df1_max = boolean_df1.max(axis=1)
    expected_row_beta_mul = pd.DataFrame({0: df0_max, 1: df1_max}) #combine two dataframes
    expected_row_beta_mul = 1-expected_row_beta_mul
    pd.testing.assert_frame_equal(returned_row_beta_mult, expected_row_beta_mul)
    
    #   0	1
    #0	1.0	1.0
    #1	1.0	1.0
    #2	0.0	0.5

def test_beta_on_susceptibility(multistrain_infectiousprocess, dummy_state, columns, cross_protect, betas):
    """
    Test beta values calculation based on above row_beta_mult for susceptibility.
    Args: multistrain_infectiousprocess object, dummy dataframe, columns, cross_protect, betas.
    """
    #return values use equation from source code
    expected_betas = np.array([0.1, 0.05])/(100+200+150)
    
    recovered_mask = (dummy_state[columns] == multistrain_infectiousprocess.r_st).values
    returned_row_beta_mult = 1 - np.max(recovered_mask[:, np.newaxis] * cross_protect, axis=2)
    returned_row_beta_mult = pd.DataFrame(returned_row_beta_mult)

    returned_row_beta = (returned_row_beta_mult * expected_betas * (dummy_state[columns]==multistrain_infectiousprocess.s_st).values)
    
    #expected values use known calculated data
    expected_row_beta = pd.DataFrame({0:[1.0, 1.0, 0.0], 1:[1.0, 1.0, 0.5]})*expected_betas*np.array([[ True, True], [False, True], [False, False]])

    pd.testing.assert_frame_equal(returned_row_beta, expected_row_beta)

    #     0         1
    #0  0.000222  0.000111
    #1  0.000000  0.000111
    #2  0.000000  0.000000

def test_no_coinfections(multistrain_infectiousprocess, dummy_state, columns, cross_protect, betas):
    """
    Test probablity of infectoin calculation based on above row_beta.
    Args: multistrain_infectiousprocess object, dummy dataframe, columns, cross_protect, betas.
    """
    #return values use equation from source code
    returned_infectious = ((dummy_state[columns] == multistrain_infectiousprocess.i_st).multiply(dummy_state['N'], axis=0)).sum(axis=0)
    returned_infectious = np.array(returned_infectious)

    expected_betas = np.array([0.1, 0.05])/(100+200+150)
    
    recovered_mask = (dummy_state[columns] == multistrain_infectiousprocess.r_st).values
    returned_row_beta_mult = 1 - np.max(recovered_mask[:, np.newaxis] * cross_protect, axis=2)
    returned_row_beta_mult = pd.DataFrame(returned_row_beta_mult)
    
    returned_row_beta = (returned_row_beta_mult * expected_betas * (dummy_state[columns]==multistrain_infectiousprocess.s_st).values)

    returned_row_beta = returned_row_beta.multiply(1-(dummy_state[columns] == multistrain_infectiousprocess.i_st).max(axis=1), axis=0)
    
    #prI done with original loop-like equation
    #returned_prI = 1-(np.exp(-1.0*returned_row_beta)).apply(lambda x: np.power(x, returned_infectious), axis=1)
    
    returned_prI = 1 - np.power(np.exp(-1.0 * returned_row_beta.values), returned_infectious)
    returned_prI = pd.DataFrame(returned_prI)

    #expected values use known calculated data
    expected_infectious = np.array([200, 150])
    expected_co_infection = 1 - pd.DataFrame({'Strain1': [False, True, False], 'Strain2': [False, False, True]}).max(axis=1)
    expected_row_beta = pd.DataFrame({0:[1.0, 1.0, 0.0], 1:[1.0, 1.0, 0.5]})*expected_betas*np.array([[ True, True], [False, True], [False, False]])
    expected_row_beta = expected_row_beta.multiply(expected_co_infection, axis=0)
    expected_prI = 1-(np.exp(-1.0*expected_row_beta)).apply(lambda x: np.power(x, expected_infectious), axis=1)
    
    pd.testing.assert_frame_equal(returned_prI, expected_prI)
    
#          0         1
#0  0.043471  0.016529
#1  0.000000  0.000000
#2  0.000000  0.000000

def test_prI(multistrain_infectiousprocess, dummy_state, columns, cross_protect, betas):
    """
    Test probablity of infectoin calculation based on above row_beta.
    Args: multistrain_infectiousprocess object, dummy dataframe, columns, cross_protect, betas.
    """
    #return values use equation from source code
    returned_infectious = ((dummy_state[columns] == multistrain_infectiousprocess.i_st).multiply(dummy_state['N'], axis=0)).sum(axis=0)
    returned_infectious = np.array(returned_infectious)

    expected_betas = np.array([0.1, 0.05])/(100+200+150)
    recovered_mask = (dummy_state[columns] == multistrain_infectiousprocess.r_st).values
    returned_row_beta_mult = 1 - np.max(recovered_mask[:, np.newaxis] * cross_protect, axis=2)
    returned_row_beta_mult = pd.DataFrame(returned_row_beta_mult)
    returned_row_beta = (returned_row_beta_mult * expected_betas * (dummy_state[columns]==multistrain_infectiousprocess.s_st).values)

    returned_row_beta = returned_row_beta.multiply(1-(dummy_state[columns] == multistrain_infectiousprocess.i_st).sum(axis=1), axis=0)
    returned_prI = 1 - np.power(np.exp(-1.0 * returned_row_beta.values), returned_infectious)
    returned_prI = pd.DataFrame(returned_prI)
    
    returned_deltas = dummy_state.loc[returned_prI.sum(axis=1)>0]
    returned_prI_filtered = returned_prI.loc[returned_prI.sum(axis=1)>0]
    returned_prI_filtered.columns=columns

    #expected values use known calculated data
    expected_prI = pd.DataFrame({0:[0.043471260896961295, 0.0, 0.0], 1:[0.01652854617837851, 0.0, 0.0]})
    expected_deltas = dummy_state.loc[expected_prI.sum(axis=1)>0]
    pd.testing.assert_frame_equal(returned_deltas, expected_deltas)

    expected_prI_filtered = expected_prI.loc[expected_prI.sum(axis=1)>0]
    expected_prI_filtered.columns = ['Strain1', 'Strain2']
    pd.testing.assert_frame_equal(returned_prI, expected_prI)

#deltas
#     N strain1 strain2
#0  100       S       S

#prI
#    strain1	 strain2
#0	0.043471	0.016529

def test_get_deltas_deterministic(multistrain_infectiousprocess, dummy_state):
    """
    #Test the get_deltas method for the deterministic scenario.
    #Args: multistrain_infectiousprocess, dummy_state.
    """
    #return values use get_deltas method from source code
    returned_deltas = multistrain_infectiousprocess.get_deltas(dummy_state)

    #expected values use known calculated data
    expected_prI = pd.DataFrame({'Strain1':[0.043471260896961295], 'Strain2':[0.01652854617837851]})
    expected_deltas = pd.DataFrame({'N':[100], 'Strain1':'S', 'Strain2':'S'})
    #deltas for determinsitic process
    expected_deltas = expected_deltas.assign(N=-expected_deltas.N*(1 - (1-expected_prI['Strain1'])*(1-expected_prI['Strain2'])))

    tmp1 = pd.DataFrame()
    for col in ['Strain1', 'Strain2']:
        tmp2 = expected_deltas.assign(N = -expected_deltas['N']* (expected_prI[col]/expected_prI.sum(axis=1)))
        tmp2[col]='I'
        tmp1 = pd.concat([tmp1, tmp2])
    expected_deltas = pd.concat([expected_deltas, tmp1]).reset_index(drop=True)
    expected_deltas = expected_deltas[expected_deltas['N']!=0]

    pd.testing.assert_frame_equal(returned_deltas, expected_deltas)
#           N	strain1	strain2
#0	-5.928129	S	    S
#1	4.295068	I	    S
#2	1.633061	S	    I

def test_get_deltas_stochastic(multistrain_infectiousprocess, dummy_state, columns):
    """
    #Test the get_deltas method for the stochastic scenario.
    #Args: multistrain_infectiousprocess object, dummy dataframe, columns.
    """
    with mock.patch("numpy.random.multinomial", return_value=np.array([7, 2, 91])):
        #return values use get_deltas method from source code
        returned_deltas = multistrain_infectiousprocess.get_deltas(dummy_state, stochastic=True)

        #expected values use known calculated data
        expected_prI = pd.DataFrame({'Strain1':[0.043471260896961295], 'Strain2':[0.01652854617837851]})
        expected_deltas = pd.DataFrame({'N':[100], 'Strain1':'S', 'Strain2':'S'})

        N_index = expected_deltas.columns.get_loc('N') #location is 0
        for i in range(expected_prI.shape[0]): #shape is 1 by 2
            tmp = np.array([7, 2, 91])
            #deltas for stochastic process
            expected_deltas.iloc[i, N_index] = -(tmp[0]+tmp[1]).sum() #delta's row 0 column 0 value is -9
            for j in range(expected_prI.shape[1]):
                toadd = expected_deltas.iloc[[i]] #deltas's ith row values, in this case row 0: [-9, 'S', 'S'] 
                toadd = toadd.assign(N=tmp[j]) #update toadd's column N value with tmp's jth value, in this case [7, 'S', 'S']
                toadd[columns[j]] = 'I' #update toadd's jth strain column value to I
                expected_deltas = pd.concat([expected_deltas, toadd])
        expected_deltas = expected_deltas.reset_index(drop=True)
        expected_deltas = expected_deltas[expected_deltas['N']!=0]
        
        pd.testing.assert_frame_equal(returned_deltas, expected_deltas)

#    N Strain1 Strain2
#0  -9       S       S
#1   7       I       S
#2   2       S       I

def test_to_yaml(multistrain_infectiousprocess):
    """
    Test the to_yaml method of the MultiStrainInfectiousProcess object.
    Args: multistrain_infectiousprocess object.
    """
    returned_yaml = multistrain_infectiousprocess.to_yaml()

    expected_yaml = {
            'tabularepimdl.MultiStrainInfectiousProcess': {
                'betas': np.array([0.1, 0.05]),
                'columns': ["Strain1", "Strain2"],
                'cross_protect': np.array([[1.0, 0.5], [0.5, 1.0]]),
                's_st': 'S',
                'i_st': 'I',
                'r_st': 'R',
                'inf_to': 'I',
                'freq_dep':True,
                'stochastic':False
            }
        }

    #extract matrix from yaml for comparison
    returned_yaml_betas_only = {'betas': returned_yaml['tabularepimdl.MultiStrainInfectiousProcess'].pop('betas')}
    expected_yaml_betas_only = {'betas': expected_yaml['tabularepimdl.MultiStrainInfectiousProcess'].pop('betas')}
        
    returned_yaml_cross_protect_matrix_only = {'cross_protect': returned_yaml['tabularepimdl.MultiStrainInfectiousProcess'].pop('cross_protect')}
    expected_yaml_cross_protect_matrix_only = {'cross_protect': expected_yaml['tabularepimdl.MultiStrainInfectiousProcess'].pop('cross_protect')}
        
    np.testing.assert_array_equal(returned_yaml_betas_only['betas'],  expected_yaml_betas_only['betas'])
    np.testing.assert_array_equal(returned_yaml_cross_protect_matrix_only['cross_protect'],  expected_yaml_cross_protect_matrix_only['cross_protect'])
    

    #compare the remaining parts of two yaml data
    assert expected_yaml == returned_yaml