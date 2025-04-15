"""
Unit test for EpiModel.py. Pytest package is used.
The EpiModel applies a list of rules to a changing current state through 
some number of time steps to produce an epidemic.
The unit tests ensure that the class behaves as expected under various conditions, including both 
deterministic and stochastic scenarios. 
"""
import pytest
import pandas as pd
import numpy as np
from unittest import mock #used for mocking binomial distribution
import copy

import os
import sys
sys.path.append('../')
from tabularepimdl.Rule import Rule
from tabularepimdl.BirthProcess import BirthProcess
from tabularepimdl.SimpleInfection import SimpleInfection
from tabularepimdl.SimpleTransition import SimpleTransition
from tabularepimdl.WAIFWTransmission import WAIFWTransmission
from tabularepimdl.EpiModel import EpiModel

@pytest.fixture
def init_state():
    """
    Create a dummy DataFrame to simulate the initial state of a population.
    Returns: DataFrame containing population counts, their infection states and age groups.
    """
    init_state = {
    'T': [0, 0], #column T might not be needed in initial state setup, it will be added in do_timestep method
    'N': [100, 200],
    'Infection_State': ['S', 'I'],
    'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult']) #links to group_col of WAIFWTransmission, need to designate the order of categories
    }
    return(pd.DataFrame(init_state))

@pytest.fixture
def cur_state(init_state):
    """
    Create a current state that equals to the initial state of a population.
    Returns: DataFrame containing population counts, their infection states and age groups.
    """
    cur_state = init_state.copy()
    return(cur_state)

@pytest.fixture
def full_epi(init_state):
    """
    Create a full epidemic that equals to the initial state of a population.
    Returns: DataFrame containing population counts, their infection states and age groups.
    """
    full_epi = init_state.copy()
    return(full_epi)

@pytest.fixture
def instantiated_rules():
    """
    Create a list of rules which are instantiated through existing rule classes under tabluarepimdl package.
    Birthprocess, SimpleInfection, SimpleTransition, WAIFWTransmission classes are selected.
    Returns: a list of instantiated rules.
    """
    #initialize each rule with their parameter values
    birth_process = BirthProcess(rate=0.1, start_state_sig={'age': 10, 'health': 'good'}, stochastic=False)
    simple_infection = SimpleInfection(beta=0.3, column='Infection_State', s_st='S', i_st='I', inf_to='I', freq_dep=True, stochastic=False)
    simple_transition = SimpleTransition(column='Infection_State', from_st='S', to_st='I', rate=0.3, stochastic=False)
    waifw_transmission = WAIFWTransmission(waifw_matrix= np.array([[0.1, 0.2], [0.3, 0.4]]), inf_col='Infection_State', group_col='Age_Group', s_st='S', i_st='I', inf_to='I', stochastic=False)

    #put the rules to a list of lists. It will be used in __init__ method and to_yaml method of EpiModel
    instantiated_rules = [ [birth_process, simple_infection], [simple_transition], [birth_process, waifw_transmission] ]
    return(instantiated_rules)

@pytest.fixture
def epi_yaml():
    """
    Create an object that contains the init_sate, rule classes and stochastic policy all in dictionary data type.
    Returns: epi_yaml dictionary.
    """
    epi_yaml = {
        'init_state': {
                        'T': [0, 0],
                        'N': [100, 200],
                        'Infection_State': ['S', 'I'],
                        'Age_Group': pd.Categorical(['youth', 'adult'], categories=['youth', 'adult']) #links to group_col of WAIFWTransmission, need to designate the order of categories
                      },

        'rules': [ #rules is set up as a single list for now --> rules: [ {b_p}, {s_i}, {s_t}, {w_t} ]
                    { #birth_process
                    'tabularepimdl.BirthProcess': {
                                                    'rate': 0.1,
                                                    'start_state_sig': {'age': 10, 'health': 'good'},
                                                    'stochastic': False
                                                   }
                    },
                    { #simple_infection
                    'tabularepimdl.SimpleInfection': {
                                                        'beta': 0.3,
                                                        'column': 'Infection_State',
                                                        's_st': 'S',
                                                        'i_st': 'I',
                                                        'inf_to': 'I',
                                                        'freq_dep': True,
                                                        'stochastic': False
                                                    }
                    },
                    { #simple_transition
                    'tabularepimdl.SimpleTransition': {
                                                        'column': 'Infection_State',
                                                        'from_st': 'S',
                                                        'to_st': 'I',
                                                        'rate': 0.3, 
                                                        'stochastic': False
                                                    }
                    },
                    { #waifw_transmission
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
                ],

        'stoch_policy': 'rule_based'
    }
    return(epi_yaml)

@pytest.fixture
def dt():
    """
    Initialize delta t value
    Returns: dt value.
    """
    dt = 1.0
    return (dt)

@pytest.fixture
def epimodel(init_state, instantiated_rules):
    """
    Initialize the EpiModel object with specified parameters.
    Returns: Initialized EpiModel object/instance.
    """
    return(EpiModel(init_state=init_state, rules=instantiated_rules))

#The following four fixtures are to create rule class objects for EpiModel's comparison purpose
@pytest.fixture
def b_p():
    """
    Initialize the BirthProcess rule with specified parameters.
    Returns: Initialized BirthProcess object/instance.
    """
    b_p = BirthProcess(rate=0.1, start_state_sig={'age': 10, 'health': 'good'}, stochastic=False)
    return(b_p)

@pytest.fixture
def s_i():
    """
    Initialize the SimpleInfection rule with specified parameters.
    Returns: Initialized SimpleInfection object/instance.
    """
    s_i = SimpleInfection(beta=0.3, column='Infection_State', s_st='S', i_st='I', inf_to='I', freq_dep=True, stochastic=False)
    return(s_i)

@pytest.fixture
def s_t():
    """
    Initialize the SimpleTransition rule with specified parameters.
    Returns: Initialized SimpleTransition object/instance.
    """
    s_t = SimpleTransition(column='Infection_State', from_st='S', to_st='I', rate=0.3, stochastic=False)
    return(s_t)

@pytest.fixture
def w_t():
    """
    Initialize the WAIFWTransmission rule with specified parameters.
    Returns: Initialized WAIFWTransmission object/instance.
    """
    w_t = WAIFWTransmission(waifw_matrix= np.array([[0.1, 0.2], [0.3, 0.4]]), inf_col='Infection_State', group_col='Age_Group', s_st='S', i_st='I', inf_to='I', stochastic=False)
    return(w_t)

def test_initialization(epimodel, b_p, s_i, s_t, w_t):
    """
    Test the initialization of the EpiModel object.
    Args: epimodel object.
    """
    assert epimodel.stoch_policy == 'rule_based'
    assert isinstance(epimodel.rules[0],list) == True

    #to compare if two objects are equal, each class' __eq__ method needs to be re-written. Instead, compare each list element's selected parameters here.
    #birth process
    assert epimodel.rules[0][0].rate == b_p.rate
    pd.testing.assert_frame_equal(epimodel.rules[0][0].start_state_sig, b_p.start_state_sig)

    #simple infection
    assert epimodel.rules[0][1].column == s_i.column
    assert epimodel.rules[0][1].stochastic == s_i.stochastic
    
    #simple transition
    assert epimodel.rules[1][0].from_st == s_t.from_st
    assert epimodel.rules[1][0].rate == s_t.rate

    #waifw transmission
    assert (epimodel.rules[2][1].waifw_matrix == w_t.waifw_matrix).all()
    assert epimodel.rules[2][1].group_col == w_t.group_col

def test_reset(epimodel, init_state):
    """
    Test the reset method.
    Args: epimodel object.
    """
    epimodel.cur_state = pd.DataFrame({'cur_state':['changed']}) #assign a new dataframe to cur_state
    epimodel.full_epi = pd.DataFrame({'full_epi':['changed']}) #assign a new dataframe to full_epi

    epimodel.cur_state, epimodel.full_epi = epimodel.reset() #reset cur_state and full_epi

    pd.testing.assert_frame_equal(epimodel.cur_state, init_state)
    pd.testing.assert_frame_equal(epimodel.full_epi, init_state)

def test_from_yaml(epimodel, epi_yaml, init_state, b_p, s_i, s_t, w_t):
    """
    Test the from_yaml method.
    Args: epimodel object, epi_yaml dictionary, initial state, rule object such as BirthProcess, SimpleTranistion, SimpleInfection, WAIFWTranmission.
    """
    #create an instance of EpiModel by using the data defined in epi_yaml dictionary
    #this EpiModel instance contains init_state dataframe, a list of list which contains different rule objects and, and stochastic policy string
    returned_class_from_yaml = epimodel.from_yaml(epi_yaml)

    #compare returned class' init_state to the defined init_state in EpiModel
    pd.testing.assert_frame_equal(returned_class_from_yaml.init_state, init_state) 

    #compare returned class' stochastic policy to the defined sotchasitc policy in EpiModel
    assert returned_class_from_yaml.stoch_policy == 'rule_based' 

    #epi_yaml object sets up all rules in a single list which is turned to a list of list by from_yaml method --> rules: [ [{}, {}, {}, {}] ]
    #compare selected parameters of each rule in this single list to the same parameters of the same rule in the instantiated rules/list
    #birth process
    assert returned_class_from_yaml.rules[0][0].rate == b_p.rate
    pd.testing.assert_frame_equal(returned_class_from_yaml.rules[0][0].start_state_sig, b_p.start_state_sig)

    #simple infection
    assert returned_class_from_yaml.rules[0][1].column == s_i.column
    assert returned_class_from_yaml.rules[0][1].stochastic == s_i.stochastic
    
    #simple transition
    assert returned_class_from_yaml.rules[0][2].from_st == s_t.from_st
    assert returned_class_from_yaml.rules[0][2].rate == s_t.rate

    #waifw transmission
    assert (returned_class_from_yaml.rules[0][3].waifw_matrix == w_t.waifw_matrix).all()
    assert returned_class_from_yaml.rules[0][3].group_col == w_t.group_col

def test_to_yaml(epimodel, init_state, epi_yaml):
    """
    Test the to_yaml method.
    Args: epimodel object, instantiated rules such as BirthProcess, SimpleTranistion, SimpleInfection, WAIFWTranmission, epi_yaml dictionary.
    """
    #create a dictionary that is converted from instantiated rules defined in EpiModel instance
    returned_dict_to_yaml = epimodel.to_yaml()
    #instantiated rules is a list of lists --> [ [birth_process, simple_infection], [simple_transition], [birth_process, waifw_transmission] ]
    #so returned dictionary's rule value is a also list of lists --> rules: [ [b_p, s_i], [s_t], [b_p, w_t] ], each object in this list can be compared to the corresponding object in epi_yaml dictionary

    #compare returned dictionary's stoch_policy value to the defined string in EpiModel
    assert returned_dict_to_yaml['stoch_policy'] == 'rule_based' 

    #compare returned dictionary's init_state value to the dict-formated init_state defined in EpiModel
    assert returned_dict_to_yaml['init_state'] == init_state.to_dict(orient='list')

    #note epi_yaml dictionary's rules is just a list --> rules: [ {b_p}, {s_i}, {s_t}, {w_t} ]
    #birth process
    assert returned_dict_to_yaml['rules'][0][0]['tabularepimdl.BirthProcess']['rate'] == epi_yaml['rules'][0]['tabularepimdl.BirthProcess']['rate']
    
    #convert epi_yaml's start_state_sig object to dataframe in order to line up with the dataframe format generated by BirthProcess's __init__ method
    pd.testing.assert_frame_equal(returned_dict_to_yaml['rules'][0][0]['tabularepimdl.BirthProcess']['start_state_sig'], pd.DataFrame([epi_yaml['rules'][0]['tabularepimdl.BirthProcess']['start_state_sig']]).reset_index(drop=True))
    #the other way is to convert epi_yaml's start_state_sig value to be a list type, and convert the dataframe start_state_sig in returned dict to list type, then compare.
    start_state_sig_list_type = {key: [value] for key, value in epi_yaml['rules'][0]['tabularepimdl.BirthProcess']['start_state_sig'].items()}
    assert returned_dict_to_yaml['rules'][0][0]['tabularepimdl.BirthProcess']['start_state_sig'].to_dict(orient='list') == start_state_sig_list_type

    #simple infection, straight dictionary to dictionary comparison
    assert returned_dict_to_yaml['rules'][0][1] == epi_yaml['rules'][1]

    #simple transition, straight dictionary to dictionary comparison
    assert returned_dict_to_yaml['rules'][1][0] == epi_yaml['rules'][2]

    #waifw transmission
    #array comparison for waifw_matrix
    assert (returned_dict_to_yaml['rules'][2][1]['tabularepimdl.WAIFWTransmission']['waifw_matrix'] == epi_yaml['rules'][3]['tabularepimdl.WAIFWTransmission']['waifw_matrix']).all()
    assert returned_dict_to_yaml['rules'][2][1]['tabularepimdl.WAIFWTransmission']['group_col'] == epi_yaml['rules'][3]['tabularepimdl.WAIFWTransmission']['group_col']

def test_do_timestep_rule_based(epimodel, dt, instantiated_rules, cur_state, stoch_policy = "rule_based"):
    """
    Test the do_timestamp method with rule_based policy.
    Args: epimodel object, instantiated rules, current state.
    """
    returned_cur_state = epimodel.do_timestep(dt, ret_nw_state=True)

    for ruleset in instantiated_rules:
        all_deltas = pd.DataFrame() #reset all_deltas
        for rule in ruleset:
            if stoch_policy == "rule_based":
                nw_deltas = rule.get_deltas(cur_state, dt=dt) 
            else:
                nw_deltas = rule.get_deltas(cur_state, dt=dt, stochastic=(stoch_policy=="stochastic"))
            
            all_deltas = pd.concat([all_deltas, nw_deltas])
       
        if all_deltas.shape[0]==0:
            continue

        #all_deltas = all_deltas.assign(T=0)
        if 'T' in cur_state.columns:
            pass #if column T exists in the initial cur_state dataframe, do nothing
        else:
            all_deltas = all_deltas.assign(T=0.0) #add a new column T with initial value 0 to all_deltas
            full_epi = full_epi.assign(T=0.0)

        nw_state = pd.concat([cur_state, all_deltas])#.reset_index(drop=True) #append all_deltas to cur_state
    
        # Get grouping columns
        tbr = {'N','T'}
        gp_cols = [item for item in all_deltas.columns if item not in tbr] #filter out all column names that are not 'N' or 'T' => ['age', 'health', 'Infection_State', 'Age_Group']

        if gp_cols:
            nw_state = nw_state.groupby(gp_cols, dropna=False, observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index() #group by the filtered columns and make aggregation on N and T
            #question: do we want to use dropna=False option to keep data iterating through the for loop?
            #question: T column values are always assigned with 0, why do we search the max value of T?
    
        nw_state = nw_state[nw_state['N']!=0].reset_index(drop=True)
        cur_state = nw_state
        print('test case cur_state is\n', cur_state)
    
    expected_cur_state = cur_state.assign(T=max(cur_state['T'])+dt) #T values are always 0, +dt = 1
    
    pd.testing.assert_frame_equal(returned_cur_state, expected_cur_state)

#    age	health	Infection_State	Age_Group	N	        T
#0	10.0	good	NaN	            NaN	        59.814324	1.0
#1	NaN	    NaN	    I	            youth	    100.000000	1.0
#2	NaN	    NaN	    I	            adult	    200.000000	1.0

def test_do_timestep_stochastic(epimodel, dt):
    """
    Given each rule class will generate random data due to random.poisson or random.binomial function used,
    the stochasitic process will not be tested. The do_timestep_rule_based test has the method's functionality covered.
    Test the do_timestamp method with stochastic process.
    Args: epimodel object, instantiated rules, current state.
    """

def test_add_rule(epimodel, s_i, s_t, epi_yaml):
    """
    Test the add_rule method, verifying new rules can be added sucessfully to EpiModel.
    Args: epimodel object, two sample rules simpleinfection and simpletransition.
    """
    epimodel.add_rule([s_i, s_t]) #add a list of two new rules, initial rules will become [ [b_p, s_i], [s_t], [b_p, w_t], [si, st] ]

    #create a dictionary that is converted from instantiated rules and added new rules
    returned_dict_to_yaml_with_new_rules = epimodel.to_yaml()

    #each object in this returned dictionary can be compared to the corresponding object in epi_yaml dictionary
    #epi_yaml['rules']: [ {b_p}, {s_i}, {s_t}, {w_t} ]
    #newly added simple infection, straight dictionary to dictionary comparison
    assert returned_dict_to_yaml_with_new_rules['rules'][3][0] == epi_yaml['rules'][1]

    #newly added simple transition, straight dictionary to dictionary comparison
    assert returned_dict_to_yaml_with_new_rules['rules'][3][1] == epi_yaml['rules'][2]




