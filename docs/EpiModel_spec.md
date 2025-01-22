# EpiModel Module

## 1. Introduction
Description: This module applies a list of rules to a changing current state through some number of time steps to produce an epidemic. It has attributes representing the current state and the full epidemic thus far.

Purpose: The 'EpiModel' is a class of utilizing rule classes. It provides a standard interface for loading rules from YAML files and converting rules back to YAML format. It also offers method of applying a list of rules to a changing current state through some number of time steps to produce an epidemic.

## 2. Getting Started
### Installation
pip install -r requirements.txt

### Dependency
* Python 3.11.9
* pandas
* numpy
* pytest
* copy

### Quickstart
* Create a subclass of EpiModel.
* Implement the from_yaml, to_yaml, do_timestep and add_rule methods.

## 3. Usage
### Class: 'EpiModel'
Description: applies a list of rules to a changing current state through some number of time steps to produce an epidemic. It has attributes representing the current state and the full epidemic thus far.

### Methods:
#### reset():
* Description: Resets the class state to have the initial state.
* Parameters: None.
* Returns: initialized current state and full epi.

#### from_yaml(epi_yaml):
* Description: Creates the class from a dictionary object presumed to be read in from a yaml object.
* Parameters: 
  * epi_yaml: A dictionary created from the epi yaml object.
* Returns: An instantiated EpiModel with initial state, used rules and stochastic policy.

#### to_yaml():
* Description: Creates a dictionary object appropriate to be saved to YAML for this EpiModel. The dictionary will include EpiModel's inital state, rules used and stochastic policy.
* Parameters:
  * save_epi: Flag to indicate if the full epidemic data should be saved.
  * save_state: Flag to indicate if the current state data shold be saved.
* Returns: A dictionary representation of the EpiModel.

#### do_timestep(dt, ret_nw_state):
* Description: Changes current state through some number of time steps to produce an epidemic.
* Parameters:
  * dt: Timestep size.
  * ret_nw_state: Flag to indicate if the new state data should be saved.
* Returns: A dataframe of new state.

#### add_rule(new_rule):
* Description: Adds a new rule or a list of rules to the EpiModel.
* Parameters:
  * new_rule: A single rule object or a list of rule objects.
* Returns: Updated list of rules for EpiModel.

## 4. Testing
### Testing Strategy
* Unit Test: Each method in the 'EpiModel' class has corresponding unit tests to ensure correct functionality.

### Test Setup
* A dummy DataFrame to simulate the state of a EpiModel.
* An instantiated rule set.
* A dictionary object.
* Temp step size.
* Individual instantiated rules, a few rules are selected from the rule class pool.

### Example Tests:
* test_intialization: Verifies that the rule is correctly initialized with the setup above.
* test_rest: Verifies that the EpiModel's current state and full epi are set back to their initial state.
* test_from_yaml: Ensures that the EpiModel is correctly created through the epi_yaml dictionary object.
* test_to_yaml: Ensures that the EpiModel is correctly saved to a dictionary object which can be saved to a YAML file.
* test_do_timestep_rule_based: Verifies the current state values after timestep size.
* test_add_rule: Verifies new rules can be added sucessfully to EpiModel.

