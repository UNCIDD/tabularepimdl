# Rule Module

## 1. Introduction
Description: This module provides a framework for defining transition rules in an epidemic model. These rules govern how the epidemic current state evolves over time, based on current conditions.

Purpose: The 'Rule' class serves as an abstract base class for defining specific transition rules. It provides a standard interface for loading rules from YAML files and converting them back to YAML format. It also offers method of calucating changes (deltas) from current state in the amount of tempsteps.

## 2. Getting Started
### Installation
pip install -r requirements.txt

### Dependency (module and unit test)
* Python 3.11.9
* abc
* inspect
* importlib
* pandas
* numpy
* pytest
* yaml

### Quickstart
* Create a subclass of Rule.
* Implement the get_deltas and to_yaml methods.
* Load and save rules using the from_yaml and to_yaml methods.

## 3. Usage
### Class: 'Rule'
Description: The 'Rule' class is an abstract base class that defines the interface for epidemic state transition rules.

### Methods:
#### get_deltas(current_state, dt:float, stochastic):
* Description: Computes the changes (deltas) to the epidemic state.
* Parameters:
  * current_state: The current epidemic state as a DataFrame.
  * dt: Timestep size.
  * stochastic: Whether the rule is stochastic.
* Returns: A dataframe of state changes.

#### from_yaml(rule_yaml):
* Description: Loads a rule from a YAML dictionary.
* Parameters:
  * rule_yaml: Dictionary containing the YAML definition.
* Returns: An instance of the rule.

#### from_yaml_def(definition):
* Description: Loads the parameter values from a rule which is from a YAML dictionary.
* Parameters: Dictionary values.
* Returns: An instance of the rule.

#### to_yaml():
* Description: Converts the rule to a dictionary suitable for YAML serialization.
* Returns: A dictionary representation of the rule.

## 4. Testing
### Testing Strategy
* Unit Test: Each method in the 'Rule' class and its subclasses has corresponding unit tests to ensure correct functionality.

### Test Setup
* A sample YAML file.

### Example Tests:
* test_from_yaml: Verifies that the rule is correctly loaded from a YAML file.
* test_get_deltas: Ensures that the correct deltas are calculated based on the current state.
* test_to_yaml: Confirms that the rule is correctly serialized back to YAML format.
