# BirthProcess Module

## 1. Introduction
Description: This module represents a birth process where people are borne based on a birth rate based on the full poplation size.

Purpose: The 'BirthProcess' is a subclass of 'Rule' class. It provides a standard interface for converting rules to YAML format. It also offers method of calucating changes (deltas) from current state in the amount of tempsteps including both deterministic and stochastic scenarios.

## 2. Getting Started
### Installation
pip install -r requirements.txt

### Dependency
* Python 3.11.9
* pandas
* numpy
* pytest
* unittest mock

### Quickstart
* Create a subclass of BirthProcess.
* Implement the get_deltas and to_yaml methods.

## 3. Usage
### Class: 'BirthProcess'
Description: The 'BirthProcess' class models a birth process where new individuals are added to a population based on a specified birth rate, which can be deterministic or stochastic.

### Methods:
#### get_deltas(current_state, dt:float, stochastic):
* Description: Computes the birth changes (deltas) to the epidemic state.
* Parameters:
  * current_state: The current epidemic state as a DataFrame.
  * dt: Timestep size.
  * stochastic: Whether the rule is stochastic.
* Returns: A dataframe of birth changes.

#### to_yaml():
* Description: Converts the rule to a dictionary suitable for YAML serialization.
* Returns: A dictionary representation of the rule.

## 4. Testing
### Testing Strategy
* Unit Test: Each method in the 'BirthProcess' class has corresponding unit tests to ensure correct functionality.

### Test Setup
* A dictionary representing the start state.

### Example Tests:
* test_birthprocess_initialization: Verifies that the rule is correctly initialized with start state.
* test_birthprocess_get_deltas_non_stochastic: Ensures that the correct deltas for non-stochastic senario are calculated based on the current state.
* test_birthprocess_get_deltas_stochastic: Ensures that the correct deltas for stochastic senario are calculated based on the current state.
* test_birthprocess_to_yaml: Confirms that the rule is correctly serialized back to YAML format.
