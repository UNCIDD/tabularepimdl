# WAIFWTransmission Module

## 1. Introduction
Description: This module represents transmission based on a simple "Who Acquire Infection From Whom" transmission matrix.

Purpose: The 'WAIFWTransmission' is a subclass of 'Rule' class. It provides a standard interface for converting rules to YAML format. It also offers method of calucating changes (deltas) from current state in the amount of tempsteps including both deterministic and stochastic scenarios.

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
* Create a subclass of WAIFWTransmission.
* Implement the get_deltas and to_yaml methods.

## 3. Usage
### Class: 'WAIFWTransmission'
Description: The 'WAIFWTransmission' class models a transmission process based on a simple WAIFW (who acuqire infection from whom) transmission matrix.

### Methods:
#### get_deltas(current_state, dt:float, stochastic):
* Description: Computes the changes (deltas) to the epidemic state.
* Parameters:
  * current_state: The current epidemic state as a DataFrame.
  * dt: Timestep size.
  * stochastic: Whether the rule is stochastic.
* Returns: A dataframe of state changes.

#### to_yaml():
* Description: Converts the rule to a dictionary suitable for YAML serialization.
* Returns: A dictionary representation of the rule.

## 4. Testing
### Testing Strategy
* Unit Test: Each method in the 'WAIFWTransmission' class has corresponding unit tests to ensure correct functionality.

### Test Setup
* A dummy DataFrame to simulate the state of a population with different age groups.
* A dummy WAIFW matrix to simulate the transmission rate.

### Example Tests:
* test_intialization: Verifies that the rule is correctly initialized with the setup above.
* test_len_of_categories_match_len_of_waifw: Test the number of unique categories in current_state's group_col matches the waifw matrix's length.
* test_inf_array_slicing: Test slicing and aggregation of current state.
* test_categorical_type: Test the parameter data type defined in dummy dataframe.
* test_probablity_of_infection_calculation: Test infection probability for individuals from filtered groups.
* test_deltas_calculation: Test the number of susceptible individuals (deltas).
* test_get_deltas_deterministic: Ensures that the correct deltas for deterministic senario are calculated based on the current state.
* test_get_deltas_stochastic: Ensures that the correct deltas for stochastic senario are calculated based on the current state.
* test_to_yaml: Confirms that the rule is correctly serialized back to YAML format.
