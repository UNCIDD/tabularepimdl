# SimpleInfection Module

## 1. Introduction
Description: This module represents a simple infection process where people in one state are infected by people in anothe state with a probability.

Purpose: The 'SimpleInfection' is a subclass of 'Rule' class. It provides a standard interface for converting rules to YAML format. It also offers method of calucating changes (deltas) from current state in the amount of tempsteps including both deterministic and stochastic scenarios.

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
* Create a subclass of SimpleInfection.
* Implement the get_deltas and to_yaml methods.

## 3. Usage
### Class: 'SimpleInfection'
Description: The 'SimpleInfection' class models a simple infectious disease process where individuals in a population can transition from a susceptible state to an infectious state based on a transmission probability.

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
* Unit Test: Each method in the 'SimpleInfection' class has corresponding unit tests to ensure correct functionality.

### Test Setup
* A dummy DataFrame to simulate the state of a population.

### Example Tests:
* test_intialization: Verifies that the rule is correctly initialized with the setup above.
* test_beta_freq_true: Test beta value when freq_dep is True.
* test_beta_freq_flase: Test beta value when freq_dep is False.
* test_infectious_calculation: Test the calculation of the total number of infectious individuals.
* test_deltas_calculation: Test the number of susceptible individuals (deltas).
* test_get_deltas_deterministic: Ensures that the correct deltas for deterministic senario are calculated based on the current state.
* test_get_deltas_stochastic: Ensures that the correct deltas for stochastic senario are calculated based on the current state.
* test_to_yaml: Confirms that the rule is correctly serialized back to YAML format.
