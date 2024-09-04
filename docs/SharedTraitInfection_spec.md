# SharedTraitInfection Module

## 1. Introduction
Description: This module models infection process for individuals with shared trait.

Purpose: The 'SharedTraitInfection' is a subclass of 'Rule' class. It provides a standard interface for converting rules to YAML format. It also offers method of calucating changes (deltas) from current state in the amount of tempsteps including both deterministic and stochastic scenarios.

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
* Create a subclass of SharedTraitInfection.
* Implement the get_deltas and to_yaml methods.

## 3. Usage
### Class: 'SharedTraitInfection'
Description: The 'SharedTraitInfection' class models infection process for individuals with shared trait.

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
* Unit Test: Each method in the 'SharedTraitInfection' class has corresponding unit tests to ensure correct functionality.

### Test Setup
* A dummy DataFrame to simulate the state of a population aginst different traits.

### Example Tests:
* test_intialization: Verifies that the rule is correctly initialized with the setup above.
* test_detlas_calculation: Test the number of incident observations.
* test_inI_outI_prI_deltas: Test the values of parameters inI, outI and prI, and N population values in deltas.
* test_get_deltas_deterministic: Ensures that the correct deltas for deterministic senario are calculated based on the current state.
* test_get_deltas_stochastic: Ensures that the correct deltas for stochastic senario are calculated based on the current state.
* test_to_yaml: Confirms that the rule is correctly serialized back to YAML format.
