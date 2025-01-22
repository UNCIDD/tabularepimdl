# MultiStrainInfectiousProcess Module

## 1. Introduction
Description: This module provides a simple multi-strain infectious process.

Purpose: The 'MultiStrainInfectiousProcess' is a subclass of 'Rule' class. It provides a standard interface for converting rules to YAML format. It also offers method of calucating changes (deltas) from current state in the amount of tempsteps including both deterministic and stochastic scenarios.

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
* Create a subclass of MultiStrainInfectiousProcess.
* Implement the get_deltas and to_yaml methods.

## 3. Usage
### Class: 'MultiStrainInfectiousProcess'
Description: The 'MultiStrainInfectiousProcess' class models a multi-strain infection process with cross-protect involved.

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
* Unit Test: Each method in the 'MultiStrainInfectiousProcess' class has corresponding unit tests to ensure correct functionality.

### Test Setup
* A dummy DataFrame to simulate the state of a population aginst different strains.
* An array to contain transmission rates for different strains.
* A matrix to contain protection rate for intra- and inter-groups. It should be a N(strain)* N(strain) matrix.

### Example Tests:
* test_intialization: Verifies that the rule is correctly initialized with the setup above.
* test_betas_on_freq_dep: Test the beta calculation based on the boolean of freq_dep.
* test_infectious_of_strain: Test the infectious calculation of each strain type.
* test_cross_protection_multiplier: Test beta values calculation based on cross-protect matrix.
* test_beta_on_susceptibility: Test strain specific probablity of infection for susceptibility.
* test_no_coinfections: Test probablity of infectoin calculation when no coinfections happen.
* test_prI: test probablity of infectoin calculation.
* test_get_deltas_deterministic: Ensures that the correct deltas for deterministic senario are calculated based on the current state.
* test_get_deltas_stochastic: Ensures that the correct deltas for stochastic senario are calculated based on the current state.
* test_to_yaml: Confirms that the rule is correctly serialized back to YAML format.
