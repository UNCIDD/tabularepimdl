import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, field_validator, PrivateAttr
from tabularepimdl._types.constrained_types import UniqueNonEmptyStrList
from tabularepimdl._validators.domain_attribute_validators import domain_membership_validator

from tabularepimdl.Rule import Rule


class StateBasedDeathProcess_Vec_Encode(Rule, BaseModel):
    """
    Rule that represents a death process that takes people out of a state defined by one or more columns at some rate.

    Attributes:
        column: single column that we will check states against. Changing columns to column based on the AgingPopulation example.
        column_states: all the states of the single column. All the state values are required as input, otherwise the encoding logic cannot assign the correct mapping.
        target_states: targeted states of single column.
        rate: the rate at whihc people will die from.
        stochastic: whether the process is stochastic or deterministic.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """

    column: str = Field(description = "one column that we will check states against.")
    column_states: UniqueNonEmptyStrList = Field(description = "all the states of the single column.")
    target_states: UniqueNonEmptyStrList = Field(description = "targeted states to be processed of single column.")
    rate: float = Field(ge=0, description = "the rate at whihc people will die from.")
    stochastic: bool = Field(default=False, description = "whether the transition is stochastic or deterministic.")
    infstate_compartments: UniqueNonEmptyStrList = Field(description = "the infection compartments used in epidemics.")

    #_columns_code: list[str] | None = PrivateAttr(default_factory=list) #not needed given it is single column
    _states_code: list[str] | None = PrivateAttr(default_factory=list)

    _check_domain_membership = domain_membership_validator(
            attribute_fields = ("target_states"),
            domain_fields = ("column_states")
        )
    
    def model_post_init(self, _):
        """
        Encode the input states based on each column's attribute values.
        
        Returns:
            Numerical values of encoded column states.
        """
        states_sorted = sorted(self.column_states)
        colstate_to_int = {s: i for i, s in enumerate(states_sorted)} #for the single column mapping
        self._states_code = [colstate_to_int[state] for state in states_sorted if state in self.target_states] #encoded column states

    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required infection compartments if the `column` takes 'infstate' value.
        """
        return self.infstate_compartments
    
    #set up a property to return all the required states used in general column
    @property
    def column_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required categories if the `column` takes other string values.
        """
        return self.column_states
    
    @property
    def expansion_factor(self) -> int:
        """Maximum number of rows this rule can return per input rows."""
        return max(len(self.column_states)*len(self.infstate_compartments), len(self.target_states)*len(self.infstate_compartments))
        
    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float =1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Compute the population deltas for the current state at a given time step.

        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): mapping of column names to their index positions. e.g. {'N':0, 'InfState':1, 'Hosp':2}
            result_buffer (np.ndarray): A pre-allocated array that will be populated with the computed deltas. This array is modified in-place and returned.
            dt (float): The size of the time step. Defaults to 1.0.
            stochastic (bool, optional): Whether to apply stochastic modeling. If `None`, the class-level `self.stochastic` attribute is used.
        
        Returns:
            np.ndarray: A NumPy structured array containing the population deltas.

        Raises:
            ValueError: If the column `'N'` is missing in `current_state`.
        
        Examples: # in pandas DataFrame formt instead of Numpy array
            import pandas as pd
            data = pd.DataFrame({
            'N': [10, 20, 30],
            'InfState': ['I', 'R', 'S'], #column1
            'Hosp':     ['P1', 'U1', 'U2'] #column2
            })

            column = 'InfState' #one column only
            column_states = ['I', 'R', 'S']
            target_states = ['I', 'S'] #values from single column

            process_object = StateBasedDeathProcess_Vec_Encode(column, column_states, target_states, ...)
        
        selected_from = project_object.get_deltas(current_state, ...)
            N   InfState    Hosp
        0  10          I      P1
        1  30          S      U2
        """
        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}.")
        
        if stochastic is None:
            stochastic = self.stochastic

        col_idx = col_idx_map[self.column]
        n_idx = col_idx_map['N']

        states_mask = np.isin(current_state[:, col_idx], self._states_code) #all rows match single column's states code
        #print('state mask', states_mask)
        #need to add an empty array return condition if state_mask is empty

        #all satisfied records are wanted based on column and state values
        selected_from = current_state[states_mask]
        N = selected_from[:, n_idx]
        #print('selected from', selected_from)

        rate_const = 1 - np.exp(-dt * self.rate)

        if stochastic:
            changed_N = -np.random.binomial(N.astype(np.int32), rate_const)
        else:
            changed_N = -N * rate_const
        
        count = len(selected_from)
        #print('count:', count)
        
        #Fill selected rows with changed_N values
        result_buffer[:count, :] = selected_from 
        result_buffer[:count, n_idx] = changed_N #update column N with changed_N (negative value)

        return result_buffer[:count, :]


    def to_dict(self) -> dict:
        """
        Save the rule's attributes and their associated values to a dictionary.
        
        Returns:
            Rule attributes in a dictionary.
        """
        rc = {
            'tabularepimdl.StateBasedDeathProcess_Vec_Encode' : self.model_dump()
        }
        return rc