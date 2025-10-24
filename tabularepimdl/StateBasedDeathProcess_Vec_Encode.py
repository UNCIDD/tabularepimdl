from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator, PrivateAttr

from tabularepimdl.Rule import Rule


class StateBasedDeathProcess_Vec_Encode(Rule, BaseModel):
    """! 
    Represents a death process that takes people out of a state defined by one or more columns at some rate.

    Attributes:
        column: single column that we will check states against. Changing columns to column based on the AgingPopulation example.
        all_states: all the states of single column. All the state values are required as input, otherwise the encoding logic cannot assign the correct mapping.
        target_states: targeted states of single column.
        rate: the rate at whihc people will die from.
        stochastic: whether the process is stochastic or deterministic.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']

    Examples:
    data = pd.DataFrame({
        'N': [10, 20, 30],
        'InfState': ['I', 'R', 'S'], #column1
        'Hosp':     ['P1', 'U1', 'U2'] #column2
    })

    column = 'InfState' #one column only
    all_states = ['I', 'R', 'S']
    target_states = ['I', 'S'] #values from single column

    selected_data = pd.DataFrame({
        'N': [10, 30],
        'InfState': ['I', 'S'], #column1
        'Hosp':     ['P1', 'U2'] #column2
    })
    """

    column: str = Field(description = "one column that we will check states against.")
    all_states: list[str] = Field(description = "all the states of single column.")
    target_states: list[str] = Field(description = "targeted states to be processed of single column.")
    rate: float = Field(ge=0, description = "the rate at whihc people will die from.")
    stochastic: bool = Field(default=False, description = "whether the transition is stochastic or deterministic.")
    infstate_compartments: list[str] = Field("the infection compartments used in epidemics.")

    #_columns_code: list[str] | None = PrivateAttr(default_factory=list) #not needed given it is single column
    _states_code: list[str] | None = PrivateAttr(default_factory=list)

    @field_validator("all_states", "target_states", mode="before") #validate list type
    @classmethod
    def validate_list(cls, list_parameters, field: ValidationInfo):
        """Ensure the input is a list."""
        if not isinstance(list_parameters, list):
            raise ValueError(f"{cls.__name__} expects a list for {field.field_name}, received {type(list_parameters)}")
        return list_parameters
    
    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))} #for InfState column mapping only
        states_sorted = sorted(self.all_states)
        colstate_to_int = {s: i for i, s in enumerate(states_sorted)} #for the single column mapping
        self._states_code = [colstate_to_int[state] for state in states_sorted if state in self.target_states] #encoded column states
        
        
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
        """
        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        col_idx = col_idx_map[self.column]
        n_idx = col_idx_map['N']

        states_mask = np.isin(current_state[:, col_idx], self._states_code) #all rows match single column's states code
        #print('state mask', states_mask)

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


    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.StateBasedDeathProcess' : self.model_dump()
        }
        return rc
    

    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]: 
        return self.infstate_compartments