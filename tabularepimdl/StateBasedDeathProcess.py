from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Annotated

class StateBasedDeathProcess(Rule, BaseModel):
    """! 
    Represents a death process that takes people out of a state defined by one or more columns at some rate. """

    #def __init__ (self, columns, states, rate, stochastic = False) -> None:
    """!
    @param columns: one or more columns that we will check states against.
    @param states: the state this rate applies to for each of these columns.
    @param rate: the rate at whihc people will die from.
    @param stochastic: whether the transition is stochastic or deterministic.
    """
    columns: list[str]
    states: list[str]
    rate: Annotated[int | float, Field(ge=0)]
    stochastic: bool = False

    @field_validator("columns", "states", mode="before") #validate list type
    @classmethod
    def validate_list(cls, list_parameters, field: ValidationInfo):
        """Ensure the input is a list."""
        if not isinstance(list_parameters, list):
            raise ValueError(f"{cls.__name__} expects a list for {field.field_name}, got {type(list_parameters)}")
        return list_parameters
    
    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool = None) -> pd.DataFrame:
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep.
        @return: a DataFrame containing changes in population.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        ##first let's reduce to just the columns we need.
        #deltas_temp = current_state.copy()
        
        #all satisfied records are wanted based on column and state values
        state_mask = np.logical_or.reduce([current_state[column]==state for column, state in zip(self.columns, self.states)], axis=0)
        deltas = current_state.loc[state_mask].copy()

        exp_change_rate = np.exp(-dt*self.rate)
        if not stochastic:
            deltas["N"] = -deltas["N"] * (1 - exp_change_rate)
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], 1 - exp_change_rate)
        
        return deltas.reset_index(drop=True)

    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.StateBasedDeathProcess' : self.model_dump()
        }
        return rc #add return operation