from collections.abc import Iterable
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tabularepimdl.Rule import Rule


class BirthProcess(Rule, BaseModel):
    """!
    Represents a birth process where people are borne based
    on a birth rate based on the full poplation size."""

    """!Initialization.
    @param rate: birth rate at per time step (where N*rate births occur).
    @param state_start_sig: initial state configuration for new births.
    @param stochastic: whether the transition is stochastic or deterministic.
    """

    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
      
    rate: Annotated[int | float, Field(ge=0)]
    start_state_sig: dict | pd.DataFrame
    stochastic: bool = False
    
    @field_validator("start_state_sig") #validate if start_state_sig is an instance of dict or dataframe
    @classmethod
    def validate_start_state_sig(cls, start_state_sig):
        if isinstance(start_state_sig, dict):
            key, value = next(iter(start_state_sig.items())) #obtain the first item from the dictionary
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)): #value contains two or more elements
                start_state_sig = pd.DataFrame(start_state_sig)
            else: #value contains single element
                start_state_sig = pd.DataFrame([start_state_sig])
        elif isinstance(start_state_sig, pd.DataFrame):
            start_state_sig = start_state_sig.copy()
        else:
            raise ValueError (f"start_state_sig must be either a dictionary or a pandas DataFrame, received {type(start_state_sig)}.")
        return start_state_sig
        
    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool | None = None) -> pd.DataFrame:
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @return: a pandas DataFrame containing changes in population birth.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic
        
        #returns a delta with the start_state_signature and the right number of births.
        N = current_state["N"].sum()
        births = self.start_state_sig.copy()
        birth_value = N * (1 - np.exp(-dt*self.rate))

        if not stochastic:
            births["N"] = birth_value
        else:
            births["N"] = np.random.poisson(birth_value)  #get a random single outcome value by poisson distribution
        return births
        
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        if isinstance(self.start_state_sig, pd.DataFrame): #convert dataframe to dict before saving to yaml data
            if len(self.start_state_sig) == 1: #if single row in dataframe
                start_state_sig_dict = self.start_state_sig.iloc[0].to_dict()
            else:
                start_state_sig_dict = self.start_state_sig.to_dict(orient='list')#if multiple-rows in dataframe
        else:
            start_state_sig_dict = self.start_state_sig #keep the original dict data

        rc = {
            'tabularepimdl.BirthProcess' : {
                'rate': self.rate,
                'start_state_sig': start_state_sig_dict,
                'stochastic': self.stochastic
            }
        }
        return rc
    
    def to_dict(self) -> dict:
        """to accomodate the to_dict() addition in base Rule"""
        pass