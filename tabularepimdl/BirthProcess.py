from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Annotated

class BirthProcess(Rule, BaseModel):
    """!
    Represents a birth process where people are borne based
    on a birth rate based on the full poplation size."""

    #def __init__(self, rate: float, start_state_sig, stochastic=False)-> None:
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
            start_state_sig = pd.DataFrame([start_state_sig]) #convert a dict to a dataframe
        elif isinstance(start_state_sig, pd.DataFrame):
            start_state_sig = start_state_sig.copy()
        else:
            raise ValueError ("start_state_sig must be either a dictionary or a pandas DataFrame")
        return start_state_sig
        
    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool = None) -> pd.DataFrame:
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
        rc = {
            'tabularepimdl.BirthProcess' : self.model_dump()
        }
        return rc