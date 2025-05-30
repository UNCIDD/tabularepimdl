from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Annotated

class SimpleObservationProcess(Rule, BaseModel):
    '''! This rule captures a simple generic observation process where people from a particular state are
    observed to move into another state at some constant rate.'''

    #def __init__(self, source_col, source_state, obs_col, rate:float,  unobs_state="U", incobs_state="I", prevobs_state="P", stochastic = False) -> None:
    """Initialization.
    @param source_col: the column containing source_state for the observation process.
    @param source_state: the state individuals start.
    @param obs_col: the column that contains each group of individuals' observed state.
    @param rate: the number of people move from a particular state into another state per unit time.
    @param unobs_state: un-observed state.
    @param incobs_state: incident-observed state.
    @param prevobs_state: previously-observed state.
    @param stochastic: whether the transition is stochastic or deterministic.
    """
    
    source_col: str
    source_state: str
    obs_col: str
    rate: Annotated[int | float, Field(ge=0)]
    unobs_state: str = 'U'
    incobs_state: str = 'I'
    prevobs_state: str = 'P'
    stochastic: bool = False

    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool = None) -> pd.DataFrame:
        """
        @param current_state: a data frame (at the moment) w/ the current epidemic state.
        @param dt: the size of the timestep.
        @return: a DataFrame containing population changes in unobs, incobs, prevobs state.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        #variables definition
        #out_of_unobs: folks moved out unobserved (-)
        #into_incobs: folks moved in incident-observed (+)
        #out_of_incobs: folks moved out incident-observed (-)
        #into_prev: folks moved in previously-observed (+)

        #out_of_unobs supports deterministic and stochastic
        out_of_unobs = current_state.loc[(current_state[self.source_col]==self.source_state) & (current_state[self.obs_col]==self.unobs_state)].copy() #un-observed individuals with source_state

        exp_change_rate = np.exp(-dt*self.rate)
        if not stochastic:
            #subtractions
            out_of_unobs["N"] = -out_of_unobs["N"] * (1-exp_change_rate)
        else:
            out_of_unobs["N"] = -np.random.binomial(out_of_unobs["N"], 1-exp_change_rate)
            
        #additions, changes in in_ and out_ incobs and prevobs only require deterministic process
        into_incobs = out_of_unobs.assign(**{self.obs_col: self.incobs_state, "N": -out_of_unobs["N"]})
        
        #move folks out of current_state incobs state
        out_of_incobs = current_state.loc[current_state[self.obs_col]==self.incobs_state].copy()
        out_of_incobs["N"] = -out_of_incobs["N"]

        #move folks out of the incident state and into the previous state
        into_prev = current_state.loc[current_state[self.obs_col]==self.incobs_state].copy()
        into_prev[self.obs_col] = self.prevobs_state
        
        return(pd.concat([out_of_unobs, into_incobs, out_of_incobs, into_prev]).reset_index(drop=True)) 

    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SimpleObservationProcess': self.model_dump()
        }
        return rc        

