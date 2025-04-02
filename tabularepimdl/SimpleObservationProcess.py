from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
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
    unobs_state: str
    incobs_state: str
    prevobs_state: str
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

        ##first get the states that produce incident observations
        delta_incobs = current_state.loc[(current_state[self.source_col]==self.source_state) & (current_state[self.obs_col]==self.unobs_state)] #un-observed individuals with source_state

        exp_change_rate = np.exp(-dt*self.rate)
        if not stochastic:
            #subtractions
            delta_incobs["N"] = -delta_incobs["N"] * (1-exp_change_rate)
        else:
            delta_incobs["N"] = -np.random.binomial(delta_incobs["N"], 1-exp_change_rate)
            
        #additions
        tmp = delta_incobs.assign(
            N=-delta_incobs.N
        )

        tmp[self.obs_col] = self.incobs_state

        #move folks out of the incident state and into the previous state
        delta_toprev = current_state.loc[current_state[self.obs_col]==self.incobs_state].copy()
        tmp2 = delta_toprev.assign(N=-delta_toprev.N)
        delta_toprev[self.obs_col] = self.prevobs_state
        #if source_state = 'I', then following is true
        #dela_incobs = folks moved out infected and unobserved (-)
        #tmp = folks moved in infected and incident-observed (+)
        #delta_toprev = folks moved in previously-observed (+)
        #tmp2 = folks moved out incident-observed (-)
        #print('+tmp is\n', tmp)
        #print('-tmp2 is\n', tmp2)
        return(pd.concat([delta_incobs, tmp, delta_toprev, tmp2]).reset_index(drop=True)) 

    def to_yaml(self):
        rc = {
            'tabularepimdl.SimpleObservationProcess': {
                'source_col': self.source_col,
                'source_state': self.source_state,
                'rate': self.rate,
                'unobs_state': self.unobs_state,
                'incobs_state': self.incobs_state,
                'prevobs_state':self.prevobs_state,
                'stochastic':self.stochastic
            }
        }

        return rc        

