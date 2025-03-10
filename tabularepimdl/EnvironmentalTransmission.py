import pandas as pd
import numpy as np
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, Field
from typing import Annotated

class EnvironmentalTransmission(Rule, BaseModel):

    #def __init__(self, in_beta: float, inf_col: str, trait_col: str, s_st: str = "S", i_st: str = "I", inf_to: str = "I", stochastic: bool = False) -> None:
    '''!Initialization.
    @param in_beta: transmission risk if trait shared.
    @param inf_col: the column designating infection state.
    @param trait_col: the column designaing the trait.
    @param s_st: the state for susceptibles, assumed to be S
    @param i_st: the state for infectious, assumed to be I
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    '''
    in_beta: Annotated[int | float, Field(ge=0)]
    inf_col: str
    trait_col:str
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    stochastic: bool = False
        
    def get_deltas(self, current_state: pd.DataFrame, dt: int | float =1.0, stochastic: bool = None) -> pd.DataFrame:
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @return: a pandas DataFrame containing changes in s_st and inf_to.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        current_state["N"] = current_state["N"].astype(np.float64) #converting column N to float type

        ##first let's get folks who are susceptible. These are the states we will actually
        ##see deltas from.
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st].copy() #extract S folks only
        deltas_add = deltas.copy()
        #print('ST rule input deltas\n', deltas) #debug

       
        # Vectorized calculation of prI
        deltas["prI"] = 1 - np.exp(-dt*self.in_beta)
            
        # Update N values based on prI, folks out of S
        if not stochastic:
            deltas["N"] = -deltas["N"] * deltas["prI"]    
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], deltas["prI"])
        
        #drop temporary columns inI, outI, prI
        deltas.drop(["prI"], axis=1, inplace=True)
        
        #Update deltas_add DataFrames, folks into I
        deltas_add = deltas_add.assign(**{self.inf_col: self.inf_to, "N": -deltas["N"]})
        #deltas_add["N"] = -deltas["N"]
        #deltas_add[self.inf_col] = self.inf_to 

        rc = pd.concat([deltas,deltas_add])
        #print('ST combined deltas are\n', rc) #debug
        return rc.loc[rc["N"]!=0].reset_index(drop=True) #reset index for the new dataframe
    
    def to_yaml(self) -> dict:
        rc = {
            'tabularepimdl.EnvironmentalTransmission': self.model_dump()
        }
        return rc