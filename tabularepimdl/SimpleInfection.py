from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Annotated

class SimpleInfection(Rule, BaseModel):
    """!Class represents a simple infection process where people in one column are infected by people
    in a given state in that same column with a probability."""

    #def __init__(self, beta:float, column, s_st="S", i_st="I", inf_to="I", freq_dep=True, stochastic=False) -> None:
    """!Initialization. 
    @param beta: the transmission parameter. 
    @param column: name of the column this rule applies to.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param freq_dep: whether this model is a frequency dependent model.
    @param stochastic: whether the process is stochastic or deterministic.
    """
    beta: Annotated[int | float, Field(ge=0)]
    column: str
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    freq_dep: bool = True
    stochastic: bool = False

    def get_deltas(self, current_state: pd.DataFrame, dt: int | float =1.0, stochastic: bool = None) -> pd.DataFrame:
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @return: a pandas DataFrame containing changes in s_st and inf_to.
        """
        
        if stochastic is None:
            stochastic = self.stochastic

        total_population = current_state["N"].sum()

        if total_population != 0:
            if self.freq_dep:
                beta = self.beta/total_population
            else:
                beta = self.beta
        else:
            beta = self.beta

        infectious = current_state.loc[current_state[self.column]==self.i_st, "N"].sum()

        deltas = current_state.loc[current_state[self.column]==self.s_st].copy()

        if not stochastic:
            deltas["N"] = -deltas["N"] * (1 - np.power(np.exp(-dt*beta), infectious))
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], 1 - np.power(np.exp(-dt*beta),infectious))
        
        #deltas_add = deltas.copy()
        deltas_add = deltas.assign(**{self.column: self.inf_to, "N": -deltas["N"]})
        
        return pd.concat([deltas, deltas_add]).reset_index(drop=True)
        
        
    def to_yaml(self):
        rc = {
            'tabularepimdl.SimpleInfection': self.model_dump()
        }

        return rc        
