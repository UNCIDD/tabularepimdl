import pandas as pd
import numpy as np
from tabularepimdl.Rule import Rule


class EnvironmentalTransmission(Rule):

    def __init__(self, in_beta: float, inf_col: str, trait_col: str, s_st: str = "S", i_st: str = "I", inf_to: str = "I", stochastic: bool = False) -> None:
        '''!
        Initialization.

        @param in_beta: transmission risk if trait shared.
        @param inf_col: the column designating infection state.
        @param trait_col: the column designaing the trait.
        @param s_st: the state for susceptibles, assumed to be S
        @param i_st: the state for infectious, assumed to be I
        @param inf_to: the state infectious folks go to, assumed to be I.
        @param stochastic: whether the process is stochastic or deterministic.
        '''
        super().__init__()
        self.in_beta = in_beta
        self.inf_col = inf_col
        self.trait_col = trait_col
        self.s_st = s_st
        self.i_st = i_st
        self.inf_to = inf_to
        self.stochastic = stochastic
        
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
            
        # Update N values based on prI
        if not stochastic:
            deltas["N"] = deltas["N"] * deltas["prI"]    
        else:
            deltas["N"] = np.random.binomial(deltas["N"], deltas["prI"])
        
        #drop temporary columns inI, outI, prI
        deltas.drop(["prI"], axis=1, inplace=True)
        
        # Update deltas and deltas_add DataFrames
        deltas["N"] = -deltas["N"] #folks out of S
        deltas_add["N"] = -deltas["N"]
        
        deltas_add[self.inf_col] = self.inf_to #folks into I

        rc = pd.concat([deltas,deltas_add])
        #print('ST combined deltas are\n', rc) #debug
        return rc.loc[rc["N"]!=0].reset_index(drop=True) #reset index for the new dataframe
    
    def to_yaml(self) -> dict:
        rc = {
            'tabularepimdl.SharedTraitInfection': {
                'in_beta': self.in_beta,
                'out_beta': self.out_beta,
                'inf_col': self.inf_col,
                'trait_col': self.trait_col,
                's_st': self.s_st,
                'i_st': self.i_st,
                'inf_to': self.inf_to,
                'stochastic': self.stochastic
            }
        }
        return rc