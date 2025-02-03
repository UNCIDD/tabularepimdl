from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class SimpleInfection(Rule):
    """! 
    Represents a simple infection process where people in one column are infected by people
    in a given state in that same column with a probability."""

    def __init__(self, beta:float, column, s_st="S", i_st="I", inf_to="I", freq_dep=True, stochastic=False) -> None:
        """!
        Initialization. 

        @param beta: the transmission parameter. 
        @param column: name of the column this rule applies to.
        @param s_st: the state for susceptibles, assumed to be S.
        @param i_st: the state for infectious, assumed to be I.
        @param inf_to: the state infectious folks go to, assumed to be I.
        @param freq_dep: whether this model is a frequency dependent model.
        @param stochastic: whether the transition is stochastic or deterministic.
        """
        super().__init__() 
        self.beta = beta
        self.column = column
        self.s_st = s_st
        self.i_st = i_st
        self.inf_to = inf_to
        self.freq_dep = freq_dep
        self.stochastic = stochastic

    def get_deltas(self, current_state: pd.DataFrame, dt=1.0, stochastic=None):
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
                beta = self.beta/(current_state['N'].sum())
            else:
                beta = self.beta
        else:
            beta = self.beta

        infectious = current_state.loc[current_state[self.column]==self.i_st, 'N'].sum()

        deltas = current_state.loc[current_state[self.column]==self.s_st].copy()

        if not stochastic:
            deltas["N"] = -deltas["N"]*(1 - np.power(np.exp(-dt*beta), infectious))
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], 1 - np.power(np.exp(-dt*beta),infectious))
        
        tmp = deltas.copy()
        tmp["N"] = -deltas["N"]
        tmp[self.column] = self.inf_to

        return pd.concat([deltas, tmp]).reset_index(drop=True)
        
        
    def to_yaml(self):
        rc = {
            'tabularepimdl.SimpleInfection': {
                'beta': self.beta,
                'column': self.column,
                's_st': self.s_st,
                'i_st': self.i_st,
                'inf_to': self.inf_to,
                'freq_dep':self.freq_dep,
                'stochastic':self.stochastic
            }
        }

        return rc        
