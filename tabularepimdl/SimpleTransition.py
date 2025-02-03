from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class SimpleTransition(Rule):
    """! Class is going to represent a simple transition from one state to another, 
    such that if a column has the from specified value, it creates transitions with the to
    specified value at the given rate."""

    def __init__(self, column, from_st, to_st, rate: float, stochastic=False) -> None:
        """! Initialization.

        @param column: Name of the column this rule applies to.
        @param from_st: the state that column transitions from.
        @param to_st: the state that column transitions to.
        @param rate: transition rate per unit time.
        @param stochastic: whether the transition is stochastic or deterministic.
        """

        super().__init__()
        self.column = column
        self.from_st = from_st
        self.to_st = to_st
        self.rate = rate
        self.stochastic = stochastic

    def get_deltas(self, current_state: pd.DataFrame, dt=1.0, stochastic=None):
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @return: a pandas DataFrame containing changes in from_st and to_st.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")

        if stochastic is None:
            stochastic = self.stochastic
            
        deltas = current_state.loc[current_state[self.column]==self.from_st].copy()
        #print('st rule\n') #debug
        #print('st\'s current_state is\n', current_state) #debug
        if not stochastic:
            #subtractions
            deltas["N"] = -deltas["N"] * (1-np.exp(-dt*self.rate))
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], 1-np.exp(-dt*self.rate))
        

        #additions
        tmp = deltas.copy()
        tmp["N"] = -deltas["N"]
        tmp[self.column] = self.to_st

        #print('st-rule delta is\n', deltas) #debug
        return pd.concat([deltas, tmp]).reset_index(drop=True)
    
    def __str__(self) -> str:
        return "{} --> {} at rate {}".format(self.from_st, self.to_st, self.rate)
    
    def to_yaml(self):
        rc = {
            'tabularepimdl.SimpleTransition': {
                'column': self.column,
                'from_st': self.from_st,
                'to_st': self.to_st,
                'rate': self.rate, 
                'stochastic': self.stochastic
            }
        }

        return rc