from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class SimpleTransition(Rule):
    """! Class is going to represent a simple transition from one state to another, 
    such that if a column has the from specified value it creates transitions with the to
    specified value at the given rate."""

    def __init__(self, column, from_st, to_st, rate:float, stochastic = False) -> None:
        """! Initialization.

        @param column the name of the column this should be applied to.
        @param from_st the state that column should have if this is going to be applied.
        @param to_st the state folks should move to
        @param rate, the number of people move from a particular state into another state per unit time
        @param stochastic, is this rule stochastic process
        """

        super().__init__()
        self.column = column
        self.from_st = from_st
        self.to_st = to_st
        self.rate = rate #the transition rate of from_st to to_st, e.g. aging transition rate
        self.stochastic = stochastic

    def get_deltas(self, current_state,dt=1.0, stochastic = None):
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep
        """
        if stochastic is None:
            stochastic = self.stochastic
            
        deltas = current_state.loc[current_state[self.column]==self.from_st]
        #print('st rule\n') #debug
        #print('st\'s current_state is\n', current_state) #debug
        if not stochastic:
            #subtractions
            deltas = deltas.assign(
                    N=-deltas.N*(1-np.exp(-dt*self.rate))
                )
        else:
            deltas = deltas.assign(
                N = -np.random.binomial(deltas.N, 1-np.exp(-dt*self.rate))
            )

        #additions
        tmp = deltas.assign(
            N=-deltas.N
        )

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