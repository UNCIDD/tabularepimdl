from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd


class StateBasedDeathProcess(Rule):
    """! 
    Represents a death process that takes people out
    of a state defined by one or more columns at some 
    rate. """

    def __init__ (self, columns, states, rate, stochastic = False) -> None:
        """!
        @param columns one or more columns that we will check states against
        @param states the state this rate applies to for each of these columns
        @param rate the rate at whihc people will die from this
        @param stochastic is this a stochastic rule
        """
        super().__init__()
        self.columns = columns
        self.states = states
        self.rate = rate
        self.stochastic = stochastic

    
    def get_deltas(self, current_state, dt = 1.0, stochastic=None):
        if stochastic is None:
            stochastic = self.stochastic

        ##first let's reduce to just the columns we need.
        deltas = current_state
        for column, state in zip(self.columns, self.states):
            deltas = deltas.loc[deltas[column]==state]
        
        if not stochastic:
            deltas = deltas.assign(N=-deltas['N']*(1-np.exp(-dt*self.rate)))
        else:
            deltas = deltas.assign(N=-np.random.binomial(deltas['N'],1-np.exp(-dt*self.rate)))
        
        return deltas

    def to_yaml(self):
        rc = {
            'tabularepimdl.StateBasedDeathProcess' : {
                'columns' : self.columns,
                'states' : self.states,
                'rate' : self.rate,
                'stochastic' : self.stochastic
            }
        }
        