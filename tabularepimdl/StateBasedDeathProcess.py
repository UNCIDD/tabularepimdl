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
        
        if isinstance(columns, list):
            self.columns = columns
        else:
            raise TypeError("Parameter 'columns' needs to be in list type, e.g. columns=['column1', 'column2'].")
        
        if isinstance(states, list):
            self.states = states
        else:
            raise TypeError("Parameter 'states' needs to be in list type, e.g. states=['state1', 'state2'].")
        
        self.rate = rate
        self.stochastic = stochastic

    
    def get_deltas(self, current_state, dt = 1.0, stochastic=None):
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep
        """
        if stochastic is None:
            stochastic = self.stochastic

        ##first let's reduce to just the columns we need.
        deltas_temp = current_state.copy()
        
        #all satisfied records are wanted based on column and state values
        deltas = pd.DataFrame()
        for column, state in zip(self.columns, self.states):
            filtered_deltas = deltas_temp.loc[deltas_temp[column]==state]
            deltas = pd.concat([deltas, filtered_deltas])

        if not stochastic:
            deltas = deltas.assign(N=-deltas['N']*(1-np.exp(-dt*self.rate)))
        else:
            deltas = deltas.assign(N=-np.random.binomial(deltas['N'],1-np.exp(-dt*self.rate)))
        
        return deltas.reset_index(drop=True)

    def to_yaml(self):
        rc = {
            'tabularepimdl.StateBasedDeathProcess' : {
                'columns' : self.columns,
                'states' : self.states,
                'rate' : self.rate,
                'stochastic' : self.stochastic
            }
        }
        
        return rc #add return operation