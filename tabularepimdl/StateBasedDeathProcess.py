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
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep
        """
        if stochastic is None:
            stochastic = self.stochastic

        ##first let's reduce to just the columns we need.
        deltas_temp = current_state
        #for column, state in zip(self.columns, self.states):
        #    deltas = deltas.loc[deltas[column]==state]
        #YL: It seems that the above code only keeps one row of record in deltas after for loop, is it on purpose? If want to keep all records, than adopt the following code.
        # Assuming all satisfied records are wanted based on column and state value, the following code could be used
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