from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd


class BirthProcess(Rule):
    """!
    Represents a birth process where people are borne based
    on a birth rate based on the full poplation size."""

    def __init__(self, rate:float, start_state_sig, stochastic=False)-> None:
        """!
        Initializes the rule.

        @param rate the rate at which births happen per time step (N*rate births).
        @param state_start_sig the column values that newly borne individuals should have.
        """

        super().__init__()
        self.rate = rate

        #is their a better way to do this?
        if isinstance(start_state_sig, dict):
            start_state_sig = pd.DataFrame(start_state_sig,index=[0])
           
        self.start_state_sig = start_state_sig
        self.stochastic = stochastic

    def get_deltas(self, current_state, dt = 1.0, stochastic =None):
        if stochastic is None:
            stochastic = self.stochastic
        
        ##Just returns a delta with the start
        ##state signature and the right number of
        ##births.
        N = current_state['N'].sum()
        if not stochastic:
            births = self.start_state_sig.assign(N=N*(1-np.exp(-dt*self.rate)))
        else:
            births = self.start_state_sig.assign(N=np.random.poisson(N*(1-np.exp(-dt*self.rate))))
        return births
        
    def to_yaml(self):
        rc = {
            'tabularepimdl.BirthProcess' : {
                'rate': self.rate,
                'start_state_sig':self.start_state_sig,
                'stochastic': self.stochastic
            }
        }

        return rc