from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class SimpleObservationProcess(Rule):
    '''! This rule captures a simple generic observation process where people from a particular state are
    observed to move into another state at some constant rate.'''

    def __init__(self, source_col, source_state, obs_col, rate:float,  unobs_state="U", incobs_state="I", prevobs_state="P", stochastic = False) -> None:
        """
        @param source_col, the column containing source_state for the observation process
        @param source_state, the state individuals start
        @param obs_col, the column that contains each group of individuals' observed state
        @param rate, the number of people move from a particular state into another state per unit time
        @param unobs_state, un-observed state
        @param incobs_state, incident-observed state
        @param prevobs_state, previously-observed state
        @param stochastic, is this rule stochastic process
        """
        super().__init__()
        self.source_col = source_col
        self.source_state = source_state
        self.obs_col = obs_col
        self.rate = rate
        self.unobs_state = unobs_state
        self.incobs_state = incobs_state
        self.prevobs_state = prevobs_state
        self.stochastic = stochastic

    def get_deltas(self, current_state, dt = 1.0, stochastic = None):
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep
        """
        if stochastic is None:
            stochastic = self.stochastic

        ##first get the states that produce incident observations
        delta_incobs = current_state.loc[(current_state[self.source_col]==self.source_state) & (current_state[self.obs_col]==self.unobs_state)] #un-observed individuals with source_state

        if not stochastic:
            #subtractions
            delta_incobs = delta_incobs.assign(
                N=-delta_incobs.N*(1-np.exp(-dt*self.rate))
                )
        else:
            delta_incobs = delta_incobs.assign(
                N = -np.random.binomial(delta_incobs.N, 1-np.exp(-dt*self.rate))
            )



        #additions
        tmp = delta_incobs.assign(
            N=-delta_incobs.N
        )

        tmp[self.obs_col] = self.incobs_state

        #move folks out of the incident state and into the previous state
        delta_toprev = current_state.loc[current_state[self.obs_col]==self.incobs_state].copy()
        tmp2 = delta_toprev.assign(N=-delta_toprev.N)
        delta_toprev[self.obs_col] = self.prevobs_state
        #if source_state = 'I', then following is true
        #dela_incobs = folks moved out infected and unobserved (-)
        #tmp = folks moved in infected and incident-observed (+)
        #delta_toprev = folks moved in previously-observed (+)
        #tmp2 = folks moved out incident-observed (-)
        return(pd.concat([delta_incobs, tmp, delta_toprev, tmp2]).reset_index(drop=True)) 

    def to_yaml(self):
        rc = {
            'tabularepimdl.SimpleObservationProcess': {
                'source_col': self.source_col,
                'source_state': self.source_state,
                'rate': self.rate,
                'unobs_state': self.unobs_state,
                'incobs_state': self.incobs_state,
                'prevobs_state':self.prevobs_state,
                'stochastic':self.stochastic
            }
        }

        return rc        

