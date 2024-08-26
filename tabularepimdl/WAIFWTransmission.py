from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class WAIFWTransmission(Rule):
    """!
    Rule that does transmission based on a simple WAIFW transmission matrix."""

    def __init__(self, waifw_matrix, inf_col, group_col, s_st="S", i_st="I", inf_to="I", stochastic=False) -> None:
        ##TODO: Add a frequency dependent flag?
        """!
        @param waifw_martrix the waifw matrix
        @param inf_col the column for this infectious process
        @param group_col the column where group is specified. Should have the same number of
            possible unique values as the beta matrix, and they should have an order (i.e., it 
            should be a pd.categorical)
        @param s_st the state for susceptibles, assumed to be S
        @param i_st the state for infectious, assumed to be I
        @param inf_to the state infectious folks go to, assumed to be I
        @param freq_dep is this a frequency dependent model
        @param stochastic is this rule stochastic
        """

        super().__init__()
        self.waifw_matrix = waifw_matrix
        self.inf_col = inf_col
        self.group_col = group_col
        self.s_st = s_st
        self.i_st = i_st
        self.inf_to = inf_to
        self.stochastic = stochastic

    def get_deltas(self, current_state, dt = 1.0, stochastic = None):
        """
        @param current_state, a data frame (at the moment) w/ the current epidemic state
        @param dt, the size of the timestep
        """
        if stochastic is None:
            stochastic = self.stochastic

        ##create an array of the number of infections in each group.
        inf_array = current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col)['N'].sum(numeric_only=True).values #moved ['N'] position 

        #print(pd.api.types.is_categorical_dtype(current_state[self.group_col])) #YL: it this for debugging/displaying purpose?
        print(isinstance(current_state[self.group_col].dtype, pd.CategoricalDtype)) #is_categorical_dtype is deprecated, replaced with the isinstance function
        print(current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col).sum(numeric_only=True)) #YL: it this for debugging/displaying purpose?
        print("))))") #YL: it this for debugging/displaying purpose?
        print(inf_array) #YL: it this for debugging/displaying purpose?

        #get the probability of being infected in each group
        prI = np.power(np.exp(-dt*self.waifw_matrix),inf_array)
        prI = 1-prI.prod(axis=1)

        ##get folks in susceptible states
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st]

        ##do infectious process
        if not stochastic:
            deltas = deltas.assign(N=-deltas['N']*prI[deltas[self.group_col].cat.codes])
        else:
            deltas = deltas.assign(N=-np.random.binomial(deltas['N'],prI[deltas[self.group_col].cat.codes]))

        deltas_add = deltas.assign(N=-deltas['N'])
        deltas_add[self.inf_col] = self.inf_to

        rc = pd.concat([deltas,deltas_add])
        return rc.loc[rc.N!=0].reset_index(drop=True) #reset index for the new dataframe
    
    def to_yaml(self):
        rc = {
            'tabularepimdl.WAIFWTransmission' : {
                'waifw_matrix' : self.waifw_matrix,
                'inf_col' : self.inf_col,
                'group_col' : self.group_col,
                's_st': self.s_st,
                'i_st': self.i_st,
                'inf_to': self.inf_to,
                'stochastic': self.stochastic
            }
        }
        
        return rc #added return operation





