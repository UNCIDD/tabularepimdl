import pandas as pd
import numpy as np
from tabularepimdl.Rule import Rule


class SharedTraitInfection(Rule):

    def __init__(self,in_beta:float, out_beta:float,
                 inf_col, trait_col, s_st="S", i_st="I",
                 inf_to="I", stochastic = False) -> None:
        '''!
        @param in_beta transmission risk if trait shared
        @param out_beta transmission risk if trait not shared
        @param inf_col the column designating infection state
        @param trait_col the column designaing the trait
        @param s_st the suseptible state
        @param i_st the infectoius state
        @param inf_to the state to move infectous folks to
        @param stochastic is this rule stochastic if not forced by the epi model.'''
        self.in_beta = in_beta
        self.out_beta = out_beta
        self.inf_col = inf_col
        self.trait_col = trait_col
        self.s_st = s_st
        self.i_st = i_st
        self.inf_to = inf_to
        self.stochastic = stochastic
        

    def get_deltas(self, current_state, dt = 1.0, stochastic=None):
        if stochastic is None:
            stochastic = self.stochastic

        ##first let's get folks who are susceptible. These are the states we will actually
        ##see deltas from.
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st].copy()
        deltas_add = deltas.copy()

        #Now loop over folks in this state. 
        #There might be faster ways to do this.
        for ind, row in deltas.iterrows():
            inI = current_state.loc[(current_state[self.trait_col]==row[self.trait_col]) & (current_state[self.inf_col]==self.i_st)].N.sum()
            outI = current_state.loc[(current_state[self.trait_col]!=row[self.trait_col]) & (current_state[self.inf_col]==self.i_st)].N.sum()
            prI = 1-np.power(np.exp(-dt*self.in_beta),inI)*np.power(np.exp(-dt*self.out_beta),outI)

            if not stochastic:
                row['N'] = row['N']*prI
            else:
                row['N'] = np.random.binomial(row['N'],prI)
            deltas.at[ind,'N'] = -row['N']
            deltas_add.at[ind,'N'] = row['N']
        
        deltas_add[self.inf_col] = self.inf_to

        rc = pd.concat([deltas,deltas_add])

        return rc.loc[rc.N!=0]
    
    def to_yaml(self):
        rc = {
            'tabularepimdl.SimpleObservationProcess': {
                "in_beta": self.in_beta,
                "out_beta": self.out_beta,
                "inf_col": self.inf_col,
                "trait_col": self.trait_col,
                "s_st": self.s_st,
                "i_st":self.i_st,
                "inf_to":self.inf_to,
                "stochastic": self.stochastic
            }
        }
        return rc