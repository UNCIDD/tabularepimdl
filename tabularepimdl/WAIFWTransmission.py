from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd

class WAIFWTransmission(Rule):
    """!
    Rule that does transmission based on a simple WAIFW transmission matrix."""

    def __init__(self, waifw_matrix, inf_col, group_col, s_st="S", i_st="I", inf_to="I", stochastic=False) -> None:
        ##TODO: Add a frequency dependent flag?
        """!Initialization.
        @param waifw_martrix: the waifw matrix
        @param inf_col: the column for this infectious process
        @param group_col: the column where group is specified. Should have the same number of
            possible unique values as the beta matrix, and they should have an order (i.e., it 
            should be a pd.categorical)
        @param s_st: the state for susceptibles, assumed to be S
        @param i_st: the state for infectious, assumed to be I
        @param inf_to: the state infectious folks go to, assumed to be I
        @param freq_dep: is this a frequency dependent model
        @param stochastic: is this rule stochastic
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
        @param current_state: a data frame (at the moment) w/ the current epidemic state.
        @param dt: the size of the timestep.
        @return: a DataFrame containing changes in population infection.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        #question: since group_col needs to be categorical type, do we want to convert current_state[group_col] to pd.categorical in this rule? 
        #Instead of doing the conversion in model applications such as AgingPopulation?

        #quesiton and fix: convert group_col to categorical type first, so groupby observed=False generate full list of array values
        current_state[self.group_col]=pd.Categorical(current_state[self.group_col])

        #Check if the number of unique categories in current_state's group_col matches waifw matrix's size
        if len(current_state[self.group_col].cat.categories) != len(self.waifw_matrix):
            raise ValueError(f"The number of unique categories in 'current_state' group column should be equal to the size of 'waifw_matrix'. "
                             f"However, the current number of unique categories in 'current_state' group column is ({len(current_state[self.group_col].cat.categories)}), "
                             f"'waifw_matrix' current size is ({len(self.waifw_matrix)})."
                            )

        ##create an array for the total number of infections in each unique group. Only records with i_st are sumed, other records's N are filled with 0.
        #inf_array = current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col, observed=False)['N'].sum(numeric_only=True).values #moved ['N'] position 
        inf_array = np.bincount(current_state.loc[current_state[self.inf_col]==self.i_st, self.group_col].cat.codes, current_state.loc[current_state[self.inf_col]==self.i_st, "N"], minlength=len(current_state[self.group_col].cat.categories))
        print('inf_array is\n', inf_array) #debug

        #get the probability of being infected in each unique group
        print('-dt*matrix\n', -dt*self.waifw_matrix)
        print('exponential\n', np.exp(-dt*self.waifw_matrix))
        
        prI = np.power(np.exp(-dt*self.waifw_matrix), inf_array)
        print('powered prI is\n', prI) #debug
        
        prI = 1-prI.prod(axis=1)
        print('1-prI prod\n', prI) #debug

        ##get folks in susceptible states which link to all unique groups
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st].copy()
        print('deltas is\n', deltas, '\n') #debug
        print('prI codes are\n', prI[deltas[self.group_col].cat.codes], '\n') #debug

        ##do infectious process, getting the number of individuals who get infected from susceptible status
        prI_per_group = prI[deltas[self.group_col].cat.codes]
        if not stochastic:
            deltas["N"] = -deltas["N"] * prI_per_group
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], prI_per_group)

        print('deltas after infection process:\n', deltas)
        deltas_add = deltas.assign(**{self.inf_col: self.inf_to, "N": -deltas["N"]})
        
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





