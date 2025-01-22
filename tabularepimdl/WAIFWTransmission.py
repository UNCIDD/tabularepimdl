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

        #question: since group_col needs to be categorical type, do we want to convert current_state[group_col] to pd.categorical in this rule? 
        #Instead of doing the conversion in model applications such as AgingPopulation?

        #quesiton and fix: convert group_col to categorical type first, so groupby observed=False generate full list of array values
        current_state[self.group_col]=pd.Categorical(current_state[self.group_col])

        #Check if the number of unique categories in current_state's group_col matches waifw matrix's length
        if len(current_state[self.group_col].cat.codes.unique()) != len(self.waifw_matrix):
            raise ValueError(f"The number of unique categoeries in 'current_state' group column should be equal to the length of 'waifw_matrix'. "
                             f"However, the current number of unique categoeries in 'current_state' group column is ({len(current_state[self.group_col].cat.codes.unique())}), "
                             f"'waifw_matrix' current length is ({len(self.waifw_matrix)})."
                            )

        ##create an array for the total number of infections in each unique group. Only records with i_st are sumed, other records's N are filled with 0.
        inf_array = current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col, observed=False)['N'].sum(numeric_only=True).values #moved ['N'] position 

        #print('is it category?', isinstance(current_state[self.group_col].dtype, pd.CategoricalDtype)) #is_categorical_dtype is deprecated, replaced with the isinstance function
        #print(current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col).sum(numeric_only=True)) #debug
        #print("))))") #debug
        #print('inf_array is', inf_array) #debug

        #get the probability of being infected in each unique group
        prI = np.power(np.exp(-dt*self.waifw_matrix),inf_array)
        #print('powered prI is', prI) #debug
        prI = 1-prI.prod(axis=1)
        #print('1-prI prod', prI) #debug

        ##get folks in susceptible states which link to all unique groups
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st]
        #print('deltas is', deltas, '\n') #debug
        #print('prI codes are', prI[deltas[self.group_col].cat.codes], '\n') #debug

        ##do infectious process, getting the number of individuals who get infected from susceptible status
        if not stochastic:
            deltas = deltas.assign(N=-deltas['N']*prI[deltas[self.group_col].cat.codes])
        else:
            deltas = deltas.assign(N=-np.random.binomial(deltas['N'],prI[deltas[self.group_col].cat.codes]))

        #print('deltas after infection process:', deltas)
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





