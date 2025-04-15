from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from numba import njit



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
        self.waifw_matrix = np.asarray(waifw_matrix).T #transpose the input matrix
        self.inf_col = inf_col
        self.group_col = group_col
        self.s_st = s_st
        self.i_st = i_st
        self.inf_to = inf_to
        self.stochastic = stochastic
    
    @staticmethod    
    @njit
    def compute_infection_array(present_cat_codes, weights, num_of_categories):
        
        #Optimized function using numba to compute the number of infected individuals per group.
        
        inf_array = np.zeros(num_of_categories, dtype=np.float64)  # Initialize array with zeros
        for i in range(len(present_cat_codes)):
            inf_array[present_cat_codes[i]] = inf_array[present_cat_codes[i]] + weights[i]
        return inf_array
    
    @staticmethod
    @njit
    def compute_prI(waifw_matrix, inf_array, dt):
        
        #Computes probabilities of infection using numba.
        #Equivalent to: prI = 1 - np.power(np.exp(-dt*self.waifw_matrix), inf_array)
        matrix_size = len(waifw_matrix)
        prI = np.ones(matrix_size, dtype=np.float64) #initialize prI with 1s

        expo = np.exp(-dt * waifw_matrix)
        infection_power = np.power(expo, inf_array)
        
        for i in range(matrix_size):
            for j in range(matrix_size):
                prI[i] = prI[i] * infection_power[i, j]
            prI[i] = 1 - prI[i]
        return prI
        


    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool = None) -> pd.DataFrame:
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
        if not isinstance(current_state[self.group_col].dtype, pd.CategoricalDtype):
            current_state[self.group_col]=pd.Categorical(current_state[self.group_col])

        #Check if the number of unique categories in current_state's group_col matches waifw matrix's size
        if len(current_state[self.group_col].cat.categories) != len(self.waifw_matrix):
            raise ValueError(f"Mismatch between the number of unique categories and WAIFW matrix size. "
                             f"Expected {len(self.waifw_matrix)} categories, but found {len(current_state[self.group_col].cat.categories)}. "
                             f"Categories: {current_state[self.group_col].cat.categories}"
                            )

        ##create an array for the total number of infections in each unique group. Only records with i_st are sumed, other records's N are filled with 0.
        #inf_array = current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col, observed=False)['N'].sum(numeric_only=True).values #moved ['N'] position #groupby approach
        #inf_array = np.bincount(current_state.loc[current_state[self.inf_col]==self.i_st, self.group_col].cat.codes, current_state.loc[current_state[self.inf_col]==self.i_st, "N"], minlength=len(current_state[self.group_col].cat.categories)) #np.bincount approach
        
        num_of_categories = len(current_state[self.group_col].cat.categories)
        present_category_codes = current_state[self.group_col].cat.codes.to_numpy()
        infected_mask = current_state[self.inf_col] == self.i_st
        infected_group_codes = present_category_codes[infected_mask.to_numpy()]
        infected_weights = current_state.loc[infected_mask, "N"].to_numpy()
        
        inf_array = self.compute_infection_array(infected_group_codes, infected_weights, num_of_categories) #numba approach

        #print('inf_array is\n', inf_array) #debug

        #get the probability of being infected in each unique group
        #print('-dt*matrix\n', -dt*self.waifw_matrix)
        #print('exponential\n', np.exp(-dt*self.waifw_matrix))
        
        #prI = np.power(np.exp(-dt*self.waifw_matrix), inf_array) #whole matrix multiplication approach
        #print('powered prI is\n', prI) #debug
        #prI = 1-prI.prod(axis=1)
        #print('1-prI prod\n', prI) #debug

        prI = self.compute_prI(self.waifw_matrix, inf_array, dt) #numba approach


        ##get folks in susceptible states which link to all unique groups
        is_susceptible = current_state[self.inf_col] == self.s_st
        deltas = current_state[is_susceptible].copy()
        #print('deltas is\n', deltas, '\n') #debug
        #print('prI codes are\n', prI[deltas[self.group_col].cat.codes], '\n') #debug

        ##do infectious process, getting the number of individuals who get infected from susceptible status
        susceptible_group_codes = present_category_codes[is_susceptible.to_numpy()]
        prI_per_group = prI[susceptible_group_codes]

        if not stochastic:
            deltas["N"] = -deltas["N"] * prI_per_group
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], prI_per_group)

        #print('deltas after infection process:\n', deltas)
        #deltas_add = deltas.assign(N=-deltas['N'])
        #deltas_add[self.inf_col] = self.inf_to
        deltas_add = deltas.assign(**{self.inf_col: self.inf_to, "N": -deltas["N"]})
        
        rc = pd.concat([deltas,deltas_add])
        return rc.loc[rc["N"] != 0].reset_index(drop=True) #reset index for the new dataframe
    
    
    
    def to_yaml(self):
        rc = {
            'tabularepimdl.WAIFWTransmission' : {
                'waifw_matrix' : self.waifw_matrix.T, #when write rule waifw matrix to yaml file, should it be transposed back as initial order?
                'inf_col' : self.inf_col,
                'group_col' : self.group_col,
                's_st': self.s_st,
                'i_st': self.i_st,
                'inf_to': self.inf_to,
                'stochastic': self.stochastic
            }
        }
        
        return rc #added return operation


    


