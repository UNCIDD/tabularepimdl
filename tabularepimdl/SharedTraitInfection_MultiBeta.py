import pandas as pd
import numpy as np
from tabularepimdl.Rule import Rule


class SharedTraitInfection_MultiBeta(Rule):

    def __init__(self, in_beta: dict[str, float], out_beta: dict[str, float],
                 inf_col: str, trait_col: str, s_st="S", i_st="I",
                 inf_to="I", stochastic: bool = False) -> None:
        '''!
        Initialization.

        @param in_beta: transmission risk if trait shared.
        @param out_beta transmission risk if trait not shared.
        @param inf_col: the column designating infection state.
        @param trait_col: the column designaing the trait.
        @param s_st: the state for susceptibles, assumed to be S
        @param i_st: the state for infectious, assumed to be I
        @param inf_to: the state infectious folks go to, assumed to be I.
        @param stochastic: whether the infection is stochastic or deterministic.
        '''
        super().__init__()
        self.in_beta = in_beta
        self.out_beta = out_beta
        self.inf_col = inf_col
        self.trait_col = trait_col
        self.s_st = s_st
        self.i_st = i_st
        self.inf_to = inf_to
        self.stochastic = stochastic
        
    def get_deltas(self, current_state: pd.DataFrame, dt=1.0, stochastic=None):
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @return: a pandas DataFrame containing changes in s_st and inf_to.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        current_state["N"] = current_state["N"].astype(np.float64) #converting column N to float type
                
        #map each trait to their corresponding in_beta values (e.g. folks share the same trait)
        current_state["in_beta_values"] = current_state[self.trait_col].map(self.in_beta).fillna(0)
        print("current_state with in_beta_values is\n", current_state)

        #Map out_beta values dynamically based on group interactions
        #Assuming out_beta contains keys that follow the pattern: 'Group1Group2' (e.g., 'AB', 'BX')
        def map_out_beta(row):
            group = row[self.trait_col]
            print('group is ', group)
            other_groups = [ g for g in current_state[self.trait_col].unique() ]
            print('other_groups is ', other_groups)
            out_betas = [self.out_beta.get(f"{group}{other_group}", 0) for other_group in other_groups]
            print('out_betas is ', out_betas)
            return np.sum(out_betas) #currently using simple sum to calculate each household's out_beta
        
        #Apply the get_out_beta function to each row in current_state
        current_state["out_beta_values"] = current_state.apply(map_out_beta, axis=1)
        print('after sum of out betas, df is\n', current_state)

        ##current_state has all beta values mapped, let's get folks who are susceptible. These are the states we will actually see deltas from.
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st].copy() #extract S folks only
        
        #print('ST rule input deltas\n', deltas) #debug

        infect_only = current_state.loc[current_state[self.inf_col]==self.i_st].copy() #extract I folks only
        #print('ST rule original input infect\n', infect_only) #debug

        #Grouping records that have the same trait_col value
        infect_only = infect_only.groupby([self.trait_col], dropna=False, observed=True).agg({'N': 'sum'}).reset_index()
        #print('ST rule grouped input infect\n', infect_only) #debug

        total_infect = infect_only["N"].sum() #sum all the infected people no matter what trait it is
        
        trait_N_map = infect_only.set_index(self.trait_col)["N"] #set trait value as index, N remains to be the value for each trait
        #print('train_N_map is\n', trait_N_map) #debug

        # Map the number of infected folks of each trait to the correponding trait in deltas
        deltas["inI"]  = deltas[self.trait_col].map(trait_N_map).fillna(0) #the number of infected folks inside each trait
        #print('deltas is\n', deltas) #debug

        deltas["outI"] = total_infect - deltas['inI'] #the number of infected folks outside each trait
        print('deltas with InI and outI is\n', deltas) #debug
              
        
        # Vectorized calculation of prI
        deltas["prI"] = 1 - np.power(np.exp(-dt*deltas["in_beta_values"]), deltas["inI"]) * np.power(np.exp(-dt*deltas["out_beta_values"]), deltas["outI"])
        print('deltas with prI is\n', deltas) #debug
             
        # Update N values based on prI
        if not stochastic:
            deltas["N"] = deltas["N"] * deltas["prI"]    
        else:
            deltas["N"] = np.random.binomial(deltas["N"],deltas["prI"])
        
        #drop temporary columns in_beta_values, out_beta_values, inI, outI, prI
        #deltas.drop(["in_beta_values", "out_beta_values", "inI", "outI", "prI"], axis=1, inplace=True)
        
        # Update deltas and deltas_add DataFrames
        deltas["N"] = -deltas["N"] #folks out of S
        
        deltas_add = deltas.copy()
        deltas_add["N"] = -deltas["N"]
        
        deltas_add[self.inf_col] = self.inf_to #folks into I

        rc = pd.concat([deltas,deltas_add])
        #print('ST combined deltas are\n', rc) #debug
        return rc.loc[rc.N!=0].reset_index(drop=True) #reset index for the new dataframe
    
    def to_yaml(self):
        rc = {
            'tabularepimdl.SharedTraitInfection': {
                'in_beta': self.in_beta,
                'out_beta': self.out_beta,
                'inf_col': self.inf_col,
                'trait_col': self.trait_col,
                's_st': self.s_st,
                'i_st': self.i_st,
                'inf_to': self.inf_to,
                'stochastic': self.stochastic
            }
        }
        return rc