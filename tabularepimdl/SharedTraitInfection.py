import pandas as pd
import numpy as np
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, Field
from typing import Annotated

class SharedTraitInfection(Rule, BaseModel):

    #def __init__(self,in_beta:float, out_beta:float, inf_col, trait_col, s_st="S", i_st="I", inf_to="I", stochastic = False) -> None:
    '''!
    @param in_beta: transmission risk if trait shared.
    @param out_beta: transmission risk if trait not shared.
    @param inf_col: the column designating infection state.
    @param trait_col: the column designaing the trait.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state folks go to, assumed to be I.
    @param stochastic: whether the transition is stochastic or deterministic.
    '''
    
    in_beta: Annotated[int | float, Field(ge=0)]
    out_beta: Annotated[int | float, Field(ge=0)]
    inf_col: str
    trait_col: str
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    stochastic: bool = False
        

    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool = None) -> pd.DataFrame:
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

        ##first let's get folks who are susceptible. These are the states we will actually
        ##see deltas from.
        deltas = current_state.loc[current_state[self.inf_col]==self.s_st].copy() #extract S folks only
        
        #print('ST rule input deltas\n', deltas) #debug

        infect_only = current_state.loc[current_state[self.inf_col]==self.i_st].copy() #extract I folks only
        #print('ST rule original input infect\n', infect_only) #debug

        #Grouping records that have the same trait_col value
        infect_only = infect_only.groupby([self.trait_col], dropna=False, observed=True).agg({'N': 'sum'}).reset_index()
        #print('ST rule grouped input infect\n', infect_only) #debug

        total_infect = infect_only["N"].sum() #sum all the infected people no matter what trait it is
        
        trait_N_map = infect_only.set_index(self.trait_col)["N"] #set trait value as index, N remains to be the value for each trait
        
        #Now loop over folks in this state. 
        #There might be faster ways to do this.
        #for ind, row in deltas.iterrows():
        #    inI = current_state.loc[(current_state[self.trait_col]==row[self.trait_col]) & (current_state[self.inf_col]==self.i_st)].N.sum()
        #    outI = current_state.loc[(current_state[self.trait_col]!=row[self.trait_col]) & (current_state[self.inf_col]==self.i_st)].N.sum()
        #    prI = 1-np.power(np.exp(-dt*self.in_beta),inI)*np.power(np.exp(-dt*self.out_beta),outI)

        #print('train_N_map is\n', trait_N_map) #debug


        #A faster way to get inI, outI and prI values that deltas needs
        # Map the number of infected folks of each trait to the correponding trait in deltas
        deltas["inI"]  = deltas[self.trait_col].map(trait_N_map).fillna(0) #the number of infected folks inside each trait
        #print('deltas is\n', deltas) #debug

        deltas["outI"] = total_infect - deltas["inI"] #the number of infected folks outside each trait

        # Vectorized calculation of prI
        deltas["prI"] = 1 - np.power(np.exp(-dt*self.in_beta), deltas["inI"]) * np.power(np.exp(-dt*self.out_beta), deltas["outI"])
            
        # Update N values based on prI
        if not stochastic:
            deltas["N"] = -deltas["N"] * deltas["prI"]
        else:
            deltas["N"] = -np.random.binomial(deltas["N"],deltas["prI"])
        
        #drop temporary columns inI, outI, prI
        deltas.drop(["inI", "outI", "prI"], axis=1, inplace=True)
        
        # Update deltas and deltas_add DataFrames
        deltas_add = deltas.assign(**{self.inf_col: self.inf_to, "N": -deltas["N"]}) #folks into I
        
        rc = pd.concat([deltas, deltas_add])
        #print('ST combined deltas are\n', rc) #debug
        return rc.loc[rc.N!=0].reset_index(drop=True) #reset index for the new dataframe
    
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SharedTraitInfection': self.model_dump()
        }
        return rc