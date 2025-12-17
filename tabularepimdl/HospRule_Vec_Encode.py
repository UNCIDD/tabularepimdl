import numpy as np
from pydantic import BaseModel, Field, ConfigDict, ValidationInfo, field_validator, model_validator, PrivateAttr

from tabularepimdl.Rule import Rule

class HospRule(Rule, BaseModel):
    '''This rule takes multiple columns. You have some risk of hospitalization if infected from any column,
    but that probability is reduced if you are recovered in any column. We will additionally track which strain you were
    hospitalized with. Only tracking total hospitalizations.'''

    
    ''' Presume that hosp cols is of the same length as the secondary, and they have corresponding indices. '''
    inf_cols: list[str] = Field(description = "the strain columns for infection state.")
    hosp_cols: list[str] = Field(description = "hospitalization columns.")
    inf_cols_categories: list[str] = Field("all the infection categories used in epidemics.")
    hosp_cols_categories: list[str] = Field("all the hospitalization categories used in epidemics.")
    prim_hrate: float = Field(ge=0, description = "chance of being hospitalized from a primary infection.")
    sec_hrate: float = Field(ge=0, description = "chance of being hospitalized from a secondary infection.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")

    #potential model_post_init(self, _)
    
    def get_deltas(self, current_state, dt=1.0, stochastic=None):
        if stochastic is None:
            stochastic = self.stochastic
        
        mask_strain = (current_state[['Strain1', 'Strain2', 'Strain3']] == 'I') #boolean mask (dataframe) for 'I' in each Strain

        current_state['match_ind'] = np.where(mask_strain.any(axis=1), mask_strain.idxmax(axis=1).map({'Strain1': 0, 'Strain2': 1, 'Strain3': 2}), -1) #If people are infected with a Strain, then match_ind captures this Strain's column numerical index. If people are not infected with any Strain, then match_ind=-1

        current_state['hosp_rate'] = np.where((current_state[['Strain1', 'Strain2', 'Strain3']]=='R').any(axis=1), self.sec_hrate, self.prim_hrate) #If people are recovered from a Strain, then hosp_rate=sec_hrate, otherwise hosp_rate=prim_hrate

        deltas_with_infect = current_state[current_state['match_ind'] != -1] #remove uninfected records

        if not stochastic:
            #print('deterministic') #debug
            delta_decobs = deltas_with_infect.assign(N=-deltas_with_infect.N*(1-np.exp(-dt*deltas_with_infect.hosp_rate)))
        else:
            #print('stochastic') #debug
            delta_decobs = deltas_with_infect.assign(N= -np.random.binomial(deltas_with_infect.N, 1-np.exp(-dt*deltas_with_infect.hosp_rate)))

        delta_decobs = delta_decobs.reset_index(drop=True) #decayed observed?

        delta_incobs = delta_decobs.assign(N=-delta_decobs.N) #incidently observed?

        mask_hosp = np.zeros((delta_incobs.shape[0], len(self.hosp_cols)), dtype=bool) #empty mask (matrix) for the hospitalization

        mask_hosp[np.arange(len(delta_incobs)), delta_incobs['match_ind']] = True #set True for the mask based on match_ind's values

        delta_incobs[self.hosp_cols] = np.where(mask_hosp, 'H', delta_incobs[self.hosp_cols]) #Hosp column='H' where mask hospitalization is True, otherwise column value stays as-is.

        deltas = pd.concat([delta_incobs, delta_decobs])
        deltas = deltas.drop(columns=['match_ind', 'hosp_rate']) #remove intermediate columns
        deltas = deltas.reset_index(drop=True)
        
        return deltas

                
    def to_yaml(self):
        #no serialization since this is a one off rule
        pass    