from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, field_validator, PrivateAttr

from tabularepimdl.Rule import Rule


class SharedTraitInfection_Vec_Encode(Rule, BaseModel):
    """!
    Rule that does transmission based on if a trait is shared across different populations.
    
    Attributes:
        inf_col: the infection state column for this infectious process.
        in_beta: transmission rate if trait shared.
        out_beta: transmission rrate if trait not shared.
        trait_col: the trait column shared by different populations.
        trait_col_all_categories: all the categories the trait column should have.
        s_st: the state for susceptibles, assumed to be S.
        i_st: the state for infectious, assumed to be I.
        inf_to: the state susceptible populations go to, assumed to be I.
        stochastic: whether the process is stochastic or deterministic.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """

    
    inf_col: str = Field(description = "the infection state column for this infectious process.")
    in_beta: float = Field(ge=0, description = "transmission rate if trait shared.")
    out_beta: float = Field(ge=0, description = "transmission rate if trait not shared.")
    trait_col: str = Field(description = "the trait column shared by different populations.")
    trait_col_all_categories: list[str | int] = Field(description = "all the categories the trait column should have.")
    s_st: str = Field(default="S", description = "the state for susceptibles.")
    i_st: str = Field(default="I", description = "the state for infectious.")
    inf_to: str = Field(default="I", description = "the state susceptible population go to.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    infstate_compartments: list[str] = Field("the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _i_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)
    _trait_col_all_categories_code: list[int] | None = PrivateAttr(default=None)

    @field_validator("trait_col_all_categories", mode='before')
    @classmethod
    def validate_group_col_all_categories(cls, category_vals):
        if not category_vals:
            raise ValueError("The trait_col_all_categories values must not be empty.")
        
        first_element_type = type(category_vals[0])
        for item in category_vals[1:]:
            if type(item) != first_element_type:
                raise ValueError(
                    f"All elements in group_col_all_categories must be of the same datatype. "
                    f"Found both {first_element_type.__name__} and {type(item).__name__}."
                )
        return category_vals
    
    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
        self._s_code = infstate_to_int.get(self.s_st)
        self._i_code = infstate_to_int.get(self.i_st)
        self._inf_to_code = infstate_to_int.get(self.inf_to)

        self.trait_col_all_categories = sorted(self.trait_col_all_categories) #sort the trait_col's all categories
        self._trait_col_all_categories_code = [i for i, v in enumerate(self.trait_col_all_categories)] #encode each category, keeping numbers only



    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float =1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Compute the population deltas for the current state at a given time step.

        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): mapping of column names to their index positions. e.g. {'N':0, 'InfState':1, 'Hosp':2}
            result_buffer (np.ndarray): A pre-allocated array that will be populated with the computed deltas. This array is modified in-place and returned.
            dt (float): The size of the time step. Defaults to 1.0.
            stochastic (bool, optional): Whether to apply stochastic modeling. If `None`, the class-level `self.stochastic` attribute is used.
        
        Returns:
            np.ndarray: A NumPy structured array containing the population deltas.

        Raises:
            ValueError: If the column `'N'` is missing in `current_state`.
        """
        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        infstate_idx = col_idx_map[self.inf_col]
        trait_col_idx = col_idx_map[self.trait_col]
        n_idx = col_idx_map['N']

        if len(set(self.trait_col_all_categories)) < len(set(current_state[:, trait_col_idx])):
            raise ValueError(f"Number of elements in trait_col_all_categories is less than the number of categories of input data, please check trait_col_all_categories and input data.")

        s_mask = current_state[:, infstate_idx] == self._s_code
        if not np.any(s_mask):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
        
        s_row_idxs = np.flatnonzero(s_mask)
        selected_s = current_state[s_row_idxs, :]
        N_susceptible = selected_s[:, n_idx]
        #print('N susceptible:', N_susceptible) #debug

        i_mask = current_state[:, infstate_idx] == self._i_code
        i_row_idxs = np.flatnonzero(i_mask)
        infect_only = current_state[i_row_idxs, :]

        total_infect = np.sum(infect_only[:, n_idx])

        infect_N_lookup_dict = dict(zip(infect_only[:, trait_col_idx], infect_only[:, n_idx]))#combine trait and N as lookup dict

        in_I_mapped = np.array([infect_N_lookup_dict.get(val, 0) for val in selected_s[:, trait_col_idx]]) #shared traits' N values
        out_I_mapped = total_infect - in_I_mapped #non-shared traits' N values

        exp_in_beta = np.exp(-dt * self.in_beta)
        exp_out_beta = np.exp(-dt * self.out_beta)

        prI = 1 - np.power(exp_in_beta, in_I_mapped) * np.power(exp_out_beta, out_I_mapped)

        if stochastic:
            changed_N = -np.random.binomial(N_susceptible.astype(np.int32), prI)
        else:
            changed_N = -N_susceptible * prI

        count = len(N_susceptible)
        #print('count:', count)
        # Fill 'from' rows
        
        result_buffer[:count, :] = selected_s
        result_buffer[:count, n_idx] = changed_N  #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = selected_s
        result_buffer[count:2*count, infstate_idx] = self._inf_to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        filtered_result_buffer = result_buffer[:2*count, :]
        result = filtered_result_buffer[filtered_result_buffer[:, n_idx] != 0] #remove rows with N=0
        
        return result


    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SharedTraitInfection_Vec_Encode': self.model_dump()
        }
        return rc
    

    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]: 
        return self.infstate_compartments
    
    #set up a property to return all the required categories used in trait_col
    @property
    def trait_col_all(self) -> list[str]: 
        return self.trait_col_all_categories