from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule


class SimpleInfection_Vec_Encode(Rule, BaseModel):
    """!
    Rule represents a simple infection process where people in one column are infected by people
    in a given state in that same column with a probability.

    Attributes:
        beta: the transmission parameter. 
        column: name of the column this rule applies to.
        s_st: the state for susceptibles, assumed to be S.
        i_st: the state for infectious, assumed to be I.
        inf_to: the state infectious folks go to, assumed to be I.
        freq_dep: whether this model is a frequency dependent model.
        stochastic: whether the process is stochastic or deterministic.
        column_categories: all the categories the column should have.
        infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    """

    beta: float = Field(ge=0, description = "the transmission parameter.")
    column: str = Field(description = "name of the column this rule applies to.")
    s_st: str = Field(description = "the state for susceptibles, assumed to be S.")
    i_st: str = Field(description = "the state for infectious, assumed to be I.")
    inf_to: str = Field(description = "the state infectious folks go to, assumed to be I.")
    freq_dep: bool = Field(default=True, description = "whether this model is a frequency dependent model.")
    stochastic: bool = Field(default=False, description="whether the process is stochastic or deterministic.")
    column_categories: list[str] = Field(description = "all the categories the column should have.")
    infstate_compartments: list[str] = Field(description = "the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _i_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)

    def model_post_init(self, _):
        if self.column.lower() == 'infstate': #column is infection state
            infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))} #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
            self._s_code = infstate_to_int.get(self.s_st)
            self._i_code = infstate_to_int.get(self.i_st)
            self._inf_to_code = infstate_to_int.get(self.i_st) #inf_to code might be defined in infstate_compartments as well if needed
        else: #column is other attribute
            col_cat_to_int =  {s: i for i, s in enumerate(sorted(self.column_categories))}  #encode infstate strings to integers {'0 to 4': 0, '5 to 9': 1}
            self._s_code = col_cat_to_int.get(self.s_st)
            self._i_code = col_cat_to_int.get(self.i_st)
            self._inf_to_code = col_cat_to_int.get(self.i_st)
        

    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
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

        beta: float

        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}.")

        if stochastic is None:
            stochastic = self.stochastic
        
        infstate_idx = col_idx_map[self.column]
        n_idx = col_idx_map['N']
        #print('input array\n', current_state)

        total_population: float = np.sum(current_state[:, n_idx])
        #print('total population:', total_population)

        if total_population != 0:
            if self.freq_dep:
                beta = self.beta/total_population
            else:
                beta = self.beta
        else:
            beta = self.beta
        #print('beta:', beta)

        # i_state
        mask_i = current_state[:, infstate_idx] == self._i_code
        if not np.any(mask_i):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
            
        i_row_idxs = np.flatnonzero(mask_i)
        N_infectious_sum: float = np.sum(current_state[i_row_idxs, n_idx]) #number of infected individuals
        #print('N_infectious_sum:', N_infectious_sum)

        # s_state
        mask_s = current_state[:, infstate_idx] == self._s_code
        if not np.any(mask_s):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
        
        s_row_idxs = np.flatnonzero(mask_s)
        selected_s = current_state[s_row_idxs, :]
        N_susceptible = selected_s[:, n_idx] #equivalent: current_state[s_row_idxs, n_idx]
        #print('N_susceptible:', N_susceptible)

        # Compute transition amounts
        rate_const = 1 - np.power(np.exp(-dt*beta), N_infectious_sum)
        #print('rate_const:', rate_const)
        if stochastic:
            changed_N = -np.random.binomial(N_susceptible.astype(np.int32), rate_const)
        else:
            changed_N = -N_susceptible * rate_const
        #print('change_N:', changed_N)

        count = len(s_row_idxs)
        
        # Fill 'from' rows
        result_buffer[:count, :] = selected_s #equivalent: self._s_code
        result_buffer[:count, n_idx] = changed_N #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = selected_s 
        result_buffer[count:2*count, infstate_idx] = self._inf_to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        return result_buffer[:2*count, :]
    
    def __str__(self) -> str:
        return f"SimpleInfection_Vec_Encode: {self.s_st} --> {self.inf_to} at rate {self.beta}"
    
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SimpleInfection_Vec_Encode': self.model_dump()
        }
        return rc
    
    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]: 
        return self.infstate_compartments
    
    #set up a property to return all the required categories used in general column
    @property
    def column_all(self) -> list[str]: 
        return self.column_categories