from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule


#Vectorization of Simple Infection
class SimpleInfection_Vec_Encode(Rule, BaseModel):
    """!Class represents a simple infection process where people in one column are infected by people
    in a given state in that same column with a probability."""

    """! Initialization.
    @param beta: the transmission parameter. 
    @param column: Name of the column this rule applies to.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param freq_dep: whether this model is a frequency dependent model.
    @param stochastic: whether the process is stochastic or deterministic.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    @param _s_code: encoded s_st.
    @param _i_code: encoded i_st.
    @para _inf_to_code: encoded infected to state.
    """
    beta: Annotated[float, Field(ge=0)]
    column: str
    s_st: str
    i_st: str
    inf_to: str
    freq_dep: bool = True
    stochastic: bool = False
    infstate_compartments: list[str]

    _s_code: int | None = PrivateAttr(default=None)
    _i_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)

    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))} #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
        self._s_code = infstate_to_int.get(self.s_st)
        self._i_code = infstate_to_int.get(self.i_st)
        self._inf_to_code = infstate_to_int.get(self.i_st) #inf_to code might be defined in infstate_compartments as well if needed
        

    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        @param current_state: a numpy array (at the moment) representing the current epidemic state. Must include population values (e.g. 'N' values).
        @param dt: size of the timestep.
        @param: Add additional parameters...
        @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState' : 0, 'N': 1}
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. E.g. result_buffer = np.empty((2 * count, ncols), dtype=current_state.dtype)
        return: an array containing changes in s_st and inf_to.
        """
        beta: float

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
    
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SimpleInfection': self.model_dump()
        }
        return rc
    
    @property
    def source_states(self):
        return [self.s_st]

    @property
    def target_states(self):
        return [self.inf_to]