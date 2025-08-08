from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from typing import Annotated, Dict

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
    @param _from_code: encoded from_st.
    @param _to_code: encoded to_st.
    @para _inf_to_code: encoded infected to state.
    """
    beta: Annotated[int | float, Field(ge=0)]
    column: str
    s_st: str
    i_st: str
    inf_to: str
    freq_dep: bool = True
    stochastic: bool = False
    infstate_compartments: list[str]

    _from_code: int = PrivateAttr(default=None)
    _to_code: int = PrivateAttr(default=None)
    _infto_code: int = PrivateAttr(default=None)

    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(self.infstate_compartments)} #encoding -> {'S': 0, 'I': 1, 'R': 2}
        self._s_code = infstate_to_int.get(self.s_st)
        self._i_code = infstate_to_int.get(self.i_st)
        self._inf_to_code = infstate_to_int.get(self.i_st) #inf_to code might be defined in infstate_compartments as well if needed
        

    def get_deltas(self, current_state: np.ndarray, data_col: Dict[str, int] = None, result_buffer: np.ndarray = None, dt: float = 1.0) -> np.ndarray:

        infstate_idx = data_col[self.column]
        n_idx = data_col['N']
        
        total_population = np.sum(current_state[:, n_idx])

        if total_population != 0:
            if self.freq_dep:
                beta = self.beta/total_population
            else:
                beta = self.beta
        else:
            beta = self.beta

        # i_state
        mask_i = current_state[:, infstate_idx] == self._i_code
        if not np.any(mask_i):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
            
        i_row_idxs = np.flatnonzero(mask_i)
        N_infectious_sum = np.sum(current_state[i_row_idxs, n_idx]) #number of infected individuals

        # s_state
        mask_s = current_state[:, infstate_idx] == self._s_code
        if not np.any(mask_s):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
        
        s_row_idxs = np.flatnonzero(mask_s)
        N_susceptible = current_state[s_row_idxs, n_idx]

        # Compute transition amounts
        rate_const = 1 - np.power(np.exp(-dt*beta), N_infectious_sum)
        if self.stochastic:
            changed_N = -np.random.binomial(N_susceptible.astype(np.int32), rate_const)
        else:
            changed_N = -N_susceptible * rate_const

        count = len(s_row_idxs)
        
        # Fill 'from' rows
        result_buffer[:count] = self._s_code #current_state[idxs]
        result_buffer[:count, n_idx] = changed_N  # delta_N is negative

        # Fill 'to' rows
        result_buffer[count:2 * count] = self._inf_to_code
        result_buffer[count:2 * count, n_idx] = -changed_N  # reverse delta_N

        return result_buffer[:2*count]
    
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SimpleInfection': self.model_dump()
        }
        return rc