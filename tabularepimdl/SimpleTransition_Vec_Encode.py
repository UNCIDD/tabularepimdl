from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from typing import Annotated, Dict

#Vectorization of Simple Transition
class SimpleTransition_Vec_Encode(Rule, BaseModel):
    """! Class is going to represent a simple transition from one state to another, 
    such that if a column has the from specified value, it creates transitions with the to
    specified value at the given rate."""

    """! Initialization.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    @param _from_code: encoded from_st.
    @param _to_code: encoded to_st.
    """
    column: str
    from_st: str
    to_st: str
    rate: Annotated[float, Field(ge=0)]
    stochastic: bool = False
    infstate_compartments: list[str]

    _from_code: int = PrivateAttr(default=None)
    _to_code: int = PrivateAttr(default=None)

    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
        self._from_code = infstate_to_int.get(self.from_st)
        self._to_code = infstate_to_int.get(self.to_st)
        

    def get_deltas(self, current_state: np.ndarray, col_idx_map: Dict[str, int] = None, result_buffer: np.ndarray = None, dt: float = 1.0, stochastic: bool = None) -> np.ndarray:
        """
        @param current_state: a numpy array (at the moment) representing the current epidemic state. Must include population values (e.g. 'N' values).
        @param dt: size of the timestep.
        @param: Add additional parameters...
        @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState' : 0, 'N': 1}
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. E.g. result_buffer = np.empty((2 * count, ncols), dtype=current_state.dtype)
        return: an array containing changes in from_st and to_st.
        """
        if stochastic is None:
            stochastic = self.stochastic
        
        infstate_idx = col_idx_map[self.column]
        n_idx = col_idx_map['N']
        print('input array\n', current_state)
        print('_from_code:', self._from_code, '_to_code:', self._to_code)

        # Fast boolean mask for matching from-state
        mask = current_state[:, infstate_idx] == self._from_code
        if not np.any(mask):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
            
        # Get indices where mask is true (faster than slicing twice)
        from_row_idxs = np.flatnonzero(mask)
        selected_from = current_state[from_row_idxs, :]
        print('selected_from\n', selected_from)
        N = selected_from[:, n_idx] #equivalent: current_state[from_row_idxs, n_idx]
        print('from_code N:', N)

        # Compute transition amounts
        rate_const = 1 - np.exp(-dt * self.rate)
        print('rate_const:', rate_const)

        if stochastic:
            changed_N = -np.random.binomial(N.astype(np.int32), rate_const)
        else:
            changed_N = -N * rate_const
        print('change_N:', changed_N)

        count = len(from_row_idxs)
        #ncols = current_state.shape[1] #move num of columns out of class for now

        # Preallocate result: 2 rows per event (from & to)
        #result = np.empty((2 * count, ncols), dtype=current_state.dtype) #move pre-allocation out of class for now

        # Fill 'from' rows
        result_buffer[:count, :] = selected_from #equivalent: self._from_code
        result_buffer[:count, n_idx] = changed_N  #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = selected_from
        result_buffer[count:2*count, infstate_idx] = self._to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        return result_buffer[:2*count, :]

    def __str__(self) -> str:
        return "SimpleTransition_Vec_Encode: {} --> {} at rate {}".format(self.from_st, self.to_st, self.rate)
    
    def to_yaml(self) -> dict:
        rc = {
            'tabularepimdl.SimpleTransition': self.model_dump()
        }

        return rc
    
    @property
    def source_states(self) -> list[str]:
        return [self.from_st]

    @property
    def target_states(self) -> list[str]:
        return [self.to_st]