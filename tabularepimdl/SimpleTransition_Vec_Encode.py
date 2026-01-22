from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule


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
    @param column_categories: the categories used for attribute column. E.g column_categories = ['0 to 4', '5 to 9', '10-14'].
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    @param _from_code: encoded from_st.
    @param _to_code: encoded to_st.
    """
    column: str
    from_st: str
    to_st: str
    rate: Annotated[float, Field(ge=0)]
    stochastic: bool = False
    column_categories: list[str]
    infstate_compartments: list[str]

    _from_code: int | None = PrivateAttr(default=None)
    _to_code: int | None = PrivateAttr(default=None)

    def model_post_init(self, _):
        if self.column.lower() == 'infstate': #column is infection state
            infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
            self._from_code = infstate_to_int.get(self.from_st)
            self._to_code = infstate_to_int.get(self.to_st)
        else: #column is other attribute
            col_cat_to_int =  {s: i for i, s in enumerate(sorted(self.column_categories))}  #encode column strings to integers {'0 to 4': 0, '5 to 9': 1}
            self._from_code = col_cat_to_int.get(self.from_st)
            self._to_code = col_cat_to_int.get(self.to_st)

    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        @param current_state: a numpy array (at the moment) representing the current epidemic state. Must include population values (e.g. 'N' values).
        @param dt: size of the timestep.
        @param: Add additional parameters...stochastic, dt
        @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState' : 0, 'N': 1}
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. E.g. result_buffer = np.empty((2 * count, ncols), dtype=current_state.dtype)
        return: an array containing changes in from_st and to_st.
        """
        if stochastic is None:
            stochastic = self.stochastic
        
        infstate_idx = col_idx_map[self.column]
        n_idx = col_idx_map['N']
        #print('input array\n', current_state)
        #print('infstate idx:', infstate_idx, 'n idx:', n_idx)
        #print('_from_code:', self._from_code, '_to_code:', self._to_code)

        # Fast boolean mask for matching from-state
        mask = current_state[:, infstate_idx] == self._from_code
        #print('mask:', mask)
        if not np.any(mask):
            #print('empty return')
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
            
            
        # Get indices where mask is true (faster than slicing twice)
        #from_row_idxs = np.flatnonzero(mask) #redundant code
        selected_from = current_state[mask, :]
        #print('selected_from\n', selected_from)
        N = selected_from[:, n_idx] #equivalent: current_state[from_row_idxs, n_idx]
        #print('from_code N:', N)

        # Compute transition amounts
        rate_const = 1 - np.exp(-dt * self.rate)
        #print('rate_const:', rate_const)

        if stochastic:
            changed_N = -np.random.binomial(N.astype(np.int32), rate_const)
        else:
            changed_N = -N * rate_const
        #print('change_N:', changed_N)

        count = selected_from.shape[0]#len(from_row_idxs)
        #print('count:', count)
        #ncols = current_state.shape[1] #move num of columns out of class for now

        # Preallocate result: 2 rows per event (from & to)
        #result = np.empty((2 * count, ncols), dtype=current_state.dtype) #move pre-allocation out of class for now

        # Fill 'from' rows
        #print('result_buffer:\n', result_buffer)
        result_buffer[:count, :] = selected_from #equivalent: self._from_code
        result_buffer[:count, n_idx] = changed_N  #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = selected_from
        result_buffer[count:2*count, infstate_idx] = self._to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        return result_buffer[:2*count, :]

    def __str__(self) -> str:
        return f"SimpleTransition_Vec_Encode: {self.from_st} --> {self.to_st} at rate {self.rate}"
    
    def to_yaml(self) -> dict:
        rc = {
            'tabularepimdl.SimpleTransition_Vec_Encode': self.model_dump()
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