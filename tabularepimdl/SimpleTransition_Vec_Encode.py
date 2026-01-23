from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule


class SimpleTransition_Vec_Encode(Rule, BaseModel):
    """! 
    Rule represents a simple transition from one state to another, such that if a column has
    the from specified value, it creates transitions with the to specified value at the given rate.

    Attributes:
        column: name of the column this rule applies to.
        from_st: the state that column transitions from.
        to_st: the state that column transitions to.
        rate: transition rate per unit time.
        stochastic: whether the process is stochastic or deterministic.
        column_categories: all the categories the column should have if the column is not for infstate. E.g column_categories = ['0 to 4', '5 to 9', '10-14'].
        infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    """
    
    column: str = Field(description = "name of the column this rule applies to.")
    from_st: str = Field(description = "the state that column transitions from.")
    to_st: str = Field(description = "the state that column transitions to.")
    rate: float = Field(ge=0, description = "the state that column transitions to.")
    stochastic: bool = Field(default=False, description="whether the process is stochastic or deterministic.")
    column_categories: list[str] = Field(description = "all the categories the column should have.")
    infstate_compartments: list[str] = Field(description = "the infection compartments used in epidemics.")

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
            raise ValueError(f"Missing required columns in current_state: {required_columns}.")
        
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