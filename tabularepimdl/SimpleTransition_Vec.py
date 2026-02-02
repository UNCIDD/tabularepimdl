from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule


#Vectorization of Simple Transition
class SimpleTransition_Vec(Rule, BaseModel):
    """! Class is going to represent a simple transition from one state to another, 
    such that if a column has the from specified value, it creates transitions with the to
    specified value at the given rate."""

    """! Initialization.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    """
    column: str
    from_st: str
    to_st: str
    rate: Annotated[int | float, Field(ge=0)]
    stochastic: bool = False
    
    # Internal buffers for performance
    _count_initial: int = PrivateAttr(default=None)
    _buffers: dict = PrivateAttr(default_factory=dict)

    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool | None = None) -> pd.DataFrame:
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        return: a pandas DataFrame containing changes in from_st and to_st.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        mask_s = current_state[self.column].values == self.from_st
        if not np.any(mask_s): #Handle edge case when no from_st rows are filtered
            return pd.DataFrame(columns=current_state.columns)
        
        selected = current_state.loc[mask_s]
        N_values = selected["N"].to_numpy()
        count = len(N_values)
       
       
        # Precompute rate constant
        rate_const = 1 - np.exp(-dt * self.rate)

        # Vectorized delta computation
        if not stochastic:
            deltas_N = -N_values * rate_const
        else:
            deltas_N = -np.random.binomial(N_values.astype(int), rate_const)

        # Initialize or resize buffers as needed
        if self._count_initial is None or count > self._count_initial:
            self._count_initial = count
            self._buffers.clear()
            for col in current_state.columns:
                col_dtype = current_state[col].dtype
                if isinstance(col_dtype, pd.CategoricalDtype):
                    dtype = object
                elif col == "N":
                    dtype = float
                else:
                    dtype = col_dtype

                self._buffers[col] = np.empty(2 * count, dtype=dtype)
        
        # Fill in preallocated arrays
        #Fill buffers for "from" rows
        for col in current_state.columns:
            values = selected[col].values
            if col == self.column: #if col is the InfState then assign string from_st
                self._buffers[col][:count] = self.from_st
            elif col == "N": #if col is 'N', then assign delta values
                self._buffers[col][:count] = deltas_N
            else: #otherwise keep the col's original data values
                self._buffers[col][:count] = values

        # Fill buffers for "to" rows
        for col in current_state.columns:
            if col == self.column:
                self._buffers[col][count:2 * count] = self.to_st
            elif col == "N":
                self._buffers[col][count:2 * count] = -deltas_N
            else:
                self._buffers[col][count:2 * count] = self._buffers[col][:count]  # duplicate "from" row values

        result = pd.DataFrame({col: self._buffers[col][:2 * count] for col in current_state.columns})

        return result

    def __str__(self) -> str:
        return f"SimpleTransition_Vec: {self.from_st} --> {self.to_st} at rate {self.rate}"
    
    def to_dict(self) -> dict:
        rc = {
            'tabularepimdl.SimpleTransition_Vec': self.model_dump()
        }

        return rc