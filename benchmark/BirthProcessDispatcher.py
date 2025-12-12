from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import Literal, Union
import pandas as pd
import numpy as np

from tabularepimdl.BirthProcess import BirthProcess
from tabularepimdl.BirthProcess_Vec_Encode import BirthProcess_Vec_Encode


class BirthProcessDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions of BirthProcess rule at backend.
    @param structure: data structure used for rules.
    @param rate: Birth rate per timestep.
    @param start_state_sig: initial state configuration for new births. 
    @param stochastic: whether the process is stochastic or deterministic.
    """

    structure: Literal["Pandas", "Numpy_Encode"] = Field(description="data structure used for rules.")
    rate: float = Field(ge=0, description = "birth rate at per time step (where N*rate births occur).")
    start_state_sig: dict = Field(description = "initial state configuration for new births.")
    column_to_sort: str = Field(description="Specify which input field is used to sort the dataset in ascending order.")
    stochastic: bool = Field(False, description = "whether the transition is stochastic or deterministic.")

    #Dispatcher
    _dispatcher: Union[BirthProcess, BirthProcess_Vec_Encode] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = BirthProcess(
                rate=self.rate,
                start_state_sig=self.start_state_sig,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy_Encode':
            self._dispatcher = BirthProcess_Vec_Encode(
                rate=self.rate,
                column_to_sort=self.column_to_sort,
                stochastic=self.stochastic
            )
        else:
            raise ValueError(f"Unknown structure: {self.structure}")
    
    def get_deltas(self, current_state: pd.DataFrame | np.ndarray, col_idx_map: dict[str, int] | None = None, result_buffer: np.ndarray | None = None,  dt: float = 1.0, stochastic: bool | None = None) -> pd.DataFrame | np.ndarray:
        """
        @param current_state: a dataframe or numpy array (at the moment) representing the current epidemic state.
        @param col_idx_map: mapping of input data columns and their column index. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param dt: size of the timestep.
        """
        if self.structure == 'Pandas':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Encode':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)