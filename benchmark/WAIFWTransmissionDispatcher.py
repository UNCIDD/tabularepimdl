from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import Literal, Union
import pandas as pd
from numpy.typing import NDArray
import numpy as np

from tabularepimdl.WAIFWTransmissionOrig import WAIFWTransmissionOrig as WAIFWTransmission_Pandas
from tabularepimdl.WAIFWTransmission import WAIFWTransmission as WAIFWTransmission_Pandas_Numba
from tabularepimdl.WAIFWTransmission_Vec_Encode_Numba import WAIFWTransmission_Vec_Encode_Numba as WAIFWTransmission_Vec_Encode_Numba
from tabularepimdl.WAIFWTransmission_Vec_Encode_Bincount import WAIFWTransmission_Vec_Encode_Bincount as WAIFWTransmission_Vec_Encode_Bincount

class WAIFWTransmissionDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions with Numba feature of WAIFWTransmission rule at backend.
    @param structure: data structure used for rules.
    @param waifw_martrix: the waifw transmission rate matrix, a square matrix is required.
    @param inf_col: the column for this infectious process.
    @param group_col: the column where group is specified. The number of possible unique values in the column should match the waifw matrix size,
     and the unique values should have an order (i.e., it should be a pd.categorical).
    group_col_all_categories: all the categories the group column should have.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    structure: Literal["Pandas", "Pandas_Numba", "Numpy_Vec_Encode_Numba", "Numpy_Vec_Encode_Bincount"]
    waifw_matrix: NDArray[np.float64]
    inf_col: str
    group_col: str
    group_col_all_categories: list[str | int]
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    stochastic: bool = False
    infstate_compartments: list[str] = Field(default_factory=list)

    #Dispatcher
    _dispatcher: Union[WAIFWTransmission_Pandas, WAIFWTransmission_Pandas_Numba, WAIFWTransmission_Vec_Encode_Numba, WAIFWTransmission_Vec_Encode_Bincount] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = WAIFWTransmission_Pandas(
                waifw_matrix=self.waifw_matrix,
                inf_col=self.inf_col,
                group_col=self.group_col,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic
            )
        elif self.structure == 'Pandas_Numba':
            self._dispatcher = WAIFWTransmission_Pandas_Numba(
                waifw_matrix=self.waifw_matrix,
                inf_col=self.inf_col,
                group_col=self.group_col,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy_Vec_Encode_Numba':
            self._dispatcher = WAIFWTransmission_Vec_Encode_Numba(
                waifw_matrix=self.waifw_matrix,
                inf_col=self.inf_col,
                group_col=self.group_col,
                group_col_all_categories=self.group_col_all_categories,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic,
                infstate_compartments=self.infstate_compartments
            )
        elif self.structure == 'Numpy_Vec_Encode_Bincount':
            self._dispatcher = WAIFWTransmission_Vec_Encode_Bincount(
                waifw_matrix=self.waifw_matrix,
                inf_col=self.inf_col,
                group_col=self.group_col,
                group_col_all_categories=self.group_col_all_categories,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic,
                infstate_compartments=self.infstate_compartments
            )
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_deltas(self, current_state: pd.DataFrame | np.ndarray, col_idx_map: dict[str, int] | None = None, result_buffer: np.ndarray | None = None,  dt: int | float = 1.0, stochastic: bool | None = None) -> pd.DataFrame | np.ndarray:
        """
        @param current_state: a dataframe or numpy array (at the moment) representing the current epidemic state.
        @param col_idx_map: mapping of input data columns and their column index. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param dt: size of the timestep.
        """
        if self.structure == 'Pandas' or self.structure == 'Pandas_Numba':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Vec_Encode_Numba' or self.structure == 'Numpy_Vec_Encode_Bincount':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)
