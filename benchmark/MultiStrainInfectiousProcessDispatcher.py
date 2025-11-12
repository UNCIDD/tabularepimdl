from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union
import pandas as pd
import numpy as np

from tabularepimdl.MultiStrainInfectiousProcess import MultiStrainInfectiousProcess as MultiStrainInfectiousProcess_Pandas
from tabularepimdl.MultiStrainInfectiousProcess_Vec_Encode import MultiStrainInfectiousProcess_Vec_Encode as MultiStrainInfectiousProcess_Vec_Encode_1
from tabularepimdl.MultiStrainInfectiousProcess_Vec_Encode_2 import MultiStrainInfectiousProcess_Vec_Encode_2 as MultiStrainInfectiousProcess_Vec_Encode_2

class SimpleObservationProcessDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions with Numba feature of SimpleObservationProcess rule at backend.
    @param structure: data structure used for rules.
    @param betas: a beta for each strain.
    @param columns: the strain columns for infection state. The number of strains should be the same length and order as betas.
    @param columns_all_categories: all the infection state categories the strain columns should have.
    @param cross_protect: a N(strain)*N(strain) matrix of cross protections.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param r_st: the state for immune/recovered, assumed to be R.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    @param freq_dep: whether this model is a frequency dependent model.
    @param infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """

    structure: Literal["Pandas", "Numpy_Vec_Encode_1", "Numpy_Vec_Encode_2"]
    betas: np.ndarray
    columns: list[str]
    columns_all_categories: list[str]
    cross_protect: np.ndarray
    s_st: str
    i_st: str
    r_st: str
    inf_to: str
    stochastic: bool
    freq_dep: bool
    infstate_compartments: list[str]

    #Dispatcher
    _dispatcher: Union[MultiStrainInfectiousProcess_Pandas, MultiStrainInfectiousProcess_Vec_Encode_1, MultiStrainInfectiousProcess_Vec_Encode_2] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = MultiStrainInfectiousProcess_Pandas(
                betas=self.betas,
                columns=self.columns,
                cross_protect=self.cross_protect,
                s_st=self.s_st,
                i_st=self.i_st,
                r_st=self.r_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic,
                freq_dep=self.freq_dep
            )
        elif self.structure == 'Numpy_Vec_Encode_1':
            self._dispatcher = MultiStrainInfectiousProcess_Vec_Encode_1(
                betas=self.betas,
                columns=self.columns,
                columns_all_categories=self.columns_all_categories,
                cross_protect=self.cross_protect,
                s_st=self.s_st,
                i_st=self.i_st,
                r_st=self.r_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic,
                freq_dep=self.freq_dep,
                infstate_compartments=self.infstate_compartments
            )
        elif self.structure == 'Numpy_Vec_Encode_2':
            self._dispatcher = MultiStrainInfectiousProcess_Vec_Encode_2(
                betas=self.betas,
                columns=self.columns,
                columns_all_categories=self.columns_all_categories,
                cross_protect=self.cross_protect,
                s_st=self.s_st,
                i_st=self.i_st,
                r_st=self.r_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic,
                freq_dep=self.freq_dep,
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
        if self.structure == 'Pandas':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Vec_Encode_1':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Vec_Encode_1':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)