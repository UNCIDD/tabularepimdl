from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union
import pandas as pd
import numpy as np

from tabularepimdl.SimpleObservationProcess import SimpleObservationProcess as SimpleObservationProcess_Pandas
from tabularepimdl.SimpleObservationProcess_Vec_Encode import SimpleObservationProcess_Vec_Encode as SimpleObservationProcess_Vec_Encode
from tabularepimdl.SimpleObservationProcess_Vec_Encode_nobuffer import SimpleObservationProcess_Vec_Encode_nobuffer

class SimpleObservationProcessDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions with Numba feature of SimpleObservationProcess rule at backend.
    @param structure: data structure used for rules.
    @param source_col: the column containing source_state for the observation process.
    @param source_state: the state individuals start, listed in source_col.
    @param obs_col: the column that contains each group of individuals' observed state.
    @param rate: the number of people move from a particular state into another state per unit time.
    @param unobs_state: un-observed state, listed in obs_col.
    @param incobs_state: incident-observed state, listed in obs_col.
    @param prevobs_state: previously-observed state, listed in obs_col.
    @param stochastic: whether the process is stochastic or deterministic.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    @param observation_compartments: the observation compartments used in epidemics. e.g. ['U', 'P', 'I'], U=unobserved, P=previously-observed, I=incident-observed
    """

    structure: Literal["Pandas", "Numpy_Vec_Encode", "Numpy_Vec_Encode_nobuffer"]
    source_col: str
    source_state: str
    obs_col: str
    rate: float
    unobs_state: str
    incobs_state: str
    prevobs_state: str
    stochastic: bool
    infstate_compartments: list[str]
    obs_col_all_categories: list[str]

    #Dispatcher
    _dispatcher: Union[SimpleObservationProcess_Pandas, SimpleObservationProcess_Vec_Encode, SimpleObservationProcess_Vec_Encode_nobuffer] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = SimpleObservationProcess_Pandas(
                source_col=self.source_col,
                source_state=self.source_state,
                obs_col=self.obs_col,
                rate=self.rate,
                unobs_state=self.unobs_state,
                incobs_state=self.incobs_state,
                prevobs_state=self.prevobs_state,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy_Vec_Encode':
            self._dispatcher = SimpleObservationProcess_Vec_Encode(
                source_col=self.source_col,
                source_state=self.source_state,
                obs_col=self.obs_col,
                rate=self.rate,
                unobs_state=self.unobs_state,
                incobs_state=self.incobs_state,
                prevobs_state=self.prevobs_state,
                stochastic=self.stochastic,
                infstate_compartments=self.infstate_compartments,
                obs_col_all_categories=self.obs_col_all_categories
            )
        elif self.structure == 'Numpy_Vec_Encode_nobuffer':
            self._dispatcher = SimpleObservationProcess_Vec_Encode_nobuffer(
                source_col=self.source_col,
                source_state=self.source_state,
                obs_col=self.obs_col,
                rate=self.rate,
                unobs_state=self.unobs_state,
                incobs_state=self.incobs_state,
                prevobs_state=self.prevobs_state,
                stochastic=self.stochastic,
                infstate_compartments=self.infstate_compartments,
                obs_col_all_categories=self.obs_col_all_categories
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
        elif self.structure == 'Numpy_Vec_Encode':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Vec_Encode_nobuffer':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)