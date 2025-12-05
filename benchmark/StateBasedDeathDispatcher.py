from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import Literal, Union
import pandas as pd
import numpy as np

from tabularepimdl.StateBasedDeathProcess import StateBasedDeathProcess
from tabularepimdl.StateBasedDeathProcess_Vec_Encode import StateBasedDeathProcess_Vec_Encode


class StateBasedDeathProcessDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions of StateBasedDeathProcess rule at backend.
    @param structure: data structure used for rules.
    @param column: single column used for StateBasedDeathProcess_Vec_Encode.
    @param columns: list of string used for StateBasedDeathProcess.
    @param states: the states of each column.
    @param rate: the rate at whihc people will die from.
    @param stochastic: whether the process is stochastic or deterministic.
    infstate_compartments: the infection compartments used in epidemics.
    """

    structure: Literal["Pandas", "Numpy_Encode"] = Field(description="data structure used for rules.")
    column: str = Field(description = "one column that we will check states against in StateBasedDeathProcess_Vec_Encode.")
    columns: list[str] = Field(description = "one or more columns that we will check states against in StateBasedDeathProcess.")
    column_states: list[str] = Field(description = "all the states of single column.")
    target_states: list[str] = Field(description = "targeted states to be processed of each column.")
    #states: list[str] = Field(description = "the states of each column.")
    rate: float = Field(ge=0, description = "the rate at whihc people will die from.")
    stochastic: bool = Field(default=False, description = "whether the transition is stochastic or deterministic.")
    infstate_compartments: list[str] = Field("the infection compartments used in epidemics.")

    #Dispatcher
    _dispatcher: Union[StateBasedDeathProcess, StateBasedDeathProcess_Vec_Encode] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = StateBasedDeathProcess(
                columns=self.columns,
                states=self.target_states,
                rate=self.rate,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy_Encode':
            self._dispatcher = StateBasedDeathProcess_Vec_Encode(
                column=self.column,
                column_states=self.column_states,
                target_states=self.target_states,
                rate=self.rate,
                stochastic=self.stochastic,
                infstate_compartments=self.infstate_compartments
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