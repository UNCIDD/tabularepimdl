from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union
import pandas as pd
import numpy as np

from tabularepimdl.SimpleTransition import SimpleTransition
from tabularepimdl.SimpleTransition_Vec import SimpleTransition_Vec
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode
from tabularepimdl.ST_Josh_Encode_Vec import SimpleTransition as Josh_SimpleTransition

class SimpleTransitionDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions of SimpleTransition rule at backend.
    @param structure: data structure used for rules.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    """
    structure: Literal["Pandas", "Numpy", "Numpy_Encode", "Josh_Encode_Vec"]
    column: str
    from_st: str
    to_st: str
    rate: float = Field(ge=0)
    infstate_compartments: list[str] = Field(default_factory=list)
    stochastic: bool = False

    #Dispatcher
    _dispatcher: Union[SimpleTransition, SimpleTransition_Vec, SimpleTransition_Vec_Encode, Josh_SimpleTransition] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = SimpleTransition(
                column=self.column,
                from_st=self.from_st,
                to_st=self.to_st,
                rate=self.rate,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy':
            self._dispatcher = SimpleTransition_Vec(
                column=self.column,
                from_st=self.from_st,
                to_st=self.to_st,
                rate=self.rate,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy_Encode':
            self._dispatcher = SimpleTransition_Vec_Encode(
                column=self.column,
                from_st=self.from_st,
                to_st=self.to_st,
                rate=self.rate,
                infstate_compartments=self.infstate_compartments,
                stochastic=self.stochastic
            )
        elif self.structure == 'Josh_Encode_Vec':
            self._dispatcher = Josh_SimpleTransition(
                column=self.column,
                from_st=self.from_st,
                to_st=self.to_st,
                rate=self.rate,
                stochastic=self.stochastic
            )
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_deltas(self, current_state: pd.DataFrame | np.ndarray, col_idx_map: dict[str, int] | None = None, result_buffer: np.ndarray | None = None,  dt: int | float = 1.0, stochastic: bool | None = None) -> pd.DataFrame | np.ndarray:
        """
        @param current_state: a dataframe or numpy array (at the moment) representing the current epidemic state.
        @param col_idx_map: mapping of input data columns and their column index. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param dt: size of the timestep.
        No need to add stochastic argument to dispatcher's get_deltas() method.
        """
        if self.structure == 'Pandas' or self.structure == 'Numpy':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Encode':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)

    def apply(self, state: np.ndarray, col_idx: dict[str, int], dt: float) -> np.ndarray: #run Josh's code
            return self._dispatcher.apply(state=state, col_idx=col_idx, dt=dt)
    
    def compile(self, comp_map: dict[str, int]) -> None: #for Josh's class
        """Resolve compartment string labels to integer codes."""
        return self._dispatcher.compile(comp_map=comp_map)
