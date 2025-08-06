from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union, Dict
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
    """
    structure: Literal["Pandas", "Numpy", "Numpy_Encode", "Josh_Encode_Vec"]
    column: str
    from_st: str
    to_st: str
    rate: float = Field(ge=0)
    infstate_compartments: list[str] = None
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

    def get_deltas(self, current_state: pd.DataFrame | np.ndarray, data_col: Dict[str, int] = None, result_buffer: np.ndarray = None,  dt: int | float = 1.0) -> pd.DataFrame | np.ndarray:
        """
        @param current_state: a dataframe or numpy array (at the moment) representing the current epidemic state.
        @param data_col: mapping of input data columns and their column index.
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state.
        @param dt: size of the timestep.
        No need to add stochastic argument to dispatcher's get_deltas() method.
        """
        if self.structure == 'Pandas' or self.structure == 'Numpy':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt)
        elif self.structure == 'Numpy_Encode':
            return self._dispatcher.get_deltas(current_state=current_state, data_col=data_col, result_buffer=result_buffer, dt=dt)

    def apply(self, state: np.ndarray, col_idx: Dict[str, int], dt: float) -> np.ndarray: #run Josh's code
            return self._dispatcher.apply(state=state, col_idx=col_idx, dt=dt)
    
    def compile(self, comp_map: Dict[str, int]) -> None: #for Josh's class
        """Resolve compartment string labels to integer codes."""
        return self._dispatcher.compile(comp_map=comp_map)
