from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union, Annotated
import pandas as pd
import numpy as np

from tabularepimdl.SimpleInfection import SimpleInfection
from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
from tabularepimdl.SI_Josh_Encode_Vec import SimpleInfection as Josh_SimpleInfection

class SimpleInfectionDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions of SimpleInfection rule at backend.
    @param structure: data structure used for rules.
    @param beta: the transmission parameter. 
    @param column: Name of the column this rule applies to.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param freq_dep: whether this model is a frequency dependent model.
    @param stochastic: whether the process is stochastic or deterministic.
    """
    structure: Literal["Pandas", "Numpy", "Numpy_Encode", "Josh_Encode_Vec"]
    beta: Annotated[int | float, Field(ge=0)]
    column: str
    s_st: str
    i_st: str
    inf_to: str
    freq_dep: bool = True
    infstate_compartments: list[str] = Field(default_factory=list)
    stochastic: bool = False

    #Dispatcher
    _dispatcher: Union[SimpleInfection, SimpleInfection_Vec_Encode, Josh_SimpleInfection] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = SimpleInfection(
                beta=self.beta,
                column=self.column,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy': #no Numpy option in the runner
            pass
        elif self.structure == 'Numpy_Encode':
            self._dispatcher = SimpleInfection_Vec_Encode(
                beta=self.beta,
                column=self.column,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                infstate_compartments=self.infstate_compartments,
                stochastic=self.stochastic
            )
        elif self.structure == 'Josh_Encode_Vec':
            self._dispatcher = Josh_SimpleInfection(
                beta=self.beta,
                column=self.column,
            )
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_deltas(self, current_state: pd.DataFrame | np.ndarray, col_idx_map: dict[str, int] | None = None, result_buffer: np.ndarray | None = None,  dt: int | float = 1.0) -> pd.DataFrame | np.ndarray:
        """
        @param current_state: a dataframe or numpy array (at the moment) representing the current epidemic state.
        @param col_idx_map: mapping of input data columns and their column index. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param dt: size of the timestep.
        No need to add stochastic argument to dispatcher's get_deltas() method.
        """
        if self.structure == 'Pandas':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt)
        elif self.structure == 'Numpy': #no Numpy option in the runner
            pass
        elif self.structure == 'Numpy_Encode':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt)

    def apply(self, state: np.ndarray, col_idx: dict[str, int], dt: float) -> np.ndarray: #run Josh's code
            return self._dispatcher.apply(state=state, col_idx=col_idx, dt=dt)
    
    def compile(self, comp_map: dict[str, int]) -> None: #for Josh's class
        """Resolve compartment string labels to integer codes."""
        return self._dispatcher.compile(comp_map=comp_map)
