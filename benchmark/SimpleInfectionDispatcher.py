from typing import Annotated, Literal, cast

import numpy as np
import pandas as pd
from legacy.pandas_reference.SimpleInfection import SimpleInfection
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode

# from tabularepimdl.SI_Josh_Encode_Vec import SimpleInfection as Josh_SimpleInfection #SI_Josh_Encode_Vec needs specific input data to work and it generates incorrect results.


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

    structure: Literal["Pandas", "Numpy", "Numpy_Encode"]
    beta: Annotated[int | float, Field(ge=0)]
    column: str
    s_st: str
    i_st: str
    inf_to: str
    freq_dep: bool = True
    infstate_compartments: list[str] = Field(default_factory=list)
    column_categories: list[str] = Field(default_factory=list)
    stochastic: bool = False

    # Dispatcher
    _dispatcher: SimpleInfection | SimpleInfection_Vec_Encode = PrivateAttr(default=None)

    def model_post_init(self, _):  # initialize dispatcher based on data structures
        if self.structure == "Pandas":
            self._dispatcher = SimpleInfection(beta=self.beta, column=self.column, s_st=self.s_st, i_st=self.i_st, inf_to=self.inf_to, stochastic=self.stochastic)
        elif self.structure == "Numpy":  # no Numpy option in the runner
            pass
        elif self.structure == "Numpy_Encode":
            self._dispatcher = SimpleInfection_Vec_Encode(
                beta=self.beta,
                column=self.column,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                infstate_compartments=self.infstate_compartments,
                column_categories=self.column_categories,
                stochastic=self.stochastic,
            )
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_deltas(
        self, current_state: pd.DataFrame | np.ndarray, col_idx_map: dict[str, int] | None = None, result_buffer: np.ndarray | None = None, dt: float = 1.0, stochastic: bool | None = None
    ) -> pd.DataFrame | np.ndarray:
        """
        @param current_state: a dataframe or numpy array (at the moment) representing the current epidemic state.
        @param col_idx_map: mapping of input data columns and their column index. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param result_buffer: takes pre-allocated numpy array and saves changing amount of current_state. Default is None so Pandas version's get_deltas() can invoke dispather's get_deltas().
        @param dt: size of the timestep.
        """
        if self.structure == "Pandas":
            return cast("SimpleInfection", self._dispatcher).get_deltas(current_state=current_state, dt=dt, stochastic=stochastic)
        if self.structure == "Numpy":  # no Numpy option in the runner
            pass
        elif self.structure == "Numpy_Encode":
            assert col_idx_map is not None
            assert result_buffer is not None
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt, stochastic=stochastic)
        return None

    def apply(self, state: np.ndarray, col_idx: dict[str, int], dt: float) -> np.ndarray:  # run Josh's code
        return self._dispatcher.apply(state=state, col_idx=col_idx, dt=dt)

    def compile(self, comp_map: dict[str, int]) -> None:  # for Josh's class
        """Resolve compartment string labels to integer codes."""
        return self._dispatcher.compile(comp_map=comp_map)
