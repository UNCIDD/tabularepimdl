from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union
import pandas as pd
import numpy as np

from tabularepimdl.SharedTraitInfection import SharedTraitInfection as SharedTraitInfection_Pandas
from tabularepimdl.SharedTraitInfection_Vec_Encode import SharedTraitInfection_Vec_Encode as SharedTraitInfection_Vec_Encode

class SharedTraitInfectionDispatcher(BaseModel):
    """
    Dispatches Pandas and various Numpy versions with Numba feature of SharedTraitInfection rule at backend.
    @param structure: data structure used for rules.
    @param inf_col: the infection state column for this infectious process.
    @param in_beta: transmission rate if trait shared.
    @param out_beta: transmission rrate if trait not shared.
    @param trait_col: the trait column shared by different populations.
    @param trait_col_all_categories: all the categories the trait column should have.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state susceptible populations go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R']. 
    """

    structure: Literal["Pandas", "Numpy_Vec_Encode"]
    inf_col: str
    in_beta: float
    out_beta: float
    trait_col: str
    trait_col_all_categories: list[str | int]
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    stochastic: bool = False
    infstate_compartments: list[str] = Field(default_factory=list)

    #Dispatcher
    _dispatcher: Union[SharedTraitInfection_Pandas, SharedTraitInfection_Vec_Encode] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = SharedTraitInfection_Pandas(
                in_beta=self.in_beta,
                out_beta=self.out_beta,
                inf_col=self.inf_col,
                trait_col=self.trait_col,
                s_st=self.s_st,
                i_st=self.i_st,
                inf_to=self.inf_to,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy_Vec_Encode':
            self._dispatcher = SharedTraitInfection_Vec_Encode(
                in_beta=self.in_beta,
                out_beta=self.out_beta,
                inf_col=self.inf_col,
                trait_col=self.trait_col,
                trait_col_all_categories=self.trait_col_all_categories,
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
        if self.structure == 'Pandas':
            return self._dispatcher.get_deltas(current_state=current_state, dt=dt, stochastic=stochastic)
        elif self.structure == 'Numpy_Vec_Encode':
            return self._dispatcher.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer, dt=dt)