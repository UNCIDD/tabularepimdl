from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule


class EnvironmentalTransmission_Vec_Encode(Rule, BaseModel):
    """!
    Class represents disease transmission that happens in nature environment without any human-to-human or cross-species infection.
    The EnvironmentalTransmission implements the same logic and functionality as the SimpleTransition rule.
    
    Attributes:
        beta: transmission rate.
        inf_col: the column designating infection state.
        s_st: the state for susceptibles, assumed to be S
        inf_to: the state infectious population move to, assumed to be I.
        stochastic: whether the process is stochastic or deterministic.
        infstate_compartments: the infection compartments used in epidemics.
    """

    beta: float = Field(ge=0, description = "transmission rate.")
    inf_col: str = Field(description = "the column designating infection state.")
    s_st: str = Field(default="S", description = "the state for susceptibles, assumed to be S.")
    inf_to: str = Field(default="I", description = "the state infectious population move to, assumed to be I.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    infstate_compartments: list[str] = Field("the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _inf_code: int | None = PrivateAttr(default=None)

    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}
        self._s_code = infstate_to_int.get(self.s_st)
        self._inf_to_code = infstate_to_int.get(self.inf_to)

    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float =1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Compute the population deltas for the current state at a given time step.

        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): mapping of column names to their index positions.
            result_buffer (np.ndarray): A pre-allocated array that will be populated with the computed deltas. This array is modified in-place and returned.
            dt (float): The size of the time step. Defaults to 1.0.
            stochastic (bool, optional): Whether to apply stochastic modeling. If `None`, the class-level `self.stochastic` attribute is used.
        
        Returns:
            np.ndarray: A NumPy structured array containing the population deltas for s_st and inf_to.

        Raises:
            ValueError: If the column `'N'` is missing in `current_state`.
        """
        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        infstate_idx = col_idx_map[self.inf_col]
        n_idx = col_idx_map['N']
        
        # Fast boolean mask for matching from-state
        mask = current_state[:, infstate_idx] == self._s_code
        if not np.any(mask):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)

        from_row_idxs = np.flatnonzero(mask)
        selected_from = current_state[from_row_idxs, :]
        #print('selected_from\n', selected_from)
        N = selected_from[:, n_idx]
        
        rate_const = 1 - np.exp(-dt * self.beta)
     
        # Update N values based on prI, folks out of S
        if stochastic:
            changed_N = -np.random.binomial(N.astype(np.int32), rate_const)
        else:
            changed_N = -N * rate_const

        count = len(from_row_idxs)

        result_buffer[:count, :] = selected_from #equivalent: self._from_code
        result_buffer[:count, n_idx] = changed_N  #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = selected_from
        result_buffer[count:2*count, infstate_idx] = self._inf_to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        return result_buffer[:2*count, :]
    
    
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.EnvironmentalTransmission': self.model_dump()
        }
        return rc
    
    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]: 
        return self.infstate_compartments