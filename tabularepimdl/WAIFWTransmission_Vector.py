import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Annotated, Dict
from tabularepimdl.Rule import Rule
from tabularepimdl.operations import (
    apply_stochastic_transition,
    apply_deterministic_transition,
    grouped_sum_meta,
)
from tabularepimdl.matrixops_utils import encode_dense_groups, encode_sparse_groups

class WAIFWTransmission(Rule, BaseModel):
    """
    Vectorized WAIFW-style infection rule.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    waifw_matrix: Annotated[np.ndarray, Field()]
    inf_col: str
    group_col: str = "group"
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    stochastic: bool = False

    # Compiled attributes
    s_code: int = -1
    i_code: int = -1
    inf_to_code: int = -1
    num_comps: int = -1
    group_col_idx: int = -1
    inf_col_idx: int = -1

    @field_validator("waifw_matrix", mode="before")
    @classmethod
    def validate_waifw_matrix(cls, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError("waifw_matrix must be a NumPy array or convertible list.")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("waifw_matrix must be a square matrix.")
        if np.any(value < 0) or np.isnan(value).any() or np.isinf(value).any():
            raise ValueError("waifw_matrix must be non-negative and finite.")
        return value.T  # Transpose for consistency with legacy behavior

    def compile(self, comp_map: Dict[str, int], num_comps: int = None) -> None:
        self.s_code = comp_map[self.s_st]
        self.i_code = comp_map[self.i_st]
        self.inf_to_code = comp_map[self.inf_to]
        self.num_comps = len(comp_map) if num_comps is None else num_comps

    def apply(self, state: np.ndarray, col_idx: Dict[str, int], dt: float) -> np.ndarray:
        N_col = col_idx["N"]
        group_col_idx = col_idx[self.group_col]
        comp_col = col_idx[self.inf_col]
        labels = state[:, comp_col].astype(np.int32)
        N = state[:, N_col]
        group_ids = state[:, group_col_idx].astype(np.int32)

        num_rows = state.shape[0]
        num_steps = num_rows // self.num_comps
        if num_rows % self.num_comps != 0:
            raise RuntimeError("State array size is not divisible by number of compartments.")
        step_idx = num_steps - 1
        start = step_idx * self.num_comps
        end = start + self.num_comps

        cur_labels = labels[start:end]
        cur_N = N[start:end]
        cur_group_ids = group_ids[start:end]

        # Compute group membership matrix (dense)
        G = int(np.max(cur_group_ids)) + 1
        if (cur_N.shape[0] <= 1e5 and G <= 100) or (cur_N.shape[0] <= 1e6 and G <= 25):
            group_matrix = encode_dense_groups(cur_group_ids, G)
        else:
            group_matrix = encode_sparse_groups(cur_group_ids, G)




        # Compute I_j per group (I: infected)
        infected_mask = cur_labels == self.i_code
        I_vector = cur_N * infected_mask
        I_per_group = group_matrix @ I_vector  # shape (G,)

        # Compute Pr(infection) for each group
        waifw = self.waifw_matrix
        expo = np.exp(-dt * waifw)              # shape (G x G)
        power = expo ** I_per_group[None, :]    # shape (G x G)
        prI = 1.0 - np.prod(power, axis=1)       # shape (G,)

        # Apply to susceptibles
        susceptible_mask = cur_labels == self.s_code
        s_idx_local = np.flatnonzero(susceptible_mask)
        s_idx_global = start + s_idx_local
        s_counts = cur_N[s_idx_local]
        s_groups = cur_group_ids[s_idx_local]
        s_probs = prI[s_groups]

        transition = (
            apply_stochastic_transition(s_counts, s_probs)
            if self.stochastic else
            apply_deterministic_transition(s_counts, s_probs)
        )
        transition = np.minimum(transition, s_counts)

        delta = np.zeros_like(state)
        delta[s_idx_global, N_col] -= transition

        if self.s_code != self.inf_to_code:
            to_idx_local = np.where(cur_labels == self.inf_to_code)[0]
            if len(to_idx_local) != len(s_idx_local):
                raise RuntimeError("Mismatch in number of S and target rows.")
            to_idx_global = start + to_idx_local
            delta[to_idx_global, N_col] += transition
        else:
            delta[s_idx_global, N_col] += transition

        return delta

    def to_yaml(self) -> dict:
        return {
            "tabularepimdl.WAIFWTransmission": {
                "waifw_matrix": self.waifw_matrix.T.tolist(),
                "inf_col": self.inf_col,
                "group_col": self.group_col,
                "s_st": self.s_st,
                "i_st": self.i_st,
                "inf_to": self.inf_to,
                "stochastic": self.stochastic,
            }
        }

    @property
    def source_states(self):
        return [self.s_st]

    @property
    def target_states(self):
        return [self.inf_to]
