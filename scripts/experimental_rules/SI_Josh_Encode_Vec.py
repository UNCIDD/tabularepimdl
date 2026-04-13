#Josh's SimpleInfection_Vector class
import numpy as np
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, Field
from typing import Annotated, Dict
from tabularepimdl.operations import (
    apply_deterministic_transition,
    apply_stochastic_transition,
    masked_sum_meta,
)

class SimpleInfection(BaseModel):
    """
    Vectorized infection rule assuming preallocated modular layout: 
    rows are ordered by group × compartment with fixed layout at each time step.
    """

    beta: Annotated[float, Field(ge=0)]
    column: str                         # Column to match on (e.g. "compartment")
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    freq_dep: bool = True
    stochastic: bool = False

    # Internal compiled values (populated during model setup)
    s_code: int = -1
    i_code: int = -1
    inf_to_code: int = -1
    num_comps: int = -1  # Needed for modular indexing

    def compile(self, comp_map: Dict[str, int]) -> None:
        """Compile string state names to integer codes using model's comp_map."""
        self.s_code = comp_map[self.s_st]
        self.i_code = comp_map[self.i_st]
        self.inf_to_code = comp_map[self.inf_to]
        self.num_comps = len(comp_map)

    def apply(
        self,
        state: np.ndarray,
        col_idx: Dict[str, int],
        dt: float,
    ) -> np.ndarray:
        """
        Apply infection rule using fixed row layout:
        - Each row is (group × compartment) in fixed order
        - Compartments are indexed mod num_comps
        """
        #print(f"s_code, i_code, infcode, num_comps: {self.s_code, self.i_code, self.inf_to_code, self.num_comps}")

        N_col = col_idx["N"]
        comp_col = col_idx[self.column]
        labels = state[:, comp_col].astype(np.int32)
        N = state[:, N_col]
        #print('N:', N)

        # Determine which time step we're on
        num_rows = state.shape[0]
        num_steps = num_rows // self.num_comps
        if num_rows % self.num_comps != 0:
            raise RuntimeError("State array size is not divisible by number of compartments.")
        step_idx = num_steps - 1  # assume last block is current step

        # Slice out current step
        start = step_idx * self.num_comps
        end = start + self.num_comps
        cur_labels = labels[start:end]
        cur_N = N[start:end]
        #print(f"start: {start}, end: {end}")
        s_mask = cur_labels == self.s_code
        i_mask = cur_labels == self.i_code
        #print(f"s_mask: {s_mask}\n i_mask: {i_mask}")

        total_I = masked_sum_meta(values=cur_N, mask=i_mask)
        #print('total_I:', total_I)

        if self.freq_dep:
            total_N = masked_sum_meta(values=cur_N, mask=(cur_N > 0))
            denom = total_N if total_N > 0 else 1.0
            beta_eff = self.beta / denom
        else:
            beta_eff = self.beta

        prob = 1.0 - np.exp(-dt * beta_eff * total_I)
        #print('prob:', prob)

        s_idx_local = np.flatnonzero(s_mask)
        counts = cur_N[s_idx_local]
        probs = np.full_like(counts, prob)

        transition = (
            apply_stochastic_transition(counts, probs)
            if self.stochastic else
            apply_deterministic_transition(counts, probs)
        )

        transition = np.minimum(transition, counts)

        delta = np.zeros_like(state)

        # Update susceptible losses
        s_idx_global = start + s_idx_local
        delta[s_idx_global, N_col] -= transition

        # Gain to inf_to code
        if self.s_code != self.inf_to_code:
            to_idx_local = np.where(cur_labels == self.inf_to_code)[0]
            if len(to_idx_local) != len(s_idx_local):
                raise RuntimeError("Mismatch in number of source and target rows for infection.")
            to_idx_global = start + to_idx_local
            delta[to_idx_global, N_col] += transition
        else:
            # Same-state transition (e.g., S → S), treat as no-op
            delta[s_idx_global, N_col] += transition

        return delta
    
    @property
    def source_states(self):
        return [self.s_st]

    @property
    def target_states(self):
        return [self.inf_to]