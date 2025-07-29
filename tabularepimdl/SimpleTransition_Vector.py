import numpy as np
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, Field
from typing import Annotated, Dict
from tabularepimdl.operations import (
    apply_deterministic_transition,
    apply_stochastic_transition,
)


class SimpleTransition(Rule, BaseModel):
    """
    Vectorized first-order transition rule assuming fixed-layout row structure
    with (group × compartment) layout per time step.
    """

    column: str                             # e.g., "compartment"
    from_st: str                            # source state (string label)
    to_st: str                              # destination state (string label)
    rate: Annotated[float, Field(ge=0)]     # transition rate per unit time
    stochastic: bool = False                # whether to apply randomness

    # Internal integer codes and layout info (populated at compile time)
    from_code: int = -1
    to_code: int = -1
    num_comps: int = -1

    def compile(self, comp_map: Dict[str, int]) -> None:
        """Resolve compartment string labels to integer codes."""
        self.from_code = comp_map[self.from_st]
        self.to_code = comp_map[self.to_st]
        self.num_comps = len(comp_map)

    def apply(
        self,
        state: np.ndarray,
        col_idx: Dict[str, int],
        dt: float,
    ) -> np.ndarray:
        """
        Apply the transition rule to a single time step block of state,
        returning full-size delta array aligned with `state`.
        """
        N_col = col_idx["N"]
        comp_col = col_idx[self.column]
        labels = state[:, comp_col].astype(np.int32)
        N = state[:, N_col]

        num_rows = state.shape[0]
        num_steps = num_rows // self.num_comps
        if num_rows % self.num_comps != 0:
            raise RuntimeError("State array size is not divisible by number of compartments.")
        step_idx = num_steps - 1  # assume we’re updating the last block

        start = step_idx * self.num_comps
        end = start + self.num_comps

        cur_labels = labels[start:end]
        cur_N = N[start:end]

        mask = cur_labels == self.from_code
        idx_local = np.flatnonzero(mask)
        counts = cur_N[idx_local]

        prob = 1.0 - np.exp(-dt * self.rate)
        probs = np.full_like(counts, prob)

        transition = (
            apply_stochastic_transition(counts, probs)
            if self.stochastic else
            apply_deterministic_transition(counts, probs)
        )

        transition = np.minimum(transition, counts)

        delta = np.zeros_like(state)

        idx_global = start + idx_local

        # Loss from source
        delta[idx_global, N_col] -= transition

        # Gain at destination rows
        if self.from_code != self.to_code:
            to_idx_local = np.where(cur_labels == self.to_code)[0]
            if len(to_idx_local) != len(idx_local):
                raise RuntimeError("Mismatch in number of source and target rows for transition.")
            to_idx_global = start + to_idx_local
            delta[to_idx_global, N_col] += transition
        else:
            # Same-state transition: relabel only
            delta[idx_global, N_col] += transition

        return delta

    @property
    def source_states(self) -> list[str]:
        return [self.from_st]

    @property
    def target_states(self) -> list[str]:
        return [self.to_st]

    def __str__(self) -> str:
        return f"SimpleTransition: {self.from_st} → {self.to_st} @ rate {self.rate}"

    def to_yaml(self) -> dict:
        return {
            "tabularepimdl.SimpleTransition": self.model_dump()
        }
