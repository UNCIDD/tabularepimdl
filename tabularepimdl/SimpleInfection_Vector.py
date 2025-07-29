import numpy as np
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, Field
from typing import Annotated
from tabularepimdl.operations import (
    apply_deterministic_transition,
    apply_stochastic_transition,
    masked_sum_meta,
    get_indices
)

class SimpleInfection(Rule, BaseModel):
    """Vectorized infection rule with integer-encoded compartment labels."""

    beta: Annotated[float, Field(ge=0)]
    column: str                      # e.g. "compartment"
    s_code: int = 0                  # integer code for S
    i_code: int = 1                  # integer code for I
    inf_to_code: int = 1            # transition to I
    freq_dep: bool = True
    stochastic: bool = False

    def apply(self, state: np.ndarray, col_idx: dict, dt: float, out: np.ndarray):
        """
        Applies infection rule using fully vectorized NumPy operations.
        - Only supports integer-encoded labels.
        - No DataFrame or string logic.
        """
        N = state[:, col_idx["N"]]
        labels = state[:, col_idx[self.column]].astype(np.int32)

        s_mask = labels == self.s_code
        i_mask = labels == self.i_code

        total_I = masked_sum_meta(values=N, mask=i_mask)

        if self.freq_dep:
            total_N = masked_sum_meta(values=N, mask=(N > 0))
            denom = total_N if total_N > 0 else 1.0
            beta_eff = self.beta / denom
        else:
            beta_eff = self.beta

        # Compute probability: P(infection) = 1 - exp(-β * I * dt)
        prob = 1.0 - np.power(np.exp(-dt * beta_eff), total_I)

        s_idx = get_indices(s_mask)
        counts = N[s_idx]
        transition = (
            apply_stochastic_transition(counts, np.full_like(counts, prob))
            if self.stochastic else
            apply_deterministic_transition(counts, np.full_like(counts, prob))
        )

        out[s_idx, col_idx["N"]] -= transition
        out[s_idx, col_idx[self.column]] = self.inf_to_code  # vectorized label update

    def to_yaml(self) -> dict:
        return {
            "tabularepimdl.SimpleInfection": self.model_dump()
        }
