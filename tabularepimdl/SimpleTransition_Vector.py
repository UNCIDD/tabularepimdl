import numpy as np
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, Field
from typing import Annotated
from tabularepimdl.operations import (
    apply_deterministic_transition,
    apply_stochastic_transition,
    get_indices
)

class SimpleTransition(Rule, BaseModel):
    """
    Fast array-based transition rule:
    Transitions rows in a given column from `from_code` to `to_code` at a given rate.
    All inputs and outputs are integer-encoded, NumPy only.
    """

    column: str                        # e.g. "compartment"
    from_code: int                    # encoded source state
    to_code: int                      # encoded destination state
    rate: Annotated[float, Field(ge=0)]
    stochastic: bool = False

    def apply(self, state: np.ndarray, col_idx: dict, dt: float, out: np.ndarray):
        """
        Apply a simple Markov transition: X → Y at rate r.
        Operates entirely on NumPy arrays with integer encoding.
        """
        N = state[:, col_idx["N"]]
        labels = state[:, col_idx[self.column]].astype(np.int32)
        mask = labels == self.from_code

        idx = get_indices(mask)
        counts = N[idx]
        prob = 1.0 - np.exp(-dt * self.rate)

        transition = (
            apply_stochastic_transition(counts, np.full_like(counts, prob))
            if self.stochastic else
            apply_deterministic_transition(counts, np.full_like(counts, prob))
        )

        # Subtract from from_state, add to to_state
        out[idx, col_idx["N"]] -= transition
        out[idx, col_idx[self.column]] = self.to_code  # vectorized

    def __str__(self) -> str:
        return f"SimpleTransition: {self.from_code} --> {self.to_code} at rate {self.rate}"

    def to_yaml(self) -> dict:
        return {
            "tabularepimdl.SimpleTransition": self.model_dump()
        }
