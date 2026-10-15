from tabularepimdl.Rule import Rule

import numpy as np


class DummyRule(Rule):
    """Minimal concrete implementation for testing the Rule interface."""

    stochastic = False

    @property
    def expansion_factor(self) -> int:
        return 3

    def get_deltas(
        self,
        current_state: np.ndarray,
        col_idx_map: dict[str, int],
        result_buffer: np.ndarray,
        dt: float = 1.0,
        stochastic: bool | None = None,
    ) -> np.ndarray:
        if "N" not in col_idx_map:
            raise ValueError("Missing required column 'N'")

        result_buffer[:] = 0
        result_buffer[:, col_idx_map["N"]] = (
            -current_state[:, col_idx_map["N"]] * dt
        )
        return result_buffer

    def to_dict(self) -> dict:
        return {"DummyRule": {"stochastic": self.stochastic}}
