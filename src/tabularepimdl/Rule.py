import importlib
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Rule(ABC):
    """
    Set up an abstract base class `Rule` which defines a generic interface for
    epidemic rules that are used in epidemic model engine.
    """

    """@param stochastic: whether the process is stochastic or deterministic."""
    stochastic: bool

    @abstractmethod
    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Method takes in current state and return a series of deltas to that state.
        It computes the population deltas for the current state at a given time step.

        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): mapping of column names to their index positions. e.g. {'N':0, 'InfState':1, 'Hosp':2}
            result_buffer (np.ndarray): A pre-allocated array that will be populated with the computed deltas. This array is modified in-place and returned.
            dt (float): The size of the time step. Defaults to 1.0.
            stochastic (bool, optional): Whether to apply stochastic modeling. If `None`, the class-level `self.stochastic` attribute is used.

        Returns:
            np.ndarray: A NumPy structured array containing the population deltas.

        Raises:
            ValueError: If the column `'N'` is missing in `current_state`.
        """

    @property
    def expansion_factor(self) -> int:
        """Maximum number of rows this rule can return per input row.

        Used by the model engine to size the shared delta buffer before running a timestep.
        Implemented by every NumPy rule; not declared `@abstractmethod` because
        the legacy pandas rules in `legacy/pandas_reference/` also subclass `Rule` and don't
        implement it (pandas has no preallocated buffer to size).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement expansion_factor.")

    def _encode_categorical_states(self, data_domains) -> None:
        """
        Use the fully updated data columns' domain mapping values to encode this rule's own
        column state values. Called by the model engine once per column domain update; not
        intended to be called directly by users (see `model_post_init` for the equivalent
        self-encoding path used when a rule is tested standalone).

        Implemented by every NumPy rule; not declared `@abstractmethod` for the
        same reason as `expansion_factor` above.

        Args:
            data_domains: mapping of column name to that column's category-to-code mapping.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _encode_categorical_states.")


    @abstractmethod
    def to_dict(self) -> dict:
        """
        Return a dictionary object appropriate for inclusion in a yaml definition of an epidemic.
        Should be a dictionary with the class name (in form module.classname)
        being the outer key containing information needed for the class to run `from_yaml`

        Returns:
            A dictionary representation of this object appropriate to read in by method `from_yaml`.
        """
