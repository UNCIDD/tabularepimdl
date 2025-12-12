from collections.abc import Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tabularepimdl.Rule import Rule


class BirthProcess_Vec_Encode(Rule, BaseModel):
    """!
    Represents a birth process where people are borne based
    on a birth rate based on the full poplation size.

    Attributes:
        rate: Birth rate per timestep (N * rate births).
        column_to_sort: Specify which input field is used to sort the dataset in ascending order.
        stochastic: Whether the transition is stochastic.
    """

    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
      
    rate: float = Field(ge=0, description = "birth rate at per time step (where N*rate births occur).")
    column_to_sort: str = Field(description="Specify which input field is used to sort the dataset in ascending order.")
    stochastic: bool = Field(False, description = "whether the transition is stochastic or deterministic.")

    _start_state_sig: np.ndarray = PrivateAttr(default_factory=lambda: np.array([])) #initial state configuration for new births.
    _start_state_saved: bool = PrivateAttr(default=False) #to identify if a valid value has been assigned to _start_state_sig
    
    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Compute the population birth deltas for the current state at a given time step.
        
        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): mapping of column names to their index positions.
            result_buffer (np.ndarray): A pre-allocated array that will be populated with the computed deltas. This array is modified in-place and returned.
            dt (float): The size of the time step. Defaults to 1.0.
            stochastic (bool, optional): Whether to apply stochastic modeling. If `None`, the class-level `self.stochastic` attribute is used.
        
        Returns:
            np.ndarray: A NumPy structured array containing the population birth deltas.

        Raises:
            ValueError: If the column `'N'` is missing in `current_state`.
        """
        N: float #sum of population
        
        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        if current_state.size == 0: #check if the input array is empty
            print('input array data size is zero, return empty array.')
            return np.empty((0, current_state.shape[1]))
        
        n_idx = col_idx_map["N"]
        N = np.sum(current_state[:, n_idx])
        #print('input current state\n', current_state)
        # Compute transition rate
        rate_const = 1 - np.exp(-dt * self.rate)

        birth_value = N * rate_const

        if stochastic:
            changed_N = np.random.poisson(birth_value)  #get a random single outcome value by poisson distribution
        else:
            changed_N = birth_value
        #print('result buffer init\n', result_buffer)
        
        if not self._start_state_saved:
            sort_column_index = col_idx_map[self.column_to_sort] #obtain the designated column index (e.g. AgeCat)

            sort_indices = np.argsort(current_state[:, sort_column_index], axis=0) # sorts along first axis (row by row)
            #print('sort_indices:', sort_indices)

            current_state = current_state[sort_indices] #sort the input array by designated column (e.g. AgeCat)
            #print('sorted current state\n', current_state)

            self._start_state_sig = current_state[0:1] #obtain a 2D array with one row [0:1]
            self._start_state_saved = True #once start state data is assigned, flip the flag to True
        else:
            pass #start state has had values saved, use it directly in the following code.
    

        count = len(self._start_state_sig) #this step could be moved into if-else start_state_saved
        #print('count:', count)
        #print('start_state_sig\n', self._start_state_sig)
        result_buffer[count-1] = self._start_state_sig #this step could be moved into if-else start_state_saved
        #print('result buffer\n', result_buffer)
        result_buffer[:count, n_idx] = changed_N

        return result_buffer[:count, :]
        
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.BirthProcess_Vec_Encode' : {
                'rate': self.rate,
                'column_to_sort': self.column_to_sort,
                'stochastic': self.stochastic
            }
        }
        return rc
    
    @property
    def start_state_sig(self) -> np.ndarray:
        if self._start_state_sig.size == 0:
            raise ValueError(f"No start state data is available due to no input current state data is provided to get_deltas() of the rule.")
        else:
            return self._start_state_sig 