import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from tabularepimdl.Rule import Rule
from tabularepimdl._types.constrained_types import UniqueNonEmptyStrList

class BirthProcess_Vec_Encode(Rule, BaseModel):
    """
    Represents a birth process where people are borne based
    on a birth rate based on the full poplation size.

    Attributes:
        rate: Birth rate per timestep (N * rate births).
        column_to_sort: Specify which input field is used to sort the dataset in ascending order.
        stochastic: Whether the transition is stochastic.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """

    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
      
    rate: float = Field(ge=0, description = "birth rate at per time step (where N*rate births occur).")
    column_to_sort: str = Field(description="specify which input field is used to sort the dataset in ascending order.")
    stochastic: bool = Field(default=False, description = "whether the transition is stochastic or deterministic.")
    infstate_compartments: UniqueNonEmptyStrList = Field(description = "the infection compartments used in epidemics.")

    _start_state_sig: np.ndarray = PrivateAttr(default_factory=lambda: np.array([])) #initial state configuration for new births.
    _start_state_saved: bool = PrivateAttr(default=False) #to identify if a valid value has been assigned to _start_state_sig

    
    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required infection compartments.
        """
        return self.infstate_compartments
    
    @property
    def start_state_sig(self) -> np.ndarray:
        """
        Return the start_state_sig value if it is not empty.

        Returns:
            A Numpy array of initial state configuration for new births.

        Raises:
            ValueError: If the `_start_state_sig` is empty.
        """
        if self._start_state_sig.size == 0:
            raise ValueError(f"No start state data is available due to no input current state data is provided. "
                             f"Please provide a non-empty current state data to the get_deltas() of the rule first."
                             )
        else:
            return self._start_state_sig

    @property
    def expansion_factor(self) -> int:
        """
        Maximum number of rows this rule can return per input rows.
        """
        return len(self.infstate_compartments)
        
    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Compute the population birth deltas for the current state at a given time step.
        
        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): Mapping of column names to their index positions.
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
            raise ValueError(f"Missing required columns in current_state: {required_columns}.")
        
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
    

        count = len(self._start_state_sig)
        #print('count:', count)
        #print('start_state_sig\n', self._start_state_sig)
        result_buffer[count-1] = self._start_state_sig
        #print('result buffer\n', result_buffer)
        result_buffer[:count, n_idx] = changed_N

        return result_buffer[:count, :]
        
    def to_dict(self) -> dict:
        """
        Save the rule's attributes and their associated values to a dictionary.
        
        Returns:
            Rule attributes in a dictionary.
        """
        rc = {
            'tabularepimdl.BirthProcess_Vec_Encode': self.model_dump()
        }
        return rc 