import numpy as np
from pydantic import BaseModel, Field, field_validator, PrivateAttr, ValidationInfo

from tabularepimdl.Rule import Rule
from tabularepimdl._types.constrained_types import UniqueNonEmptyStrList


class EnvironmentalTransmission_Vec_Encode(Rule, BaseModel):
    """
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
    inf_col_categories: UniqueNonEmptyStrList = Field(description = "the categories used for attribute inf_col.")
    infstate_compartments: UniqueNonEmptyStrList = Field(description = "the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)

    def model_post_init(self, _):
        """
        Encode the input states based on the infection column's attribute values.

        Returns:
            Numerical values of encoded infection states.
        """
        if self.inf_col.lower() == 'infstate': #column is infection state
            infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
            self._s_code = infstate_to_int.get(self.s_st)
            self._inf_to_code = infstate_to_int.get(self.inf_to)
        else: #column is other attribute
            col_cat_to_int =  {s: i for i, s in enumerate(sorted(self.inf_col_categories))}  #encode column strings to integers {'0 to 4': 0, '5 to 9': 1}
            self._s_code = col_cat_to_int.get(self.s_st)
            self._inf_to_code = col_cat_to_int.get(self.inf_to)


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
            raise ValueError(f"Missing required columns in current_state: {required_columns}.")
        
        if stochastic is None:
            stochastic = self.stochastic

        infstate_idx = col_idx_map[self.inf_col]
        n_idx = col_idx_map['N']
        
        # Fast boolean mask for matching from-state
        mask_s_idxs = np.flatnonzero(current_state[:, infstate_idx] == self._s_code)
        if mask_s_idxs.size == 0:
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)

        selected_from = current_state[mask_s_idxs, :]
        #print('selected_from\n', selected_from)
        N = selected_from[:, n_idx]
        
        rate_const = 1 - np.exp(-dt * self.beta)
     
        # Update N values based on prI, folks out of S
        if stochastic:
            changed_N = -np.random.binomial(N.astype(np.int32), rate_const)
        else:
            changed_N = -N * rate_const

        count = selected_from.shape[0]

        result_buffer[:count, :] = selected_from #equivalent: self._from_code
        result_buffer[:count, n_idx] = changed_N  #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = selected_from
        result_buffer[count:2*count, infstate_idx] = self._inf_to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        return result_buffer[:2*count, :]
    
    
    def to_dict(self) -> dict:
        """
        Save the rule's attributes and their associated values to a dictionary.
                
        Returns:
            Rule attributes in a dictionary.
        """
        rc = {
            'tabularepimdl.EnvironmentalTransmission_Vec_Encode': self.model_dump()
        }
        return rc
    
    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required infection compartments if the `inf_col` takes 'infstate' value.
        """
        return self.infstate_compartments
    
    #set up a property to return all the required categories used in general column
    @property
    def inf_col_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required categories if the `inf_col` takes other string values.
        """
        return self.inf_col_categories