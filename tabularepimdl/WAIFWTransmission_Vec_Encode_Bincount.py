import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationInfo, PrivateAttr
from numpy.typing import NDArray

from tabularepimdl.Rule import Rule


class WAIFWTransmission_Vec_Encode_Bincount(Rule, BaseModel):
    """!
    Rule that does transmission based on a simple WAIFW transmission matrix.
    Use np.bincount to generate inf_array.

    Attributes:
        waifw_martrix: the waifw transmission rate matrix, a square matrix is required.
        inf_col: the infection state column for this infectious process.
        group_col: the group where infection is applied, different group values are specified in this column. The number of possible unique values in the column should match the waifw matrix size,
                   and the unique values should have an order (i.e., the group_col should be a categorical datatype).
        group_col_all_categories: all the categories the group column should have.
        s_st: the state for susceptibles, assumed to be S.
        i_st: the state for infectious, assumed to be I.
        inf_to: the state susceptible population go to, assumed to be I.
        stochastic: whether the process is stochastic or deterministic.
        freq_dep (optional): whether this model is a frequency dependent model.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """

    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    waifw_matrix: NDArray[np.float64] = Field(description = "the waifw transmission rate matrix.")
    inf_col: str = Field(description = "the infection state column for this infectious process.")
    group_col: str = Field(description = "the group where infection is applied.")
    group_col_all_categories: list[str | int] = Field(description = "all the categories the group column should have.")
    s_st: str = Field(default="S", description = "the state for susceptibles.")
    i_st: str = Field(default="I", description = "the state for infectious.")
    inf_to: str = Field(default="I", description = "the state susceptible population go to.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    infstate_compartments: list[str] = Field("the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _i_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)
    _group_col_all_categories_code: list[int] | None = PrivateAttr(default=None)

    @field_validator("waifw_matrix", mode="before") #validate array type and its element sign
    @classmethod
    def validate_waifw_matrix(cls, matrix_parameters, field: ValidationInfo):
        """Ensure the input matrix is a 2-diemnsional array with sqaure shape and all elements are non-negative values."""
        #1. check list or array type.
        if isinstance(matrix_parameters, list): #convert list to array
            matrix_parameters = np.array(matrix_parameters, dtype = np.float64)
        elif isinstance(matrix_parameters, np.ndarray):
            matrix_parameters = matrix_parameters.astype(np.float64, copy = False)
        else:
            raise TypeError(f"{cls.__name__} expects a NumPy array for {field.field_name}, received {type(matrix_parameters)}.")
        
        #2. check if input is a 2-dimensional square matrix.
        if matrix_parameters.ndim !=2 or (matrix_parameters.shape[0] != matrix_parameters.shape[1]):
            raise ValueError(f"{cls.__name__} expects a 2-dimensional square matrix for {field.field_name}, received {matrix_parameters.shape}.")
        
        #3. check for non-empty matrix.
        if matrix_parameters.size == 0:
            raise ValueError("Matrix must not be empty.")
        
        #4. check for numeric data type.
        if not np.issubdtype(matrix_parameters.dtype, np.number):
            raise ValueError(f"Matrix must contain numeric data, received data type {matrix_parameters.dtype}.")
                             
        #5. check if all elements are non-negative values.
        if np.any(matrix_parameters < 0):
            raise ValueError(f"All elements in {field.field_name} must be non-negative, but received {matrix_parameters}.")
        
        #6. check for NaN or Inf.
        if np.isnan(matrix_parameters).any() or np.isinf(matrix_parameters).any():
            raise ValueError("Matrix must not contain NaN or Infinity values.")
        
        return matrix_parameters.T #transpose the input matrix
    
    @field_validator("group_col_all_categories", mode='before')
    @classmethod
    def validate_group_col_all_categories(cls, category_vals):
        if not category_vals:
            raise ValueError("The group_col_all_categories values must not be empty.")
        
        first_element_type = type(category_vals[0])
        for item in category_vals[1:]:
            if type(item) != first_element_type:
                raise ValueError(
                    f"All elements in group_col_all_categories must be of the same datatype. "
                    f"Found both {first_element_type.__name__} and {type(item).__name__}."
                )
        return category_vals
    
    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
        self._s_code = infstate_to_int.get(self.s_st)
        self._i_code = infstate_to_int.get(self.i_st)
        self._inf_to_code = infstate_to_int.get(self.inf_to)

        self.group_col_all_categories = sorted(self.group_col_all_categories) #sort the group_col's all categories
        self._group_col_all_categories_code = [i for i, v in enumerate(self.group_col_all_categories)] #encode each category, keeping numbers only

        
    
    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float =1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Compute the population deltas for the current state at a given time step.

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
        required_columns = "N" #check if column N presents in current_state
        if required_columns not in col_idx_map:
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic

        infstate_idx = col_idx_map[self.inf_col]
        group_col_idx = col_idx_map[self.group_col]
        n_idx = col_idx_map['N']

        if len(set(self.group_col_all_categories)) < len(set(current_state[:, group_col_idx])):
            raise ValueError(f"Number of elements in group_col_all_categories is less than the number of categories of input data, please check group_col_all_categories and input data.")
        
        #convert group_col to categorical type first, so groupby observed=False generate full list of array values
        #if not isinstance(current_state[self.group_col].dtype, pd.CategoricalDtype):
        #    current_state[self.group_col]=pd.Categorical(current_state[self.group_col])

        #Check if the number of unique categories in current_state's group_col matches waifw matrix's size
        #unique_val_group_col = np.unique(current_state[:, group_col_idx])
        if len(self._group_col_all_categories_code) != len(self.waifw_matrix):
            raise ValueError(f"Mismatch between the number of unique categories of input data and WAIFW matrix size. "
                             f"Expected {len(self.waifw_matrix)} categories, but found {len(self._group_col_all_categories_code)}. "
                             f"Categories: {self._group_col_all_categories_code}."
                            )

        ##create an array for the total number of infections in each unique group. Only records with i_st are sumed, other records's N are filled with 0.
        #inf_array = current_state.loc[current_state[self.inf_col]==self.i_st].groupby(self.group_col, observed=False)['N'].sum(numeric_only=True).values #moved ['N'] position #groupby approach
        
        num_of_categories = len(self._group_col_all_categories_code)
        present_category_codes = current_state[:, group_col_idx].astype(np.int64)
        infected_mask = current_state[:, infstate_idx] == self._i_code
        infected_group_codes = present_category_codes[infected_mask]
        infected_weights = current_state[infected_mask, n_idx]
        
        inf_array = np.bincount(infected_group_codes, infected_weights, num_of_categories)

        #print('inf_array is\n', inf_array) #debug

        prI_per_group = np.power(np.exp(-dt*self.waifw_matrix), inf_array)
        prI_per_group = 1-prI_per_group.prod(axis=1)
        
        
        ##get folks in susceptible states which link to all unique groups
        is_susceptible = current_state[:, infstate_idx] == self._s_code
        deltas_susceptible = current_state[is_susceptible]
        #print('is suscpet:', is_susceptible) #debug
        #print('deltas suscept\n', deltas_susceptible, '\n') #debug
        

        N_susceptible = deltas_susceptible[:, n_idx]
        #print('N_susceptible:', N_susceptible)

        #infectious process, getting the number of individuals who get infected from susceptible status
        susceptible_group_codes = present_category_codes[is_susceptible]
        prI_per_s_group = prI_per_group[susceptible_group_codes]
        #print('prI per group:', prI_per_group)

        if stochastic:
            changed_N = -np.random.binomial(N_susceptible, prI_per_s_group)
        else:
            changed_N = -N_susceptible * prI_per_s_group

        #print('changed N:', changed_N)

        count = len(N_susceptible)
        #print('count:', count)
        # Fill 'from' rows
        result_buffer[:count, :] = deltas_susceptible #equivalent: self._from_code
        result_buffer[:count, n_idx] = changed_N  #update column N with changed_N (negative value)

        # Fill 'to' rows
        result_buffer[count:2*count, :] = deltas_susceptible
        result_buffer[count:2*count, infstate_idx] = self._inf_to_code #update col infstate
        result_buffer[count:2*count, n_idx] = -changed_N  #update column N with inversed changed_N

        filtered_result_buffer = result_buffer[:2*count, :]
        result = filtered_result_buffer[filtered_result_buffer[:, n_idx] != 0] #remove rows with N=0
        
        return result


    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
                
        rc = {
            'tabularepimdl.WAIFWTransmission' : {
                'waifw_matrix' : self.waifw_matrix.T, #transpose waifw matrix back to its initial order before writting the attributes to dict
                'inf_col' : self.inf_col,
                'group_col' : self.group_col,
                's_st': self.s_st,
                'i_st': self.i_st,
                'inf_to': self.inf_to,
                'stochastic': self.stochastic
            }
        }
        
        return rc
    

    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]: 
        return self.infstate_compartments
    
    #set up a property to return all the required categories used in group_col
    @property
    def group_col_all(self) -> list[str]: 
        return self.group_col_all_categories