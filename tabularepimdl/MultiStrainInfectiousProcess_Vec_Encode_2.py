import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, ValidationInfo, field_validator, model_validator, PrivateAttr

from tabularepimdl.Rule import Rule


class MultiStrainInfectiousProcess_Vec_Encode_2(Rule, BaseModel):
    """! 
    Rule that takes a cross protection matrix, a list of infection state columns and an array of betas 
    to simulate infections contain multiple strains of the same pathogen.
    Does not allow co-infections.

    Attributes:
        betas: a beta for each strain.
        columns: the strain columns for infection state. The number of strains should be the same length and order as betas.
        columns_all_categories: all the infection state categories the strain columns should have.
        cross_protect: a N(strain)*N(strain) matrix of cross protections.
        s_st: the state for susceptibles, assumed to be S.
        i_st: the state for infectious, assumed to be I.
        r_st: the state for immune/recovered, assumed to be R.
        inf_to: the state infectious folks go to, assumed to be I.
        stochastic: whether the process is stochastic or deterministic.
        freq_dep: whether this model is a frequency dependent model.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    betas: np.ndarray = Field(description = "a beta for each strain.")
    columns: list[str] = Field(description = "the strain columns for infection state.")
    columns_all_categories: list[str] = Field(description = "all the infection state categories the strain columns should have.")
    cross_protect: np.ndarray = Field(description = "a N(strain)*N(strain) matrix of cross protections.")
    s_st: str = Field(default="S", description = "the state for susceptibles.")
    i_st: str = Field(default="I", description = "the state for infectious.")
    r_st: str = Field(default="R", description = "the state for immune/recovered.")
    inf_to: str = Field(default="I", description = "the state infectious folks go to.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    freq_dep: bool = Field(default=True, description = "whether this model is a frequency dependent model.")
    infstate_compartments: list[str] = Field("the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _i_code: int | None = PrivateAttr(default=None)
    _r_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)
    _columns_idx: list[int] = PrivateAttr(default_factory=list)
    _columns_all_categories_code: list[int] | None = PrivateAttr(default_factory=None)

    @field_validator("betas", "cross_protect", mode="before") #validate array type and its element sign
    @classmethod
    def validate_numpy_array(cls, array_parameters, field: ValidationInfo):
        """Ensure the input is a NumPy array and all elements are non-negative values."""
        #1. check list or array type.
        if isinstance(array_parameters, list): #convert list to array
            array_parameters = np.array(array_parameters)
        elif isinstance(array_parameters, np.ndarray):
            array_parameters = array_parameters
        else:
            raise ValueError(f"{cls.__name__} expects a NumPy array for {field.field_name}, received {type(array_parameters)}.")
        
        #2. check for numeric data type.
        if not np.issubdtype(array_parameters.dtype, np.number):
            raise ValueError(f"Array must contain numeric data, received data type {array_parameters.dtype}.")
        
        #3. check if all elements are non-negative values.
        if np.any(array_parameters < 0):
            raise ValueError(f"All elements in {field.field_name} must be non-negative, received {array_parameters}.")
        
        #4. check for NaN or Inf.
        if np.isnan(array_parameters).any() or np.isinf(array_parameters).any():
            raise ValueError("Arrays must not contain NaN or Infinity values.")
        
        return array_parameters

    
    @model_validator(mode="after") #after all fields are validated, check cross fields relationship
    def check_dimensions(cls, parameter_values):
        """Ensure betas and cross_protect have matching dimensions."""
        betas = parameter_values.betas
        columns = parameter_values.columns
        cross_protect = parameter_values.cross_protect

        if len(columns) != len(betas):
            raise ValueError(f"The number of 'columns' ({len(columns)}) must match the number of 'betas' ({len(betas)}).")

        if cross_protect.shape[0] != cross_protect.shape[1] or cross_protect.shape[0] != len(betas):
            raise ValueError(
                f"'cross_protect' must be a square matrix of size {len(betas)}x{len(betas)}, got {cross_protect.shape}."
            )
        
        return parameter_values
    
    @field_validator("columns_all_categories", mode='before')
    @classmethod
    def validate_group_col_all_categories(cls, category_vals):
        """Validate all elements in the list have the same data type."""
        if not category_vals:
            raise ValueError("The columns_all_categories values must not be empty.")
        
        first_element_type = type(category_vals[0])
        for item in category_vals[1:]:
            if type(item) != first_element_type:
                raise ValueError(
                    f"All elements in columns_all_categories must be of the same datatype. "
                    f"Found both {first_element_type.__name__} and {type(item).__name__}."
                )
        return category_vals
    
    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
        self._s_code = infstate_to_int.get(self.s_st)
        self._i_code = infstate_to_int.get(self.i_st)
        self._r_code = infstate_to_int.get(self.r_st)
        self._inf_to_code = infstate_to_int.get(self.inf_to)

        self.columns_all_categories = sorted(self.columns_all_categories) #sort the trait_col's all categories
        self._columns_all_categories_code = [i for i, v in enumerate(self.columns_all_categories)] #encode each category, keeping numbers only

    
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
        
        if not self._columns_idx: #fill in the columns index 
            for col in self.columns:
                self._columns_idx.append(col_idx_map[col])
        #print('columns idx:', self._columns_idx) #debug

        n_idx = col_idx_map['N']
        total_population: float = np.sum(current_state[:, n_idx])

        if total_population != 0:
            if self.freq_dep:
                betas = self.betas/total_population
            else:
                betas = self.betas
        else:
            betas = self.betas

        #print('betas:', betas) #debug

        #get number of infections for each strain type
        i_mask = current_state[:, self._columns_idx] == self._i_code
        #print('i_mask:', i_mask) #debug

        infectious_each_type = np.sum((i_mask * current_state[:, n_idx, np.newaxis]), axis=0)

        if np.sum(infectious_each_type) == 0: #no one gets infected in input data
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
        
        r_mask = (current_state[:, self._columns_idx] == self._r_code)
        #print('r_mask:', r_mask) #debug

        row_beta_mult = 1- np.max(r_mask[:, np.newaxis, :] * self.cross_protect, axis=2) #pull the max from axis=2, along the rows
        #print('row_beta_mult:', row_beta_mult)

        s_mask = (current_state[:, self._columns_idx] == self._s_code)
        #print('s_mask:', s_mask) #debug

        row_beta = row_beta_mult * betas * s_mask
        #print('row beta before is\n ', row_beta) #debug

        row_beta = row_beta * (1 - np.max((current_state[:, self._columns_idx] == self._i_code), axis=1))[:, np.newaxis]
        #print('row beta after is\n ', row_beta) #debug

        prI = 1 - np.power(np.exp(-dt * row_beta), infectious_each_type)
        #print('prI is\n', prI) #debug

        prI_sum_mask = np.sum(prI[:, :], axis=1) > 0
        #print('prI_sum_mask:', prI_sum_mask) #debug

        deltas = current_state[prI_sum_mask]
        #print('deltas\n', deltas) #debug

        prI_filter = prI[prI_sum_mask]
        #print('prI_filter\n', prI_filter) #debug

        prI_filter_sum = np.sum(prI_filter, axis=1)
        #print('prI_filter_sum\n', prI_filter_sum)

        if stochastic:
            count = deltas.shape[0] #need to verify if ndim is the always-workable way to get the correct num of rows from deltas_vec
            count_start = 0
            count_end = count
            np.random.seed(3)

            # Preallocate a reusable temporary buffer
            tmp_arr = np.empty_like(deltas[0]) #save data from deltas
            tmp_local = np.empty_like(deltas[0]) #save data from tmp_arr
            for i in range(prI_filter.shape[0]):
                # Copy only once into reusable buffer
                np.copyto(tmp_arr, deltas[i])
                
                #print('tmp_arr\n', tmp_arr)
                #print('prI_filter', prI_filter[i])
    
                changed_N = np.random.multinomial(deltas[i, n_idx], np.append(prI_filter[i], 0))
                #print('changed_N:', changed_N)
                tmp_arr[n_idx] = np.sum(-changed_N[:-1]) #the negative records, excluding the last value
                #print('1st new tmp_arr\n', tmp_arr)
                result_buffer[i:i+1, :] = tmp_arr#consider to remove tmp_arr but use preallocation directly to save memory usage
                #print('result_buffer\n', result_buffer)
                for j in range(prI_filter.shape[1]): #positive records
                    count_start = count_end
                    count_end = count_start + 1
                    np.copyto(tmp_local, tmp_arr) #data copied into preexisting tmp_local
                    # Modify the same tmp_arr in-place for reuse
                    tmp_local[self._columns_idx[j]] = self._inf_to_code
                    tmp_local[n_idx] = changed_N[j]
                    #print('tmp_local\n', tmp_local)
                    
                    result_buffer[count_start:count_end, :] = tmp_local
                    #print('final result_buffer\n', result_buffer)
        
        else:
            changed_N = -deltas[:, n_idx] * (1 - np.prod(1 - prI_filter, axis=1))
            count = len(changed_N)
            count_start = 0
            count_end = count

            # Fill first block directly from deltas (no copy)
            np.copyto(result_buffer[count_start:count_end, :], deltas)
            #result_buffer[count_start:count_end, :] = deltas
            result_buffer[count_start:count_end, n_idx] = changed_N
            #print('result buffer\n', result_buffer)

            # Preallocate a temporary working buffer (same shape as deltas)
            tmp_arr = np.empty_like(deltas)

            for i, col in enumerate(self._columns_idx):
                # Copy deltas into tmp_arr in place
                np.copyto(tmp_arr, deltas)

                tmp_arr[:, col] = self._inf_to_code
                #print('tmp arr\n', tmp_arr)
                
                # Compute comp in place (reuse a 1D scratch buffer)
                comp = np.empty_like(changed_N)
                np.divide(prI_filter[:, i], prI_filter_sum, out=comp)
                np.multiply(-changed_N, comp, out=comp)
                #comp = -changed_N * (prI_filter[:, i]/prI_filter_sum)
                #print('comp:', comp)
                
                tmp_arr[:, n_idx] = comp
                count_start = count_end
                count_end = count_end + count
                #print('new tmp_arr\n', tmp_arr)
                #result_buffer[count_start:count_end, :] = tmp_arr
                np.copyto(result_buffer[count_start:count_end, :], tmp_arr)

                #print('result_buffer\n', result_buffer)


        return result_buffer[:count_end, :] #include all rows with 0 for now
    

    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.MultiStrainInfectiousProcess_Vec_Encode': self.model_dump()
        }

        return rc


    #set up a property to return all the required categories used in trait_col
    @property
    def columns_all(self) -> list[str]: 
        return self.columns_all_categories


