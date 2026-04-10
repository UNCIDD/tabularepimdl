import numpy as np
from pydantic import BaseModel, Field, ConfigDict, ValidationInfo, field_validator, model_validator, PrivateAttr

from tabularepimdl.Rule import Rule
from tabularepimdl._types.constrained_types import UniqueNonEmptyStrList
from tabularepimdl._validators.rule_domain_membership_validator import domain_membership_validator


class MultiStrainInfectiousProcess_Vec_Encode(Rule, BaseModel):
    """
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
    columns: UniqueNonEmptyStrList = Field(description = "the strain columns for infection state.")
    columns_all_categories: UniqueNonEmptyStrList = Field(description = "all the infection state categories the strain columns should have.")
    cross_protect: np.ndarray = Field(description = "a N(strain)*N(strain) matrix of cross protections.")
    s_st: str = Field(default="S", description = "the state for susceptibles.")
    i_st: str = Field(default="I", description = "the state for infectious.")
    r_st: str = Field(default="R", description = "the state for immune/recovered.")
    inf_to: str = Field(default="I", description = "the state infectious folks go to.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    freq_dep: bool = Field(default=True, description = "whether this model is a frequency dependent model.")
    infstate_compartments: UniqueNonEmptyStrList = Field(description = "the infection compartments used in epidemics.")

    _s_code: int | None = PrivateAttr(default=None)
    _i_code: int | None = PrivateAttr(default=None)
    _r_code: int | None = PrivateAttr(default=None)
    _inf_to_code: int | None = PrivateAttr(default=None)
    _columns_idx: list[int] = PrivateAttr(default_factory=list)
    _columns_all_categories_code: list[int] | None = PrivateAttr(default_factory=None)
    _state_encoding_by_engine : bool = PrivateAttr(default=False)

    _check_domain_membership = domain_membership_validator(
            attribute_fields = ("s_st", "i_st", "r_st", "inf_to"),
            domain_fields = ("columns_all_categories", "infstate_compartments")
        )

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
            raise ValueError(f"All elements in {field.field_name} must contain numeric data, received data type {array_parameters.dtype}.")
        
        #3. check if all elements are non-negative values.
        if np.any(array_parameters < 0):
            raise ValueError(f"All elements in {field.field_name} must be non-negative, received {array_parameters}.")
        
        #4. check for NaN or Inf.
        if np.isnan(array_parameters).any() or np.isinf(array_parameters).any():
            raise ValueError("Arrays must not contain NaN or Infinity values.")
        
        return array_parameters

    @model_validator(mode="after") #after all fields are validated, check cross fields relationship
    def check_dimensions(self):
        """Ensure betas and cross_protect have matching dimensions."""
        if len(self.columns) != len(self.betas):
            raise ValueError(f"The number of 'columns' ({len(self.columns)}) must match the number of 'betas' ({len(self.betas)}).")

        if self.cross_protect.shape[0] != self.cross_protect.shape[1] or self.cross_protect.shape[0] != len(self.betas):
            raise ValueError(
                f"'cross_protect' must be a square matrix of size {len(self.betas)}x{len(self.betas)}, received {self.cross_protect.shape}."
            )
        
        return self

    def model_post_init(self, _):
        """
        Encode the input states based on each column's attribute values.
        
        Returns:
            Numerical values of encoded infection states, recover states and hosp states.
        
        Notes:
            - infstate_to_int (dict): A placeholder (not being used in this rule). Mapping of infection states of infstate_compartments to their index positions.
            - Retain rule-level state encoding to support users who test rules individually.

        """
        if not self._state_encoding_by_engine:
            infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))} #placeholder
        
            self.columns_all_categories = sorted(self.columns_all_categories) #sort the columns' all categories
            self._columns_all_categories_code = {v: i for i, v in enumerate(self.columns_all_categories)} #encode each category

            #input data columns should be values like strain type, but not 'infstate'
            self._s_code = self._columns_all_categories_code.get(self.s_st)
            self._i_code = self._columns_all_categories_code.get(self.i_st)
            self._r_code = self._columns_all_categories_code.get(self.r_st)
            self._inf_to_code = self._columns_all_categories_code.get(self.inf_to)
        else:
            pass

    #set up a property to return all the required categories used in columns
    @property
    def columns_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required categories the `columns` uses.
        """
        return self.columns_all_categories

    @property
    def expansion_factor(self) -> int:
        """Maximum number of rows this rule can return per input rows."""
        return max(len(self.columns_all_categories), len(self.infstate_compartments))

    def _encode_categorical_states(self, data_domains) -> None:
        """
        Use the fully updated data columns' domain mapping values to encode rule's own column state values.
        """
        single_strain_column = self.columns[0]
        mapping = data_domains[single_strain_column]
        self._s_code = mapping[self.s_st]
        self._i_code = mapping[self.i_st]
        self._r_code = mapping[self.r_st]
        self._inf_to_code = mapping[self.inf_to]

        self._state_encoding_by_engine = True

    
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
            raise ValueError(f"Missing required columns in current_state: {required_columns}.")
        
        if stochastic is None:
            stochastic = self.stochastic
        
        if not self._columns_idx: #fill in the columns index 
            for col in self.columns:
                self._columns_idx.append(col_idx_map[col])
        #print('columns idx:', self._columns_idx) #debug

        n_idx = col_idx_map['N']
        total_population: float = np.sum(current_state[:, n_idx])

        if total_population != 0 and self.freq_dep:
            betas = self.betas / total_population
        else:
            betas = self.betas
        #print('betas:', betas) #debug

        #get number of infections for each strain type
        i_mask = current_state[:, self._columns_idx] == self._i_code
        #print('i_mask:', i_mask) #debug

        infectious_each_type = np.sum((i_mask * current_state[:, n_idx, np.newaxis]), axis=0)

        if np.sum(infectious_each_type) == 0: #no one gets infected in input data
            #print('no infection, return empty array')
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
            count_end = 0
            #np.random.seed(3) #test purpose
            for i in range(prI_filter.shape[0]):
                tmp_arr = deltas[i].copy()
                #print('tmp_arr\n', tmp_arr)
                #print('prI_filter', prI_filter[i])
    
                changed_N = np.random.multinomial(deltas[i, n_idx], np.append(prI_filter[i], 0))
                #print('changed_N:', changed_N)
                tmp_arr[n_idx] = np.sum(-changed_N[:-1]) #the negative records, excluding the last value
                #print('new tmp_arr\n', tmp_arr)
                result_buffer[i:i+1, :] = tmp_arr#consider to remove tmp_arr but use preallocation directly to save memory usage
                #print('result_buffer\n', result_buffer)
                for j in range(prI_filter.shape[1]): #positive records
                    count_start = count
                    count_end = count_start + 1
                    count = count + 1
                    to_add = tmp_arr.copy()
                    #print('to_add\n', to_add)
                    to_add[self._columns_idx[j]] = self._inf_to_code
                    to_add[n_idx] = changed_N[j]
                    #print('new to_add\n', to_add)
                    result_buffer[count_start:count_end, :] = to_add
                    #print('final result_buffer\n', result_buffer)
        
        else:
            changed_N = -deltas[:, n_idx] * (1 - np.prod(1 - prI_filter, axis=1))
            count = len(changed_N)
            count_start = 0
            count_end = count
            result_buffer[count_start:count_end, :] = deltas
            result_buffer[count_start:count_end, n_idx] = changed_N
            #print('result buffer\n', result_buffer)

            for i, col in enumerate(self._columns_idx):
                tmp_arr = deltas.copy()
                tmp_arr[:, col] = self._inf_to_code
                #print('tmp arr\n', tmp_arr)
                comp = -changed_N * (prI_filter[:, i]/prI_filter_sum)
                #print('comp:', comp)
                tmp_arr[:, n_idx] = comp
                count_start = count_start + count
                count_end = count_end + count
                #print('new tmp_arr\n', tmp_arr)
                result_buffer[count_start:count_end, :] = tmp_arr
                #print('result_buffer\n', result_buffer)


        return result_buffer[:count_end, :] #include all rows with 0 for now
    

    def to_dict(self) -> dict:
        """
        Save the rule's attributes and their associated values to a dictionary.
        
        Returns:
            Rule attributes in a dictionary.
        """
        rc = {
            'tabularepimdl.MultiStrainInfectiousProcess_Vec_Encode': self.model_dump()
        }

        return rc


