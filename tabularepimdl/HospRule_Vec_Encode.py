import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr

from tabularepimdl.Rule import Rule
from tabularepimdl._types.constrained_types import UniqueNonEmptyStrList

class HospRule_Vec_Encode(Rule, BaseModel):
    '''
    This rule takes multiple columns. You have some risk of hospitalization if infected from any column,
    but that probability is reduced if you are recovered in any column. We will additionally track which strain you were
    hospitalized with. Only tracking total hospitalizations.
    
    Attributes:
        strain_cols: the strain columns with infection states.
        hosp_cols: hospitalization columns with hospitalized states.
        strain_cols_all_categories: all the infection categories used in strain_cols.
        hosp_cols_all_categories: all the hospitalization categories used in hosp_cols.
        infect_status: the state for infectious.
        recover_status: the state for recovery.
        hosp_status: the state for hospitalization.
        prim_hrate: chance of being hospitalized from a primary infection.
        sec_hrate: chance of being hospitalized from a secondary infection.
        stochastic: whether the process is stochastic or deterministic.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S'].
    '''

    ''' Presume that hosp cols is of the same length as the strain cols, and they have corresponding indices. '''
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strain_cols: UniqueNonEmptyStrList = Field(description = "the strain columns with infection states.")
    hosp_cols: UniqueNonEmptyStrList = Field(description = "hospitalization columns with hospitalized states.")
    strain_cols_all_categories: UniqueNonEmptyStrList = Field("all the infection categories used in strain_cols.")
    hosp_cols_all_categories: UniqueNonEmptyStrList = Field("all the hospitalization categories used in hosp_cols.")
    infect_status: str = Field(default="I", description = "the state for infectious.")
    recover_status: str = Field(default="R", description = "the state for recovery.")
    hosp_status: str = Field(default="H", description = "the state for hospitalization.")
    prim_hrate: float = Field(ge=0, description = "chance of being hospitalized from a primary infection.")
    sec_hrate: float = Field(ge=0, description = "chance of being hospitalized from a secondary infection.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    infstate_compartments: UniqueNonEmptyStrList = Field(description = "the infection compartments used in epidemics.")

    _strain_columns_idx: list[int] = PrivateAttr(default_factory=list) #the column index for each strain column
    _strain_columns_all_categories_code: list[int] | None = PrivateAttr(default_factory=None) #the numerical codes for all categories used by all strain columns
    _hosp_columns_idx: list[int] = PrivateAttr(default_factory=list) #the column index for each hosp column
    _hosp_columns_all_categories_code: list[int] | None = PrivateAttr(default_factory=None) #the numerical codes for all categories used by all hosp columns

    _infect_status_code: int | None = PrivateAttr(default=None)
    _recover_status_code: int | None = PrivateAttr(default=None)
    _hosp_status_code: int | None = PrivateAttr(default=None)
    _h_code: int | None = PrivateAttr(default=None) #may not be needed
    _u_code: int | None = PrivateAttr(default=None) #may not be needed

    
    @model_validator(mode="after")
    def validate_number_of_elements_(self):
        """Ensure then number of elements in strain_cols and hosp_cols are the same."""
        if len(self.strain_cols) != len(self.hosp_cols): #convert list to array
            raise ValueError(f"Expecte the same number of elements in strain_cols and hosp_cols "
                             f"received {len(self.strain_cols)} strain_cols and {len(self.hosp_cols)} hosp_cols.")
        return self

    def model_post_init(self, _):
        """
        Encode the input states based on each column's attribute values.

        Returns:
            Numerical values of encoded infection states, recover states and hosp states.

        Notes:
            infstate_to_int (dict): A placeholder (not being used in this rule). Mapping of infection states of infstate_compartments to their index positions.
        """
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}, not used in this rule
        
        self._strain_columns_all_categories_code = {v: i for i, v in enumerate(sorted(self.strain_cols_all_categories))} #encode each category
        self._hosp_columns_all_categories_code = {v: i for i, v in enumerate(sorted(self.hosp_cols_all_categories))} #encode each category

        #input data columns should be values like strain type or hospitalization type, but not 'infstate'
        self._infect_status_code = self._strain_columns_all_categories_code.get(self.infect_status)
        self._recover_status_code = self._strain_columns_all_categories_code.get(self.recover_status)
        self._hosp_status_code = self._hosp_columns_all_categories_code.get(self.hosp_status)

    def combination_of_input_states(self) -> int: 
        """
        Return the number of combinations of different input states of the rule.
        """
        return len(self.strain_cols_all_categories)*len(self.infstate_compartments)

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

        #cache column indices (once)
        if not self._strain_columns_idx: #fill in the infection columns index 
            for col in self.strain_cols:
                self._strain_columns_idx.append(col_idx_map[col])
        #print('strain_cols idx:', self._strain_columns_idx) #debug

        if not self._hosp_columns_idx: #fill in the hosp columns index 
            for col in self.hosp_cols:
                self._hosp_columns_idx.append(col_idx_map[col])
        #print('hosp_cols idx:', self._hosp_columns_idx) #debug

        n_idx = col_idx_map["N"]
        
        #identify infected rows under each strain
        i_mask = (current_state[:, self._strain_columns_idx] == self._infect_status_code)
        has_infection = np.any(i_mask, axis=1)

        if not np.any(has_infection): #no infections
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)

        #strain index per row
        strain_index_with_infection = np.argmax(i_mask, axis=1)
        #print('strain_index_with_infection, i.e. match index\n', strain_index_with_infection)

        #Determine hospitalization rate
        r_mask = (current_state[:, self._strain_columns_idx] == self._recover_status_code)
        has_recovered = np.any(r_mask, axis=1)
        #If people are recovered from a Strain, then hosp_rate=sec_hrate, otherwise hosp_rate=prim_hrate
        hosp_rate = np.where(has_recovered, self.sec_hrate, self.prim_hrate)
        #print('hosp rate\n', hosp_rate)

        #Filter people that are infected and can be hospitalized
        infected_that_qualify_hospitalization = np.where(has_infection)[0]
        rows_with_infected = current_state[infected_that_qualify_hospitalization]
        #print('rows with infect\n', rows_with_infected)

        rate_const = 1.0 - np.exp(-dt * hosp_rate[infected_that_qualify_hospitalization])

        if stochastic:
            changed_N = -np.random.binomial(rows_with_infected[:, n_idx], rate_const)
        else:
            changed_N = -rows_with_infected[:, n_idx] * rate_const

        count = rows_with_infected.shape[0]
        #print('count:', count)

        result_buffer[:count, :] = rows_with_infected
        result_buffer[:count, n_idx] = changed_N
        #print('array decobs\n', result_buffer[:count, :])

        result_buffer[count:2*count, :] = rows_with_infected
        result_buffer[count:2*count, n_idx] = -changed_N
        #print('array incobs before change\n', result_buffer[count:2*count, :])

        hosp_block = result_buffer[count:2*count, :]
        hosp_columns_idx_arr = np.array(self._hosp_columns_idx)
        hosp_map_strain = hosp_columns_idx_arr[strain_index_with_infection[infected_that_qualify_hospitalization]]
        hosp_block[np.arange(count), hosp_map_strain] = self._hosp_status_code
        #print('array incobs after change\n', result_buffer[count:2*count, :])

        return result_buffer[:2*count, :]

                
    def to_dict(self) -> dict:
        """
        Save the rule's attributes and their associated values to a dictionary.
        
        Returns:
            Rule attributes in a dictionary.
        """
        rc = {
            'tabularepimdl.HospRule_Vec_Encode': self.model_dump()
        }

        return rc
    

    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required infection compartments if any column in this rule takes 'infstate' value.
        """
        return self.infstate_compartments
    
    #set up a property to return all the required categories used in strain_cols
    @property
    def strain_cols_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required categories the `strain_cols` uses.
        """
        return self.strain_cols_all_categories
    
    #set up a property to return all the required categories used in hosp_cols
    @property
    def hosp_cols_all(self) -> list[str]:
        """
        Used and checked by the model engine to update input data's domain values.

        Returns:
            A list of strings of all the required categories the `hosp_cols` uses.
        """
        return self.hosp_cols_all_categories