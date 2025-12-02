from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from tabularepimdl.Rule import Rule
from typing import Iterable


class EpiModel_Vec_Encode_1(BaseModel):
    """! Class that that applies a list of rules to a changing current state through 
    some number of time steps to produce an epidemic. It has attributes representing the initial state,
    current state and the full epidemic thus far.

    #Public Attributes
    @param init_state: a data frame with the initial epidemic state. Must have at minimum columns T and N.
        Location	Age	    InfState	N	T
    0	A	        adult	S	        100	0.0
    1	A	        adult	I	        5	0.0
    2	A	        child	R	        50	0.0
    3	B	        adult	S	        80	0.0
    
    @para current_state_array: a numpy array (at the moment) with the current epidemic state.
    @para full_epi_array: a numpy array contains full epidemic history.
    @param rules: a list of epidemic rules that will represent the epidemic process. Must be a list of lists. [[B], [SI, ST], [W]]
    @param stoch_policy: whether the entire epidemic process is rule based or centralized with either deterministic or stochastic.
    @param compartment_col: a string indicating the column name that is used for saving infection compartments.

    #Private Attributes
    ...to be added
    """

    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    init_state: pd.DataFrame = Field(default_factory=pd.DataFrame)
    current_state_array: np.ndarray = Field(default_factory=lambda: np.array([]))
    full_epi_array: np.ndarray = Field(default_factory=lambda: np.array([]))
    rules: list[list[Rule]]
    stoch_policy: str = "rule_based"
    #compartment_col: str = 'InfState' #model engine does not need this attribute

    #columns listed in the input data, separating column N and T from other columns
    _agg_cols: set[str] = PrivateAttr(default_factory = lambda:{'N', 'T'})
    _grouping_cols: list[str] = PrivateAttr(default_factory = lambda:['InfState']) #e.g. ['InfState', 'Age', 'Location']

    #domains for each column
    _domains: dict[str, Any] = PrivateAttr(default_factory=dict) #e.g. {'InfState': {'I', 'R', 'S'}, 'Location': {'A', 'B'}, 'Age': {'adult', 'child'}}

    #domains for column InfState
    _infstate_all_comps: list[str] = PrivateAttr(default_factory=list) #e.g. ['S', 'I', 'R']
    _num_comps: int = PrivateAttr(default=None) #e.g. 3

    #encoding map and inverse encoding map for infstate comp, not needed, keep them here for now
    #these two variables are not set in the class since _grouping_col_map and _inverse_grouping_col_map do the same function
    #_infstate_comp_map: Dict[str, int] = PrivateAttr(default_factory=dict) #e.g. {'S': 0, 'I': 1, 'R': 2}
    #_inverse_infstate_comp_map: Dict[int, str] = PrivateAttr(default_factory=dict) #e.g. {0: 'S', 1: 'I', 2: 'R'}

    #encoding maps and inverse encoding maps for each grouping column
    _grouping_col_map: dict[str, dict[str, int]] = PrivateAttr(default_factory=dict) #e.g. { 'InfState': {'I': 0, 'S': 1}, 'Age': {'adult': 0, 'child': 1} }
    _inverse_grouping_col_map: dict[str, dict[int, str]] = PrivateAttr(default_factory=dict) #e.g. { 'InfState': {0: 'I', 1: 'S'}, 'Age': {0: 'adult', 1: 'child'} }
    
    #column names order in init_state
    _init_state_col_order: list[str] = PrivateAttr(default_factory=list) #e.g. ['InfState', 'N', 'T', 'Group']
    _col_idx_map: dict[str, int] = PrivateAttr(default_factory=dict) #e.g. {'InfState' : 0, 'N': 1, 'T': 2}

    #column index value of InfState, N, T and all grouping columns including InfState
    #_infstate_idx: int = PrivateAttr(default=None) #e.g. _infstate_idx=0 #model engine does not need this attribute
    _n_idx: int = PrivateAttr(default=None) #e.g. _n_idx=1
    _t_idx: int = PrivateAttr(default=None) #e.g. _t_idx=2
    _grouping_col_idx: list[int] = PrivateAttr(default_factory=list) #e.g. infstate, x, y, z = [0, 3, 4, 5]

    #current result array pr-eallocation
    _current_result_preallocation: np.ndarray = PrivateAttr(default_factory=lambda: np.array([]))

    #full epi list to contain full epi array
    _full_epi_list: list[np.ndarray] = PrivateAttr(default_factory=list)

    #initial current_state_array, used to save the converted initial current_state_array
    _initial_current_state_array: np.ndarray = PrivateAttr(default_factory=lambda: np.array([])) 
    
    @field_validator("init_state", mode="before") #by default, column N and T should be numerical values, if not then should raise ValueError and let users check
    @classmethod
    def validate_init_state(cls, initial_state) -> pd.DataFrame: 
        if not isinstance(initial_state, pd.DataFrame): #check if init_state is a dataFrame
            raise TypeError(f"Expected a DataFrame, but got {type(initial_state).__name__} instead.")
        required_cols = {'N', 'T'}
        missing = required_cols - set(initial_state.columns)
        if missing: #check if column T and N are in the dataframe
            raise ValueError(f"init_state is missing required columns: {missing}.")
        return initial_state
    
    #Omitting rules list check for now

    def model_post_init(self, _):
        """
        Initialization of init_state column order, internal attributes, current_state_arrays and full_epi_array.
        """
        self._init_state_column_values_grouping() #grouping init_state column values
        self._init_state_column_order_shuffle() #shuffle init_state column order
        self._update_col_domain_values_from_rules() #update each rule's selected column's unique domain values
        self._setup_internal_attributes() #set up all internal attributes
        self._convert_init_df_to_cur_arrays() #initalize current_state_array only
        self._save_initial_current_state_array()
        self._initalize_full_epi_array() #initalize full_epi_array only


    def _init_state_column_values_grouping(self):
        """
        Grouping each column of init_state and aggregate column N and T in case the input raw data has duplicate rows.
        This ensures rules that have categorical column(s) can accurately check the input raw data aginst their input categories.
        """
        #collect column names for aggregating columns and rest grouping columns
        self._agg_cols = {'N', 'T'}
        self._grouping_cols = [c for c in self.init_state.columns if c not in self._agg_cols]
        #print('grouping col:', self._grouping_cols)
        #print('before grouping init state\n', self.init_state) #debug
        #grouping column values, only the categories that are actually present in the data will be included in the groups.
        self.init_state = self.init_state.groupby(self._grouping_cols, observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index()
        #print('after grouping init state\n', self.init_state) #debug


    def _init_state_column_order_shuffle(self):
        """
        Move column N and T to the last two columns in init_state before all internal attributes and data processing steps occure.
        """
        cols = self.init_state.columns.tolist()

        # Define the target columns to move
        cols_to_move = ['N', 'T']

        # Filter out the columns to move from the list (preserves order of others)
        remaining_cols = [col for col in cols if col not in cols_to_move]

        # Build the new column order
        new_col_order = remaining_cols + cols_to_move

        # Reorder the DataFrame init_state
        self.init_state = self.init_state[new_col_order]
        #print('shuffle init state\n', self.init_state) #debug


    def _match_domain_key(self, col_name: str) -> str | None:
        """Case-insensitive lookup: return the matching key for column names of input data or None."""
        if col_name is None:
            return None
        col_name_lower = col_name.lower()
        for key in self._domains.keys():
            if key.lower() == col_name_lower:
                return key
        return None


    def _update_col_domain_values_from_rules(self) -> None:
        """
        Walk through rules list and update each rule's selected column's unique domain values in place.

        Rules must be instances of BaseModel (Pydantic), and rule attributes
        that include keyword 'col' in the attribute name will be inspected.
        """
        #unique domain values per grouping column (excludes N and T) in init_state
        self._domains = {col: set(self.init_state[col].astype(str).tolist()) for col in self._grouping_cols}
        #print('initial domain value:', self._domains)
        #collect domain values that exist in each rule's properties but not in init_state data columns
        for ruleset in self.rules:
            for rule in ruleset:
                rule_field_name_set = list(rule.model_fields_set) #return the set of fields that have been explicitly set on the rule instance
                
                for attribute_name in rule_field_name_set: #iterate all attributes that have been explicitly set in the rule instance
                    if 'col' not in attribute_name.lower(): #search keyword 'col' in rule's attribute names
                        continue

                    try:
                        attribute_value = getattr(rule, attribute_name) #if attribute_name has 'col', then obtain its value
                        #print(attribute_name, 'value :', attribute_value)
                    except Exception:
                        continue

                    # Normalize: always work with a list of column names
                    if isinstance(attribute_value, str):
                        col_names = [attribute_value]
                    elif isinstance(attribute_value, list):
                        col_names = attribute_value
                    else:
                        raise ValueError(f"Expect attribute value to be a string or list, received {type(attribute_value)}.")
                        
                    
                    for col in col_names:
                        data_col_name = self._match_domain_key(col_name = col) #check if attribute value exists in domain keys
                        #print('found data col name in domain:', data_col_name)

                        if data_col_name is None: #attribute name has 'col' but its value is not in domain keys
                            continue

                        if data_col_name.lower() == "infstate": #Special case: if column value equals 'infstate' (case-insensitive)
                            try:
                                infstate_all_compartments = getattr(rule, 'infstate_all')
                                #print('infstate full:', infstate_all_compartments)
                            except Exception:
                                infstate_all_compartments = None

                            if infstate_all_compartments:
                                if isinstance(infstate_all_compartments, (list, set, tuple, Iterable)):
                                    self._domains[data_col_name].update(infstate_all_compartments)
                                    #print('domain_values:', self._domains)

                        else: #Generic case: look for property named "<data_col_name>_all"
                            property_name = f"{attribute_name}_all"
                            if hasattr(rule, property_name):
                                try:
                                    property_value = getattr(rule, property_name)
                                    #print('property_value:', property_value)
                                except Exception:
                                    property_value = None
            
                                if property_value:
                                    if isinstance(property_value, (list, set, tuple, Iterable)):
                                        self._domains[data_col_name].update(property_value)
                                        #print('domain_values:', self._domains)
       
        #print('final domains per column:', self._domains) #debug


    def _setup_internal_attributes(self):
        """
        Set up internal attributes with values from init_state and rule list.
        """
        #self._infstate_all_comps = sorted(self._domains[self.compartment_col]) #keep a variable to save column InfState's compartment values
        #self._num_comps = len(self._infstate_all_comps) #keep a variable to save the number of compartments in column InfState
        #print('infstate_all_comps:', self._infstate_all_comps, 'num_comps:', self._num_comps)

         #create compartment and its associated index mapping, and reverse the mapping
         #used for converting column's string values to numbers
        #self._infstate_comp_map = {comp: i for i, comp in enumerate(infstate_all_comps)}
        #print('infstate_comp_map:', self._infstate_comp_map)
        #self._inverse_infstate_comp_map = {i: comp for comp, i in self._infstate_comp_map.items()}
        #print('inverse_infstate_comp_map:', self._inverse_infstate_comp_map)
        
        #build encoding maps and inverse encoding maps for each grouping column's domain values including column infstate
        for col in self._grouping_cols:
            #print('col:', col)
            vals = sorted(self._domains[col]) #sort each grouping column's unique domain values
            #print('vals:', vals)
            self._grouping_col_map[col] = {v: i for i, v in enumerate(vals)} #encode each grouping column's values
            self._inverse_grouping_col_map[col] = {i: v for v, i in self._grouping_col_map[col].items()} #reverse the above encoding
        #print('grouping col map:', self._grouping_col_map) #debug
        #print('inverse grouping col map:', self._inverse_grouping_col_map)

        #fetch column order of init_state
        self._init_state_col_order = [col for col in self.init_state.columns] #get all column names into a list, e.g. ['InfState', 'N', 'T']
        self._col_idx_map = {col: i for i, col in enumerate(self._init_state_col_order)} #e.g. {'Location': 0, 'Age': 1, 'InfState': 2, 'N': 3, 'T': 4}
        #print('col_idx_map:', self._col_idx_map) #debug

        #all columns indicies are included in _col_idx_map, extract invidual ones for separate use
        #Locate each column's index of init_state, used in array column operation.
        #self._infstate_idx = self._col_idx_map[self.compartment_col] #unused/unneeded attribute
        self._n_idx = self._col_idx_map['N'] #the code has to know/use a few fixed column names in order to get column indicies
        self._t_idx = self._col_idx_map['T']
        self._grouping_col_idx = [self._col_idx_map[c] for c in self._grouping_cols]
        #print('_infsate_idx:', self._infstate_idx, '_n_idx:', self._n_idx, '_t_idx:', self._t_idx, '_grouping_col_idx:', self._grouping_col_idx)
        

    def _convert_init_df_to_cur_arrays(self) -> np.ndarray: #might be benificial to add array args to the method and return current_state and full_epi
        """
        Convert init_state dataframe to numpy array. Save the array to current_state_array.
        """
        
        #this code block is for debugging only
        #infstate_values = self.init_state[self.compartment_col].map(self._infstate_comp_map).to_numpy() #encode compartment strings with integers
        #n_values = self.init_state['N'].to_numpy()
        #t_values = self.init_state['T'].to_numpy()

        #self.current_state_array = np.column_stack((infstate_values, n_values, t_values))
        #self.current_state_array = self.current_state_array.astype(np.float64)
        #print('converted input array\n', self.current_state_array)

        #process each column individually (including N and T) and build a current_state array without modifying original init_state
        #by default, column N and T should already be numerical values and verified in model instantiation stage
        encoded_columns = [] #to save each grouping column's converted array based on the column values
        for col in self._init_state_col_order:
            if col in self._grouping_col_map: #process grouping cols
                col_array = self.init_state[col].map(self._grouping_col_map[col]).to_numpy()
            else: #process column N, T
                col_array = self.init_state[col].to_numpy()
            encoded_columns.append(col_array)
        
        if encoded_columns: #check encoded columns empty or not and convert all values to float64
            self.current_state_array = np.column_stack(encoded_columns).astype(np.float64)
        else:
            self.current_state_array = np.empty((0, len(self._init_state_col_order)), dtype=np.float64)

        #pre-allocation of result array -- to be checked/verified
        n_rows = self.current_state_array.shape[0] #detect the number of rows and columns in current_state_array
        n_cols = self.current_state_array.shape[1]
        self._current_result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
        #print('initial preallocation buffer\n', self._current_result_preallocation)
        return self.current_state_array #return current_state_array only
    
    def _save_initial_current_state_array(self) -> np.ndarray:
            """
            After init_state dataframe is converted to current_state_array, save the array's initial values.
            """
            self._initial_current_state_array = self.current_state_array.copy()
            
    def _initalize_full_epi_array(self) -> np.ndarray:
        """
        Initialize full_epi_array with current_state_array.
        """
        self._full_epi_list = [self._initial_current_state_array] #make current_state_array the 1st elment in full_epi_array
        self.full_epi_array = np.vstack(self._full_epi_list)
        return self.full_epi_array
    
    def _covnert_list_of_arrays_to_df(self, list_of_arr: list[np.ndarray]) -> pd.DataFrame:
        """
        Convert list of arrays returned from do_timestep() to a dataframe as full epidemic history.
        @param list_of_arr: a list of arrays, each array is a epidemic historical data.
        return: a dataframe containing full epidemic history.
        """
        # _full_epi_list could be used to replace list_of_arr and remove the this argument from method definition
        if len(list_of_arr) == 0: #return empty dataframe if input list of arrays is empty
            return pd.DataFrame(columns=self._init_state_col_order)
        self.full_epi_array = np.vstack(list_of_arr)
        df_reconstructed = pd.DataFrame(self.full_epi_array, columns=self._init_state_col_order)
        
        for col in self._inverse_grouping_col_map: #convert grouping col's numeric values to domain values (not including N and T)
            df_reconstructed[col] = df_reconstructed[col].astype(np.float64).map(self._inverse_grouping_col_map[col])
        return df_reconstructed

    def Reset(self):
        """
        Reset the values of current_state_array and full_epi_array to be init_state values.
        only call back initial_current_state_array value and invoke _initalize_full_epi_array().
        """
        self.current_state_array = self._initial_current_state_array.copy()
        self._initalize_full_epi_array()

    def do_timestep(self, dt: int | float =1.0):
        """
        Does a timestep process, updating the epidemic current state by applying each epidemic rule to the current state data.
        If in cycles of simulation, appends each iteration's current state to the full epidemic history.
        @param dt: the time step.
        """
        
        for ruleset in self.rules:
            ruleset_deltas_list = []
            for rule in ruleset:
                #print('current rule:', rule)
                #print('in rule Before loop preallocation buffer\n', self._current_result_preallocation)
                if self.stoch_policy == "rule_based":
                    rule_deltas = rule.get_deltas(current_state=self.current_state_array, col_idx_map=self._col_idx_map, result_buffer=self._current_result_preallocation, dt=dt)
                else:
                    rule_deltas = rule.get_deltas(current_state=self.current_state_array, col_idx_map=self._col_idx_map, result_buffer=self._current_result_preallocation, dt=dt, stochastic = (self.stoch_policy=="stochastic"))
                #print('rule detlas\n', rule_deltas)
                #print('in rule After loop preallocation buffer\n', self._current_result_preallocation)
                #print('before append, ruleset_deltas_list\n', ruleset_deltas_list)
                if rule_deltas is not None and len(rule_deltas) > 0: #add non-None rule_deltas to the list
                    ruleset_deltas_list.append(rule_deltas.copy())
                #print('after append, ruleset_deltas_list\n', ruleset_deltas_list)

            if len(ruleset_deltas_list) == 0: #if no data added to ruleset_deltas_list, go to next ruleset
                #print('go to next ruleset')
                continue

            ruleset_deltas_list.append(self.current_state_array) #add current_state_array to the list
            #print('after append cur_state_array, ruleset_deltas_list\n', ruleset_deltas_list)

            self.current_state_array = np.vstack(ruleset_deltas_list) #convert list of arrays to array and save it to current_state_array
            #print('before grouping, cur_array\n', self.current_state_array)

            #grouping columns, sum N for each group, pick max T out of all groups
            #process grouping columns
            grouping_col_arrays = self.current_state_array[:, self._grouping_col_idx] #arrays from each grouping colum
            unique_col_value, unique_value_inverse_idx = np.unique(grouping_col_arrays, axis=0, return_inverse=True) #unique values of each grouping column
            unique_value_inverse_idx = unique_value_inverse_idx.flatten() #indicies of unique values

            #process column N and T
            num_groups = unique_col_value.shape[0] #the number of unique groups for grouping purpose
            sum_col_N_per_group = np.zeros(num_groups, dtype=np.float64)
            max_col_T_per_group = np.full(num_groups, -np.inf)

            np.add.at(sum_col_N_per_group, unique_value_inverse_idx, self.current_state_array[:, self._n_idx]) #sum of N per group
            np.maximum.at(max_col_T_per_group, unique_value_inverse_idx, self.current_state_array[:, self._t_idx]) #max T per group
            max_col_T_per_group[:] = np.max(max_col_T_per_group) #global max T

            self.current_state_array = np.column_stack((unique_col_value, sum_col_N_per_group, max_col_T_per_group)) #order of cols is grouping_cols, N, T
            #print('after grouping, cur_array\n', self.current_state_array)

        self.current_state_array[:, self._t_idx] = self.current_state_array[:, self._t_idx] + dt #increase T value by dt

        #remove all rows where column N has a value of 0
        self.current_state_array = self.current_state_array[self.current_state_array[:, self._n_idx] != 0]
        
        #this code line may not be needed
        #self.cur_state = self.cur_state.assign(T=max(self.cur_state['T'])+dt) #T is forward with dt after each timestep iteration
            
        # append the updated current state to the epidemic history.
        if len(ruleset_deltas_list) != 0:
            self._full_epi_list.append(self.current_state_array) #this is a list of arrays
            #print('full epi list\n', self._full_epi_list)
        

