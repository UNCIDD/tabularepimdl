import pandas as pd
import numpy as np
import copy
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Optional

class EpiModel(BaseModel):
    """! Class that that applies a list of rules to a changing current state through 
    some number of time steps to produce an epidemic. It has attributes representing the initial state,
    current state and the full epidemic thus far.

    @param init_state: a data frame with the initial epidemic state. Must have at minimum columns T and N.
    @param cur_state: a data frame (at the moment) with the current epidemic state.
    @param full_epi: a data frame contains full epidemic history.
    @param rules: a list of epidemic rules that will represent the epidemic process. Must be a list of lists.
    @param stoch_policy: whether the entire epidemic process is rule based or centralized with either deterministic or stochastic.
    """
    
    #def __init__(self, init_state, rules:list, stoch_policy = "rule_based") -> None:
    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    init_state: pd.DataFrame
    cur_state: Optional[pd.DataFrame] = None
    full_epi: Optional[pd.DataFrame] = None
    rules: List[List[Rule]]
    stoch_policy: str = "rule_based"
    
    
    @field_validator("init_state", mode="before")
    @classmethod
    def validate_init_state(cls, initial_state) -> pd.DataFrame: 
        if not isinstance(initial_state, pd.DataFrame): #check if init_state is a dataFrame
            raise TypeError(f"Expected a DataFrame, but got {type(initial_state).__name__} instead.")
        required_cols = {"T", "N"}
        missing = required_cols - set(initial_state.columns)
        if missing: #check if column T and N are in the dataframe
            raise ValueError(f"init_state is missing required columns: {missing}")
        return initial_state
    
    @field_validator("rules", mode="before")
    @classmethod
    def validate_rules_list(cls, input_rules) -> List[List[Rule]]: #check if the rules is a list or list of lists
        # Step 1: Wrap single Rule instance
        if isinstance(input_rules, Rule):
            return [[input_rules]]

        # Step 2: Ensure input is list-like
        if not isinstance(input_rules, list):
            raise TypeError(f"rules must be an epidemic Rule or a list (or list of lists) of epidemic Rules. Received {type(input_rules).__name__}.")
        
        normalized_list = []
        for i, item in enumerate(input_rules):
            if isinstance(item, Rule):
                normalized_list.append([item]) #Single rule instance: wrap it in list
            elif isinstance(item, list): # Sublist: validate contents
                for j, subitem in enumerate(item):
                    if isinstance(subitem, Rule):
                        continue
                    elif isinstance(subitem, list):
                        raise TypeError(
                            f"Too much nesting at input rules[{i}][{j}], element is {subitem} with type {type(subitem).__name__}. " 
                            f"Expected an epidemic Rule, received nested list with depth > 2."
                        )
                    else:
                        raise TypeError(
                            f"Invalid type at input rules[{i}][{j}]: expected an epidemic Rule, received {subitem} with type {type(subitem).__name__}."
                        )
                normalized_list.append(item)
            else:
                raise TypeError(f"Element {subitem} at input rules[{i}] must be an epidemic Rule or a list of epidemic Rules. Received {type(item).__name__}.")

        return normalized_list
            
    def model_post_init(self, _) -> pd.DataFrame:
        if self.cur_state is None:
            self.cur_state = copy.deepcopy(self.init_state)
        if self.full_epi is None:
            self.full_epi = copy.deepcopy(self.init_state)    

    def reset(self) -> pd.DataFrame:
        '''! Resets the class state to have the initial state. 
             note that this will lose all of the information on the epidemic run so far'''
        self.cur_state = self.init_state.copy(deep=True)
        self.full_epi = self.init_state.copy(deep=True)
        return(self.cur_state, self.full_epi)

    @classmethod
    def from_yaml(cls, epi_yaml): #question: given the input is actually a dict data, maybe method name should be from_yaml_dict
        '''! Creates the class from a dictionary object presumed to be read in from a yaml object
           @param epi_yaml: a dictionary created from the epi yaml object.'''
        
        required_keys = {'init_state', 'rules'}
        missing_keys = required_keys - epi_yaml.keys()
        if missing_keys:
            raise ValueError(f"Missing required fields in yaml file: {missing_keys}.")
        
        init_state = pd.DataFrame(epi_yaml['init_state'])

        #Check stochastc, defaulting to rule based. 
        if 'stoch_policy' in epi_yaml.keys():
            stoch_policy = epi_yaml['stoch_policy']
        else:
            stoch_policy = "rule_based"

        #now instantiate rules. Important to work on a copy
        rules_dict = copy.deepcopy(epi_yaml['rules'])

        #make sure rules have nested structure
        #if not isinstance(rules[0],list):
        #    rules = [rules]
        
        #now iterate over rulesets...turning dicts into rules.
        #for ruleset in rules:
            #keys = list(ruleset.keys())
        #    for i in range(len(ruleset)):
        #        ruleset[i] = Rule.from_yaml(ruleset[i])
        print('start processing rules from yaml dict.')
        processed_rules = cls.instantiate_rules(rules_section=rules_dict) #instantiated rules are returned
        print('end process rules from yaml dict.')
        print('processed rules parameters are\n', processed_rules)
        print('the type is\n', type(processed_rules))
        return cls(init_state=init_state, rules=processed_rules, stoch_policy=stoch_policy) #keyword is required when returning a class object due to use of Pydantic
    
    @staticmethod
    def instantiate_rules(rules_section):
        """Recursively process the 'rules' section and return instantiated rules by invoking Rule's from_yaml."""
        if isinstance(rules_section, dict):
            return Rule.from_yaml(rules_section)
        elif isinstance(rules_section, list):
            return [EpiModel.instantiate_rules(item) for item in rules_section]
        else:
            raise TypeError(f"Unsupported rule format, must be dict or list, received {type(rules_section).__name__}.")
    
    def to_yaml(self, save_epi: bool = False, save_state: bool = False) -> dict:
        '''!Converts the current EpiModel's: 
             - initial state from dataframe to dictionary.
             - rules objects (i.e. a list of instantiated rules) to dictionaries. 
             - EpiModel's stoch_policy from string to dictionary. 
            Creates a dictionary object to contain above dictionaries and be saved to YAML file for this EpiModel.
            @param save_epi: the full epidemic history to be saved or not.
            @param save_state: the current state to saved or not.
        '''
        #Creates a dictionary object, setting up key-value pairs
        rc = {
            'stoch_policy': self.stoch_policy,
            'rules': list()
        }

        #Converts init_state dataframe to dictionary type
        rc['init_state'] = self.init_state.to_dict(orient='list')

        #Iterates each rule object from the existing rules (i.e. a list of lists) and converts its attributes to dictionaries
        for i in range(len(self.rules)):
            ruleset = self.rules[i]
            rc['rules'].append(list()) #appends an empty list to contain rules found in current ruleset
            for rule in ruleset:
                rc['rules'][i].append(rule.to_yaml())

        rc_converted = self.convert_to_yaml_friendly(rc)

        #question: do we want to save the full epi and current state to a csv file? should these code stay in to_yaml method?
        if save_epi:
            raise ValueError("Saving the full epi is not yet implemented")
        
        if save_state:
            raise ValueError("Saving the current state is not yet implemented")
    
        return(rc_converted)

    @staticmethod    
    def convert_to_yaml_friendly(data) -> dict:
        """
        Recursively traverses the to_yaml dictionary and converts any non-serializable types into YAML-friendly formats.
        """
        if isinstance(data, dict):
            print('it is dict data:\n', data) #debug
            return {key: EpiModel.convert_to_yaml_friendly(value) for key, value in data.items()}
        elif isinstance(data, list):
            print('it is list:\n', data) #debug
            return [EpiModel.convert_to_yaml_friendly(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            print('it is dataframe:\n', data) #debug
            if len(data) == 1: #if single row in dataframe
                return data.iloc[0].to_dict()
            else:
                return data.to_dict(orient='list')#if multiple-rows in dataframe
        elif isinstance(data, np.ndarray):
            print('it is array:\n', data) #debug
            return data.tolist()
        else:
            print('data returned:\n', data)
            return data    
    
    def do_timestep(self, dt: int | float =1.0, ret_cur_state: bool = False) -> pd.DataFrame:
        """!Does a timestep process, updating the epidemic current state by applying each epidemic rule to the current state data.
            If in cycles of simulation, appends each iteration's current state to the full epidemic history.
        @param dt: the time step.
        @param ret_cur_state: whether return the the current state at the end of iterations.
        """

        #iterates through the rulesets, gets delta out of each rule, updates the current state with deltas,
        #and record each current state to full epidemic history.
        
        print('initial current_state of each dt is\n', self.cur_state) #debug
        print('Epi model starts!!!') #debug

        for ruleset in self.rules:
            print('current ruleset is\n', ruleset) #debug
            all_deltas = pd.DataFrame()
            #Processes cur_state and obtain all_detlas within the current ruleset
            for rule in ruleset:
                print('current rule is\n', rule) #debug
                if self.stoch_policy == "rule_based":
                    print('epi model rule based') #debug
                    nw_deltas = rule.get_deltas(self.cur_state, dt = dt)
                    print('nw_delta is\n', nw_deltas) #debug
                else:
                    #print('check stochastic: ', rule.stochastic) #debug
                    nw_deltas = rule.get_deltas(self.cur_state, dt = dt, stochastic = (self.stoch_policy=="stochastic"))
                    #print('nw_delta is\n', nw_deltas) #debug
                    
                if nw_deltas is None or nw_deltas.empty: #cases of returned nw_deltas is None or empty
                    all_deltas = all_deltas
                else: 
                    all_deltas = pd.concat([all_deltas, nw_deltas]) #may not need add reset index before passing 
                print('all_deltas is\n', all_deltas) #debug
                if rule is not ruleset[-1]: #debug
                    print('---next rule---') #debug
                else: print('finished current ruleset, moving on') #debug
                
            if all_deltas.shape[0]==0: #no changes out of the processed rule
                continue
            
            #Need to make sure the T for all deltas is 0 first.
            #yl: this if-else block for T may not be needed since EpiModel checks T and N at initialization
            #if 'T' in self.cur_state.columns:
            #    pass #if column T exists in the initial cur_state dataframe, do nothing
            #else:
            #    all_deltas = all_deltas.assign(T=0.0) #add a new column T with initial value 0 to all_deltas
            #    self.full_epi = self.full_epi.assign(T=0.0) #add a new column T with initial value 0 to  full epi before concatenating cur_state
            
            
            #Prepares updated cur_state for the next ruleset
            #appends all deltas to the current state, grouping all features except N and T and aggregate N and T 
            #Need to make sure the T for all deltas has non-negative values first.
            print('before concat cur_state and all_deltas, cur_state is\n', self.cur_state)
            print('before concat cur_state and all_deltas, all_deltas is\n', all_deltas)
            nw_state = pd.concat([self.cur_state, all_deltas])#.reset_index(drop=True) #1st change, confirmed this reset_index is not needed for MultiStrainSI 
            print('after concat but before grouping nw_state is\n', nw_state) #debug

            # Get grouping columns
            agg_col = {'N','T'} #rename the variable from tbr to agg_col
            gp_cols = [item for item in nw_state.columns if item not in agg_col]
            
            print('group cols are: ', gp_cols)

            #groups all feature columns and aggregates N and T
            if gp_cols:
                nw_state = nw_state.groupby(gp_cols, dropna=False, observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index(drop=False) #reset_index is to convert groupers back to columns, drop=False #question: add dropna=False option in case combined dataset nw_state has NaN so groupby() can handle them.
                
            print("***")
            print('after grouping new state is\n', nw_state)
            
            nw_state = nw_state[nw_state["N"]!=0].reset_index(drop=True) #3rd change, reset index to have clean nw_state and cur_state
            #print('remove 0 rows, nw_state is\n', nw_state)

            self.cur_state = nw_state
            print('before adding dt, current_state is\n', self.cur_state) #debug
            if ruleset is not self.rules[-1]: #debug
                print('-------next ruleset--------') #debug
            else: print('for loop ends') #debug
    
      
        self.cur_state = self.cur_state.assign(T=max(self.cur_state['T'])+dt) #T is forward with dt after each timestep iteration
        print('final current_state is\n', self.cur_state) #debug
        
        # append the updated current state to the epidemic history.
        self.full_epi = pd.concat([self.full_epi, self.cur_state]).reset_index(drop=True)
        print('full epi is\n', self.full_epi)#debug
        print('----') #debug

        if ret_cur_state:
            return self.cur_state
            
    ##Echo TODO: decide if we want to make it possible to add rules dynamically
    ##Add new rules to the exiting rules. This allows new rules to be added to the model at any time, even after the model has been initialized. 
    def add_rule(self, new_rule) -> List[List[Rule]]:
        """! Adds a new rule or a list of rules to the model.
        @param new_rule: a single rule object or a list of rule objects to be added."""
        
        # If the new rule is a single Rule object, wrap it in a list
        if isinstance(new_rule, Rule):
            new_rule = [new_rule]

        # Add the new rule(s) as a new rule set
        if isinstance(new_rule, list): # If the new rule is a list or wrapped in a list
            # If the first element of new_rule is a list, assume it's a rule set and append it directly to the existing rules
            if isinstance(new_rule[0], list):
                self.rules.extend(new_rule)
            else:
                # Otherwise, append the new_rule list as a new rule set to the existing rules
                self.rules.append(new_rule)
        else:
            raise ValueError("new_rule must be a Rule object or a list of Rule objects")
        
        self.validate_rules_list(self.rules) #validate the update rules before processing

        return(self.rules)
