import pandas as pd
import copy
from tabularepimdl.Rule import Rule


class EpiModel:
    """! Class that that applies a list of rules to a changing current state through 
    some number of time steps to produce an epidemic. It has attributes representing the current
    state and the full epidemic thus far

    @param cur_state, a data frame (at the moment) w/ the current epidemic state
    @param full_epi, full epidemic history
    """

    init_state: pd.DataFrame
    cur_state: pd.DataFrame
    full_epi: pd.DataFrame
    rules: list
    stoch_policy: str
    
    ##TODO: Add a precision option
    ##TODO: decide if we want to make it possible to add rules dynamically

    def __init__(self, init_state, rules:list, stoch_policy = "rule_based") -> None:
        """! Initialize with a initial state and set of rules.
        
        Question: should their be a default for those as NULL
        
        @param init_state the initial state. Must have at minimum columns T and N
        @param rules the initial set of rules. Can either be a list of list of lists.
        @param stoch_policy how should stochasticity be determined. If "rule_based" we
           revert to the rule, if "deterministic" we force deterministic and if 
           "stochastic" we force stochastic"""
        #check if init_state is a DataFrame
        if not isinstance(init_state, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, but got {type(init_state).__name__} instead.")

        self.init_state = init_state #for the TBI reset function
        self.cur_state = init_state
        self.full_epi = init_state #the full epidemic is just the current state

        if isinstance(rules[0],list):
            self.rules = rules
        else:
            self.rules = [rules]
            
        self.stoch_policy=stoch_policy
        

    def reset(self):
        '''! Resets the class state to have the initial state, etc. 
             note that this will lose all of the information on the epidemic run so far'''
        self.cur_state = self.init_state.copy()
        self.full_epi = self.init_state.copy()
        return(self.cur_state, self.full_epi)

    @classmethod
    def from_yaml(cls, epi_yaml):
        '''! Creates the class from a dictionary object presumed to be read in from a yaml object
        
        @param epi_yaml a dictionary created from the epi yaml object'''
        #TODO Extend this init state so it could also be read in from a CSV file.
        init_state = pd.DataFrame(epi_yaml['init_state'])

        #Check stochastc, defaulting to rule based. 
        if 'stoch_policy' in epi_yaml:
            stoch_policy = epi_yaml['stoch_policy']
        else:
            stoch_policy = "rule_based"

        #now instantiate rules. Important to work on a copy
        rules = copy.deepcopy(epi_yaml['rules'])

        #make sure rules have nested structure
        if not isinstance(rules[0],list):
            rules = [rules]

        #now iterate over rulesets...turning dicts into rules.
        for ruleset in rules:
            #keys = list(ruleset.keys())
            for i in range(len(ruleset)):
                ruleset[i] = Rule.from_yaml(ruleset[i])

        return cls(init_state, rules, stoch_policy)
    
    def to_yaml(self, save_epi = False, save_state=False)->dict:
        '''! Creates a dictionary object appropriate to be saved to YAML for this EpiModel.
        @param save_epi should the full epidemic be saved.
        @param save_state should the current state be saved'''

        rc = {
            'stoch_policy': self.stoch_policy,
            'rules':list()
        }

        for ind in range(len(self.rules)):
            ruleset = self.rules[ind]
            rc['rules'].append(list())
            for rule in ruleset:
                rc['rules'][ind].append(rule.to_yaml())

        rc['init_state'] = self.init_state.to_dict(orient='list')


        if save_epi:
            raise ValueError("Saving the full epi is not yet implemented")
        
        if save_state:
            raise ValueError("Saving the current state is not yet implemented")
    
        return(rc)
        
        
    
    def do_timestep(self, dt=1.0, ret_nw_state= False):
        """! does a timestep, updating the current state and appending to the 
        full epidemic. 
        
        @param dt the time step
        @param ret_nw_state should we return the new state at the end."""


        # interate through the rule sets updating the current state (except for time) after
        #each set of rules to be fed into the next one.
        

        for ruleset in self.rules:
            all_deltas = pd.DataFrame()
            for rule in ruleset:
                if self.stoch_policy == "rule_based":
                    nw_deltas = rule.get_deltas(self.cur_state, dt=dt)
                else:
                    nw_deltas = rule.get_deltas(self.cur_state, dt=dt, stochastic=self.stoch_policy=="stochastic")

                all_deltas = pd.concat([all_deltas, nw_deltas])
                
            if all_deltas.shape[0]==0:
                continue
            # Now apply the deltas. If the current state is properly clean
            # (i.e., no duplicates) we should just be able to append all deltas
            # to the currents state, group by everything besides N and T and 
            # sum. Need to make sure the T for all deltas is 0 first.
            all_deltas = all_deltas.assign(T=0)

            #append all deltas
            nw_state = pd.concat([self.cur_state, all_deltas]).reset_index(drop=True)

            # Get grouping columns
            tbr = {'N','T'}
            gp_cols = [item for item in all_deltas.columns if item not in tbr]

            #print("XXX")
            
            #print(nw_state)
            #print(gp_cols)
     

            #now collapse..only if we have groups. This causes problems 
            if gp_cols:
                nw_state = nw_state.groupby(gp_cols, dropna=False, observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index() #yl, question: do we want to use dropna=False option in groupby()?
                #nw_state = nw_state.groupby(gp_cols,observed=True).sum(numeric_only=False).reset_index()       #yl, question: T column values are always assigned with 0, why do we search the max value of T?

            #print("***")
            #print(nw_state)
     
            nw_state = nw_state[nw_state['N']!=0]
  

            self.cur_state = nw_state

    
        # print("----")
        # print(self.cur_state)
        # print(max(self.cur_state['T']))
        # print(dt)
        # print("++++")
  
        self.cur_state = self.cur_state.assign(T=max(self.cur_state['T'])+dt) ##max deals with new states.

        # append the new current state to the epidemic history.
        self.full_epi = pd.concat([self.full_epi, self.cur_state]).reset_index(drop=True)
        
        if ret_nw_state:
            return self.cur_state
            
