## Define a class that represents a transition rule. Its simplest form is a rule just takes in the current state,
## and returns as set of deltas that will be applied to the appropriate parts of the current state. These deltas
## should specify the full row signature.
from abc import ABC, abstractmethod
import inspect
import importlib
import pandas as pd

class Rule(ABC):

    stochastic: bool
    """
    @param stochastic: whether the process is stochastic or deterministic.
    """
        
    @abstractmethod
    def get_deltas(self, current_state: pd.DataFrame, dt: int|float, stochastic: bool) -> pd.DataFrame:
        """! Method should take in current state and return a series of deltas to that state.
        @param current_state: a data frame (at the moment) w/ the current epidemic state
        @param dt: the size of the timestep
        @return: deltas to the epidemic state.
        """

        pass

    @classmethod
    def from_yaml(cls, rule_yaml):
        '''! This method loads a rule from its full yaml definition. Should not be overridden.
        @param rule_yaml: a dictionary (key-value pair) defining the class(es) read from yaml file.
        @return: an instantiated class object with parameter values initialized.
        '''
        
        key = list(rule_yaml.keys())[0]

        if '.' in key:
            mod_nm, cls_nm = key.split('.')
            #print(f"mod_nm is {mod_nm}, cls_nm is {cls_nm}".format(mod_nm, cls_nm)) #debug
            if mod_nm != 'tabularepimdl': #Ensure the correct tabularepimdl module is imported
                raise ImportError(f"Expected pacakge 'tabularepimdl' but received {mod_nm}.")
            mod = importlib.import_module(mod_nm)
            rule_cls = getattr(mod, cls_nm) #rule_cls is expected to be a class defined in tabularepimdl
        else:
            #need to transverse all parent frames until we find the key.
            rule_cls = inspect.currentframe().f_locals.get(key) #check current frame
            if rule_cls is None:
                frames = inspect.getouterframes(inspect.currentframe()) #check outer frame
                for frameinf in frames:
                    if frameinf.frame.f_locals.get(key) is not None:
                        rule_cls = frameinf.frame.f_locals[key]
                        break
        
        yaml_para_definition = rule_yaml[key] #obtain the parameter values from yaml's dictionary
        Rule._validate_definition(rule_cls, yaml_para_definition) #validate parameter types and names first

        return(rule_cls.from_yaml_def(rule_yaml[key]))#key's dict-type values are passed to class rule_cls, then the class is instantiated and returned
    
    @classmethod
    def _validate_definition(cls, tepi_rule, definition: dict) -> None:

        """Processes a epidemic rule's class definition from a YAML file
           and validate its definition keys against the same Pydantic-integrated model's fields.
        @para tepi_rule: a tabularepimdl class.
        @para definition: parameters defined from a YAML file for the above tabularepimdl class.   
        """
        #Expect the parameter defition to be a dictioary type data.
        if not isinstance(definition, dict):
            raise TypeError(f"Epidemic rule's parameters must be in dictionary type. Received {type(definition)}")
        
        rule_fields_mapping = tepi_rule.model_fields #get all field items from Pydantic-integrated model
        print('tepi rule fields mapping: ', rule_fields_mapping) #debug
        rule_required_fields = {name for name, field in rule_fields_mapping.items() if field.is_required()==True} #get required fields defined in an epidemic rule
        print('rule required fields: ', rule_required_fields) #debug, e.g. {'column', 'from_st', 'to_st', 'rate'}
        rule_all_fields = set(rule_fields_mapping.keys()) #get all field names defined in an epidemic rule
        print('rule all fields: ', rule_all_fields) #debug, e.g. {'to_st', 'column', 'from_st', 'rate', 'stochastic'}

        #Check if the YAML definitions match the epidemic rule class's __init__ fields.
        yaml_provided_fields = set(definition.keys()) #get all the field names defined in yaml file
        print('yaml provided fields: ', yaml_provided_fields) #debug

        missing_fields = rule_required_fields - yaml_provided_fields #fields that are in epidemic rule but missed in yaml definition
        #print(missing_fields) #debug
        extra_fields = yaml_provided_fields - rule_all_fields #fields that are in yaml definition but not belonged to the epidemic rule
        #print(extra_fields) #debug

        if missing_fields:
            raise ValueError(f"YAML file missed required fields for {tepi_rule.__name__}: {missing_fields}")
        if extra_fields:
            raise ValueError(f"Unexpected parameters in YAML file for {tepi_rule.__name__}: {extra_fields}")

        
    @classmethod
    def from_yaml_def(cls, definition):
        '''! This is the method to override to do any special processing of a rule class definition
        from a yaml file. 
        @param definition: a dictionary giving the parameters required by a epidemic rule.
        return: an instantiated class object with parameter values intialized.
        '''
        
        return(cls(**definition))

    @abstractmethod
    def to_yaml(self) -> dict:
        '''! This method should return dictionary object appropriate for inclusion in a yaml 
        definition of an epidemic. Should be a dictionary with the class name (in form module.classname)
        being the outer key containing information needed for the class to run "from_yaml"
        
        @return: a dictionary representation of this object appropriate to read in by from_yaml
        '''
        pass