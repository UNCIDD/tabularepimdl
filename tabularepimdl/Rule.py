## Define a class that represents a transition rule. It its simplest form a rule just takes in the current state,
## and returns as set of deltas that will be applied to the appropriate parts of the current state. These deltas
## should specify the full row signature. 
from abc import ABC, abstractclassmethod
import inspect
import importlib

class Rule(ABC):

    stochastic: bool

    @abstractclassmethod
    def get_deltas(self, current_state, dt:float, stochastic):
        """! Method should take in current state and return a 
        series of deltas to that state

        @param current_state a data frame (at the moment) w/ the current epidemic state
        @param dt the size of the timestep

        @return deltas to the epidemic state. 
        """

        pass

    @classmethod
    def from_yaml(cls, rule_yml):
        '''! This method loads a rule from its full yaml definition. Should not be overridden.
        
        @param rule_yml a dictionary defining the class read from yaml
        
        @raturn a version of the class'''
        key = list(rule_yml.keys())[0]

        if '.' in key:
            mod_nm, cls_nm = key.split('.')
            mod = importlib.import_module(mod_nm)
            rule_cls = getattr(mod, cls_nm)
        else:
            #this ends up being a bit tricky. We need to transverse all
            #parent frames until we find the key.
            rule_cls = inspect.currentframe().f_globals.get(key)
            if rule_cls is None:
                frames = inspect.getouterframes(inspect.currentframe())
                for frameinf in frames:
                    if frameinf.frame.f_globals.get(key) is not None:
                        rule_cls = frameinf.frame.f_globals[key]
                        break

        return(rule_cls.from_yaml_def(rule_yml[key]))
    
    @classmethod
    def from_yaml_def(cls, definition):
        '''! This is the method to override to do any special processing of a rule class definition
         from a yaml file. 
         
         @param definition a dictionary giving what is needed. if parameters no need to overide'''
        
        return(cls(**definition))

    @abstractclassmethod
    def to_yaml(self):
        '''! This method should return dictionary object appropriate for inclusion in a yaml 
        definition of an epi. Should be a dictionary with the class name (in form module.classname)
        being the outer key containing information needed for the class to run "from_yaml"
        
        @return a dictionary representation of this object appropriate to read in by from_yaml'''
        pass