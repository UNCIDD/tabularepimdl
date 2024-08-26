## Define a class that represents a transition rule. It its simplest form a rule just takes in the current state,
## and returns as set of deltas that will be applied to the appropriate parts of the current state. These deltas
## should specify the full row signature. 
from abc import ABC, abstractmethod  #the class "abstractclassmethod" is deprecated, use classmethod with abstractmethod()
import inspect
import importlib

class Rule(ABC):

    stochastic: bool
    """
    @param stochastic, is a rule stochastic process
    """
        
    @abstractmethod
    def get_deltas(self, current_state, dt:float, stochastic):
        """! Method should take in current state and return a 
        series of deltas to that state

        @param current_state a data frame (at the moment) w/ the current epidemic state
        @param dt the size of the timestep

        @return deltas to the epidemic state. 
        """

        pass

    @classmethod
    def from_yaml(cls, rule_yaml):
        '''! This method loads a rule from its full yaml definition. Should not be overridden.
        
        @param rule_yaml a dictionary defining the class(es) read from yaml file
        
        @raturn a version of the class'''
        key = list(rule_yaml.keys())[0]

        if '.' in key:
            mod_nm, cls_nm = key.split('.')
            mod = importlib.import_module(mod_nm)
            rule_cls = getattr(mod, cls_nm)
        else:
            #this ends up being a bit tricky. We need to transverse all
            #parent frames until we find the key.
            rule_cls = inspect.currentframe().f_locals.get(key) #change f_globals to f_locals
            if rule_cls is None:
                frames = inspect.getouterframes(inspect.currentframe())
                for frameinf in frames:
                    if frameinf.frame.f_locals.get(key) is not None: #change f_globals to f_locals
                        rule_cls = frameinf.frame.f_locals[key] #change f_globals to f_locals
                        break

        return(rule_cls.from_yaml_def(rule_yaml[key]))#key's values are included in class rule_cls and this class is instantiated and returned
    
    @classmethod
    def from_yaml_def(cls, definition):
        '''! This is the method to override to do any special processing of a rule class definition
         from a yaml file. 
         
         @param definition a dictionary giving what is needed. if parameters no need to overide'''
        
        return(cls(**definition))

    @abstractmethod
    def to_yaml(self):
        '''! This method should return dictionary object appropriate for inclusion in a yaml 
        definition of an epidemic. Should be a dictionary with the class name (in form module.classname)
        being the outer key containing information needed for the class to run "from_yaml"
        
        @return a dictionary representation of this object appropriate to read in by from_yaml'''
        pass