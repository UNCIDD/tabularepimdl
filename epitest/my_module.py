from tabularepimdl.Rule import Rule

class MyRule(Rule):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def get_deltas(self, current_state, dt: float, stochastic):
        # Simple implementation that just returns a dummy delta
        return {'state' : current_state + self.param1 * dt}  # Simplified example

    def to_yaml(self):
        # Return the parameters needed to recreate the input rule
        return {
            'my_module.MyRule': {
                'param1': self.param1,
                'param2': self.param2
            }
        }

    @classmethod
    def from_yaml_def(cls, definition):
        """Processes a rule class definition from a YAML file."""
        if not isinstance(definition, dict):
            raise TypeError("Definition must be a dictionary.")
        
        expected_keys = {'param1', 'param2'}
        if not expected_keys.issubset(definition.keys()):
            raise ValueError(f"Definition must contain keys: {expected_keys}")
        
        return cls(**definition) #return the value of dictionary