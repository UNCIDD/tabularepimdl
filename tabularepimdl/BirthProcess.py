from tabularepimdl.Rule import Rule
import numpy as np
import pandas as pd


class BirthProcess(Rule):
    """!
    Represents a birth process where people are borne based
    on a birth rate based on the full poplation size."""

    def __init__(self, rate: float, start_state_sig, stochastic=False)-> None:
        """!
        Initialization.

        @param rate: birth rate at per time step (where N*rate births occur).
        @param state_start_sig: initial state configuration for new births.
        @param stochastic: whether the transition is stochastic or deterministic.
        """

        super().__init__()
        self.rate = rate

        if isinstance(start_state_sig, dict):
            self.start_state_sig = pd.DataFrame([start_state_sig]) #convert a dict to a dataframe
        elif isinstance(start_state_sig, pd.DataFrame):
            self.start_state_sig = start_state_sig.copy()
        else:
            raise ValueError ("start_state_sig must be either a dictionary or a pandas DataFrame")

        self.stochastic = stochastic

    def get_deltas(self, current_state: pd.DataFrame, dt=1.0, stochastic=None):
        """
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @return: a pandas DataFrame containing changes in population birth.
        """
        required_columns = {"N"} #check if column N presents in current_state
        if not required_columns.issubset(current_state.columns):
            raise ValueError(f"Missing required columns in current_state: {required_columns}")
        
        if stochastic is None:
            stochastic = self.stochastic
        
        #returns a delta with the start_state_signature and the right number of births.
        N = current_state["N"].sum()
        births = self.start_state_sig.copy()
        birth_value = N * (1-np.exp(-dt*self.rate))
        if not stochastic:
            births["N"] = birth_value
        else:
            births["N"] = np.random.poisson(birth_value)  #get a random single outcome value by poisson distribution
        return births
        
    def to_yaml(self):
        rc = {
            'tabularepimdl.BirthProcess' : {
                'rate': self.rate,
                'start_state_sig':self.start_state_sig,
                'stochastic': self.stochastic
            }
        }

        return rc