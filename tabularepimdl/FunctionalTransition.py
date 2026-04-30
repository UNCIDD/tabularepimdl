import pandas as pd
import numpy as np
import inspect
from tabularepimdl.Rule import Rule
from pydantic import BaseModel, field_validator, PrivateAttr
from typing import Union, Callable, Any

#define a RateType
RateType = Union[int, float, Callable[[pd.DataFrame, float, dict[str, Any]], float | pd.Series]]

class FunctionalTransition(Rule, BaseModel):
    '''
    Functional Transition class represent a transition from one state to another.
    It takes constant rate or a functional-based rate.
    It suits real-time simulations that accommodate dynamic and context-aware transitions.
    * Hybrid rate Handling: constant & function
    * Time-Awareness: time tracking
    * Context-Awareness: modeling interventions like lockdowns, vaccination rollouts
    * Fallback capabiliy: supports both 2-arg and 3-arg callables
    '''

    """! Initialization.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    @para _time: time tracker of the process.
    """
    column: str
    from_st: str
    to_st: str
    rate: RateType
    stochastic: bool = False
    _time: float = PrivateAttr(default=0.0)

    #Validate rate's datatype
    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate_type(cls, rate_value):
        """Ensure the rate is a positive number or a callable function."""
        if isinstance(rate_value, (int, float)):
            if rate_value >= 0:
                return rate_value
            else:
                raise ValueError(f"'rate' value expects to be a non-negative number, received {rate_value}.")
        elif callable(rate_value):
            sig = inspect.signature(rate_value) #callable parameters check
            params = sig.parameters

            param_list = list(params.values())
            if len(param_list) < 2:
                raise TypeError("Callable 'rate' must have at least 2 positional arguments(e.g. DataFrame type and numerice type) as input.")

            if param_list[0].annotation not in (pd.DataFrame, inspect._empty):
                raise TypeError("First parameter in user-defined function should be a pandas DataFrame.")

            if param_list[1].annotation not in (float, int, inspect._empty):
                raise TypeError("Second parameter in user-defined function should be a float or integer number.")

            if len(param_list) >= 3:
                third_arg = param_list[2]
                if third_arg.annotation not in (dict, type(None), inspect._empty):
                    raise TypeError("Third parameter in user-defined function should be a dict or None.")

            for p in param_list[3:]:
                if p.default is inspect.Parameter.empty:
                    raise TypeError(f"Extra parameter '{p.name}' in user-defined function must have a default value.")

            return rate_value
        else:
            raise ValueError(f"'rate' must be a non-negative number or a callable, received {type(rate_value)}.")
        
    
    def evaluate_rate(self, current_state: pd.DataFrame, t: int | float, context: dict | None = None) -> float | pd.Series:
        '''
        @param current_state: a dataframe (at the moment) representing the current epidemic state.
        @param t: current time in a epidemic process.
        @para context: interventions in a epidemic process.
        @return: a constant transition rate or a callable transition function.
        '''
        if callable(self.rate):
            try: #func with current state, time and context
                return self.rate(current_state, t, context or {})
            except: #func only with current state and time
                return self.rate(current_state, t)
        return self.rate
    
    
    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, context: dict | None = None, stochastic: bool = None) -> pd.DataFrame:
        '''
        @param current_state: a dataframe (at the moment) representing the current epidemic state. Must include column 'N'.
        @param dt: size of the timestep.
        @para context: interventions in a epidemic process.
        @para stochastic: whether the process is stochastic or deterministic.
        @return: a pandas DataFrame containing changes in from_st and to_st.
        '''
        if "N" not in current_state.columns:
            raise ValueError("Missing required column 'N' in current_state.")
        
        if stochastic is None:
            stochastic = self.stochastic

        #print('t begins: ', self._time) #debug
        deltas = current_state.loc[current_state[self.column]==self.from_st].copy()

        #evaluate rate: a constant or callable
        rate_value = self.evaluate_rate(current_state, t=self._time, context=context)
        prob = 1 - np.exp(-dt * rate_value)
        
        #deltas calculation
        if not stochastic:
            deltas["N"] = -deltas["N"] * prob
        else:
            deltas["N"] = -np.random.binomial(deltas["N"], prob)

        #increase t by dt
        self._time = self._time + dt
        #print('updated t is:', self._time) #debug
        
        #deltas_add =  deltas.assign(**{self.column: self.to_st, "N": -deltas["N"]}) #skip deltas_add for now...
        return self._time, rate_value, -deltas["N"].values[0]


    def __str__(self) -> str:
        return "FunctionalTransition: {} --> {} at rate {}".format(self.from_st, self.to_st, self.rate)
    

    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.FunctionalTransition': {
                'column': self.column,
                'from_st': self.from_st,
                'to_st': self.to_st,
                'rate': self.rate.__name__ if callable(self.rate) else self.rate,
                'stochastic': self.stochastic
            }
        }

        return rc

