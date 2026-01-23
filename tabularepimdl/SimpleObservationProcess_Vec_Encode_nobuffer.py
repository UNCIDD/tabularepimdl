from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from tabularepimdl.Rule import Rule

#This class removes the pre-allocated buffer parameter from get_deltas() method
#This class is used to compare performance to the class that has pre-allocated buffer
class SimpleObservationProcess_Vec_Encode_nobuffer(Rule, BaseModel):
    """!
    Rule that captures a simple generic observation process where people from a particular state are
    observed to move into another state at some constant rate.
    
    Attributes:
        source_col: the column containing source_state for the observation process.
        source_state: the state individuals start, listed in source_col.
        obs_col: the column that contains each group of individuals' observed state.
        rate: the number of people move from a particular state into another state per unit time.
        unobs_state: un-observed state, listed in obs_col.
        incobs_state: incident-observed state, listed in obs_col.
        prevobs_state: previously-observed state, listed in obs_col.
        stochastic: whether the process is stochastic or deterministic.
        infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
        obs_col_all_categories: all the observation categories used in epidemics. e.g. ['U', 'P', 'I'], U=unobserved, P=previously-observed, I=incident-observed
    """

    source_col: str = Field(description = "the column containing source_state for the observation process.")
    source_state: str = Field(description = "the state individuals start.")
    obs_col: str = Field(description = "the column that contains each group of individuals' observed state.")
    rate: float = Field(ge=0, description = "the number of people move from a particular state into another state per unit time.")
    unobs_state: str = Field(default='U', description="un-observed state.")
    incobs_state: str = Field(default='I', description = "incident-observed state.")
    prevobs_state: str = Field(default='P', description = "previously-observed state.")
    stochastic: bool = Field(default=False, description = "whether the process is stochastic or deterministic.")
    infstate_compartments: list[str] = Field(description = "the infection compartments used in epidemics.")
    obs_col_all_categories: list[str] = Field(description = "all the observation categories used in epidemics.") #new attribute is needed for encoding

    _source_state_code: int | None = PrivateAttr(default=None)
    
    #_observation_all_categories_code: list[int] | None = PrivateAttr(default=None) #unused
    _unobs_code: int | None = PrivateAttr(default=None)
    _incobs_code: int | None = PrivateAttr(default=None)
    _prevobs_code: int | None = PrivateAttr(default=None)

    def model_post_init(self, _):
        infstate_to_int = {s: i for i, s in enumerate(sorted(self.infstate_compartments))}  #encode infstate strings to integers {'I': 0, 'R': 1, 'S': 2}
        self._source_state_code = infstate_to_int.get(self.source_state)
        
        
        observation_to_int = {obs: i for i, obs in enumerate(sorted(self.obs_col_all_categories))} #encode observation strings to integer
        self._unobs_code = observation_to_int.get(self.unobs_state)
        self._incobs_code = observation_to_int.get(self.incobs_state)
        self._prevobs_code = observation_to_int.get(self.prevobs_state)


    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray | None = None, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
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

        infstate_idx = col_idx_map[self.source_col]
        obs_col_idx = col_idx_map[self.obs_col]
        n_idx = col_idx_map['N']

        #variables definition
        #out_of_unobs: folks moved out unobserved (-)
        #into_incobs: folks moved in incident-observed (+)
        #out_of_incobs: folks moved out incident-observed (-)
        #into_prev: folks moved in previously-observed (+)

        current_state = current_state.astype(np.float64) #temperary dtype assignment, model engine already has it
        #out_of_unobs supports deterministic and stochastic
        #print('current_state\n', current_state, current_state.dtype)
        mask_source_state_unobs = (current_state[:, infstate_idx] == self._source_state_code) & (current_state[:, obs_col_idx] == self._unobs_code)
        #print('mask source state:', mask_source_state_unobs)
        if not np.any(mask_source_state_unobs):
            return np.empty((0, current_state.shape[1]), dtype=current_state.dtype)
        
        out_of_unobs = current_state[mask_source_state_unobs] #.astype(np.float64) or convert the dtype to floating here
        #print('out of unobs\n', out_of_unobs, out_of_unobs.dtype) #debug
        N_out_of_unobs = out_of_unobs[:, n_idx]
        #print('N out of unobs\n', N_out_of_unobs) #debug

        count_out_Unobs = len(N_out_of_unobs)
        #print('count out unbos:', count_out_Unobs) #debug
        
        rate_const = 1 - np.exp(-dt * self.rate)
        
        if stochastic:
            changed_N = -np.random.binomial(N_out_of_unobs.astype(np.int32), rate_const)
        else:
            changed_N = -N_out_of_unobs * rate_const
        #print('changed_N:', changed_N, type(changed_N)) #debug

        #out_of_unobs
        out_of_unobs_buffer = out_of_unobs.copy() #need to make a copy
        #print('out unobs buffer nidx:', out_of_unobs_buffer[:, n_idx])
        out_of_unobs_buffer[:, n_idx] = changed_N
        #print('out unobs buffer nidx --:', out_of_unobs_buffer[:, n_idx])
        #print('1. out_of_unobs buffer:\n', out_of_unobs_buffer) #debug
        
        #additions, changes in in_ and out_ incobs and prevobs only require deterministic process
        #into_incobs = N_out_of_unobs.copy()
        #into_incobs[:, obs_col_idx] = self._incobs_code
        #into_incobs[:, n_idx] *= -1
        
        #into_incobs
        #print('out unobs orig:', out_of_unobs)
        into_incobs_buffer = out_of_unobs #no need to make a copy again
        into_incobs_buffer[:, n_idx] = -changed_N
        into_incobs_buffer[:, obs_col_idx] = self._incobs_code
        #print('2. into_incobs:\n', into_incobs_buffer) #debug

        #move folks out of current_state incobs state, out_of_incobs
        mask_incobs = current_state[:, obs_col_idx] == self._incobs_code
        out_of_incobs = current_state[mask_incobs]
        #print('out_of_incobs\n', out_of_incobs) #out_of_incobs[:, n_idx] *= -1
        count_out_Incobs = len(out_of_incobs)
        #print('count_out_Incobs:', count_out_Incobs)
        out_of_incobs_buffer = out_of_incobs.copy() #need to make a copy
        out_of_incobs_buffer[:, n_idx] *= -1
        #print('3. out_of_incobs:\n', out_of_incobs_buffer) #debug

        #move folks out of the incident state and into the previous state, into_prev
        #into_prev = current_state[mask_incobs]
        #into_prev[:, obs_col_idx] = self._prevobs_code
        into_prev_buffer = out_of_incobs #no need to make a copy again
        into_prev_buffer[:, obs_col_idx] = self._prevobs_code
        #print('4. into_prev:\n', into_prev_buffer) #debug

        #print('5. result buffer:\n', np.vstack((out_of_unobs_buffer, into_incobs_buffer, out_of_incobs_buffer, into_prev_buffer)))
        return np.vstack((out_of_unobs_buffer, into_incobs_buffer, out_of_incobs_buffer, into_prev_buffer))
    
    
    def to_yaml(self) -> dict:
        """
        return the rule's attributes to a dictionary.
        """
        rc = {
            'tabularepimdl.SimpleObservationProcess_Vec_Encode_nobuffer': self.model_dump()
        }
        return rc
    

    #set up a property to return all the required compartments used in infstate column
    @property
    def infstate_all(self) -> list[str]: 
        return self.infstate_compartments
    
    #set up a property to return all the required categories used in obs_col
    @property
    def obs_col_all(self) -> list[str]: 
        return self.obs_col_all_categories