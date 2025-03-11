import pandas as pd
import numpy as np
from tabularepimdl.EpiModel import EpiModel
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Annotated, Optional

class EpiRunner (BaseModel):
    """Class to automate running multiple iterations of an EpiModel simulation."""
    """
    Initialization.

    @para model (EpiModel): The epidemiological model to run simulations on.
    @para time_start: the start time of model simulation.
    @para time_end: the end time of model simulation.
    @para time_step: the incremenal size of time stamp.
    @para reset_flag: whether to reset the model before running.
    @para added_data: new data points to be added to the model's current state.
    @para added_time: the time point when the new data is added.
    """
    
    # Pydantic Configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow pandas & EpiModel

    time_start: Annotated[int | float, Field(ge = 0)]
    time_end: Annotated[int | float, Field(ge = 0)]
    time_step: Annotated[int | float, Field(ge = 0)]
    reset_flag: bool = False
    added_data: Optional[pd.DataFrame] = None
    added_time: Optional[list[int | float]] = None
    model: EpiModel

    @field_validator("model") #validate model is an instance of EpiModel
    @classmethod
    def validate_model(cls, model):
        """Ensure the input is an instance of EpiModel."""
        if not isinstance(model, EpiModel):
            raise ValueError("model must be an instance of EpiModel.")
        return model
    
    @field_validator("added_data") #validate added_data is pandas DataFrame
    @classmethod
    def validate_added_data(cls, added_data):
        """Ensure added_data is a pandas DataFrame."""
        if added_data is not None and not isinstance(added_data, pd.DataFrame):
            raise ValueError("added_data must be a pandas DataFrame.")
        return added_data
    
    @field_validator("added_time") #validate added_time is None or a list of numbers that are >= 0
    @classmethod
    def validate_added_time(cls, added_time):
        if added_time is None:
            return added_time

        """Ensure all elements in added_time are >= 0."""
        if not isinstance(added_time, list) or not all(isinstance(i, (int, float)) and i >= 0 for i in added_time):
            raise ValueError("All values in added_time must be numbers (int or float) and >= 0.")
        return added_time

    def reset_model(self) -> None:
        """Resets the model if the reset_flag is True."""
        if self.reset_flag:
            self.model.reset()
        else:
            pass

    def add_new_data(self, added_time: int | float = 0) -> None:
        """Adds new data to the current state."""
        required_columns = {"T"} #added_data needs to have column T
        #print("going into add_new_data method.")
        
        if not required_columns.issubset(self.model.cur_state.columns): #raise error when adding data at t=0
            raise ValueError ("The initial epi data missed column T.")
        
        #if required_columns.issubset(self.model.cur_state.columns): print('there is T')
        #else: print('no T')
        if self.added_data is not None:
            if required_columns.issubset(self.added_data.columns) and added_time in self.added_data["T"].values: #check if the event time is present in added_data column T
                self.model.cur_state = pd.concat([ self.model.cur_state, self.added_data.loc[self.added_data["T"]==added_time] ], ignore_index=True)
                #print('add new data, cur_state is\n', self.model.cur_state)
            else: raise ValueError (f"The added data does not include record for time {added_time}. Please check the added data.")
        else:
            raise ValueError (f"The added data is None. Please check the added data.")

    def run(self) -> pd.DataFrame:
        """
        Runs the simulation over the specified time range.
        Returns: cur_state at the end of the simulation.
        """
        for t in np.arange(self.time_start, self.time_end, self.time_step):
            if self.added_time is not None and t in self.added_time:
                self.add_new_data(added_time=t) # Inject data if needed at this timestep
            
            self.model.do_timestep(dt=self.time_step)  # Run one timestep

        return self.model.cur_state #only return current state of the Epi model