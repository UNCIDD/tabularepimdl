import pandas as pd

class WithFilters():
    """Mixin class to add filtering capability to other rules."""
    
    #def __init__(self, filter_column:str = None, filter_value: list[str] = None, *args, **kwargs) -> None: #allows the filter to accept extra arguments for other rules
    """
    Initialization.
    @para filter_column: name of the column in dataframe this filter applies to.
    @para filter_value: values of the filter_column.
    @para *args: other positional arguments this filter may take for other rules.
    @para **kwargs: other keyword arguments this filter may take for other rules.
    """
    print('start init of withFilters')
        #self.filter_column = filter_column
        #self.filter_value = filter_value
        #print('next go to SimpleTransition to init')
        #super().__init__(*args, **kwargs)
    #filter_column: str | None = None #can remove it as long as it is used with other pydantic class
    #filter_value: list[str] | None = None #can remove it as long as it is used with other pydantic class
        
    def filter_data(self, current_state: pd.DataFrame) -> pd.DataFrame:
        """
        @para current_state: a dataframe (at the moment) representing the current epidemic state.
        @return: filtered dataframe based on filter values.
        """
        #print('withfilters filtering data.')
        if self.filter_column is not None and self.filter_value is not None:
            return current_state.loc[current_state[self.filter_column].isin(self.filter_value)]
        return current_state

    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0, stochastic: bool = None) -> pd.DataFrame:
        """Applies the filter before executing the rule logic.
        @para current_state: a dataframe (at the moment) representing the current epidemic state.
        @return: a pandas DataFrame containing changes after executing the rule logic.
        """
        #print('filter_get_delta')
        filtered_state = self.filter_data(current_state)
        #print('filter_get_delta returns')
        return super().get_deltas(filtered_state, dt=dt, stochastic=stochastic)
    
    def __str__(self) -> str:
        return "WithFilter rule."