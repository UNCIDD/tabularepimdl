from tabularepimdl.SimpleTransition import SimpleTransition
from tabularepimdl.WithFilters import WithFilters

class FilteredSimpleTransition(WithFilters, SimpleTransition):
    """A version of SimpleTransition that supports filtering."""
    def __init__(self, column, from_st, to_st, rate, filter_column=None, filter_value=None, stochastic=False):
        #initialize the filter and the rule that applies the filter
        super().__init__(filter_column=filter_column, filter_value=filter_value, column=column, from_st=from_st, to_st=to_st, rate=rate, stochastic=stochastic)
       