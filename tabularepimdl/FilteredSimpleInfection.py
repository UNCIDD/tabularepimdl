from tabularepimdl.SimpleInfection import SimpleInfection
from tabularepimdl.WithFilters import WithFilters

class FilteredSimpleInfection(WithFilters, SimpleInfection):
    """A version of SimpleTransition that supports filtering."""
    def __init__(self, beta, column, s_st, i_st, inf_to, filter_column=None, filter_value=None, stochastic=False):
        #initialize the filter and the rule that applies the filter
        super().__init__(filter_column=filter_column, filter_value=filter_value, beta=beta, column=column, s_st=s_st, i_st=i_st, inf_to=inf_to, stochastic=stochastic)
       