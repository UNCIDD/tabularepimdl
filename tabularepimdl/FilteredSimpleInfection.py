from tabularepimdl.SimpleInfection import SimpleInfection
from tabularepimdl.WithFilters import WithFilters

class FilteredSimpleInfection(WithFilters, SimpleInfection):
    """A version of SimpleInfection that supports filtering."""
    #def __init__(self, beta, column, s_st, i_st, inf_to, filter_column=None, filter_value=None, stochastic=False):
    filter_column: str | None = None #subclass attribute for filter
    filter_value: list[str] | None = None #subclass attribute for filter
       