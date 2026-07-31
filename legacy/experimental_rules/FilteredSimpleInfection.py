from tabularepimdl.SimpleInfection import SimpleInfection
from tabularepimdl.WithFilters import WithFilters

class FilteredSimpleInfection(WithFilters, SimpleInfection):
    """A version of SimpleInfection that supports filtering."""
    
    filter_column: str | None = None #subclass attribute for filter
    filter_value: list[str] | None = None #subclass attribute for filter
       