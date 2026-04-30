from tabularepimdl.SimpleTransition import SimpleTransition
from tabularepimdl.WithFilters import WithFilters

class FilteredSimpleTransition(WithFilters, SimpleTransition):

    """A version of SimpleTransition that supports filtering."""
            
    filter_column: str | None = None #subclass attribute for filter
    filter_value: list[str] | None = None #subclass attribute for filter