from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Union
import pandas as pd

from tabularepimdl.SimpleTransition import SimpleTransition
from tabularepimdl.SimpleTransition_Vec import SimpleTransition_Vec


class SimpleTransitionDispatcher(BaseModel):
    """
    Dispatches Pandas and Numpy versions of SimpleTransition rule at backend.
    @param structure: data structure used for rules.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    """
    structure: Literal["Pandas", "Numpy"]
    column: str
    from_st: str
    to_st: str
    rate: float = Field(ge=0)
    stochastic: bool = False

    #Dispatcher
    _dispatcher: Union[SimpleTransition, SimpleTransition_Vec] = PrivateAttr(default=None)

    def model_post_init(self, _): #initialize dispatcher based on data structures
        if self.structure == 'Pandas':
            self._dispatcher = SimpleTransition(
                column=self.column,
                from_st=self.from_st,
                to_st=self.to_st,
                rate=self.rate,
                stochastic=self.stochastic
            )
        elif self.structure == 'Numpy':
            self._dispatcher = SimpleTransition_Vec(
                column=self.column,
                from_st=self.from_st,
                to_st=self.to_st,
                rate=self.rate,
                stochastic=self.stochastic
            )
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def get_deltas(self, current_state: pd.DataFrame, dt: int | float = 1.0) -> pd.DataFrame:
        return self._dispatcher.get_deltas(current_state, dt=dt)
