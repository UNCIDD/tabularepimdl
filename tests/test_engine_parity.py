"""
Parity tests between the pandas reference engine (EpiModel) and the production NumPy engine
(EpiModel_Vec_Encode_1_5).

These tests run an identical rule set through both engines for the same initial state and assert
the resulting population counts match, deterministically, over several timesteps. Comparison is
restricted to stochastic=False.

The pandas engine/rules live in legacy/pandas_reference/ (not part of the installable package) 
and are imported here only as a trusted baseline to validate the NumPy engine
that actually ships.
"""
import pandas as pd
import pytest
from legacy.pandas_reference.EpiModel import EpiModel
from legacy.pandas_reference.SimpleInfection import SimpleInfection
from legacy.pandas_reference.SimpleTransition import SimpleTransition

from tabularepimdl.EpiModel_Vec_Encode1_5 import EpiModel_Vec_Encode_1_5
from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode

infstate_compartments = ["S", "I", "R"]


def _sir_init_state() -> pd.DataFrame:
    return pd.DataFrame({
        "InfState": ["S", "I"],
        "N": [990.0, 10.0],
        "T": [0, 0],
    })


def _build_pandas_sir_model() -> EpiModel:
    rules = [
        [SimpleInfection(beta=0.5, column="InfState", s_st="S", i_st="I", inf_to="I", freq_dep=True, stochastic=False)],
        [SimpleTransition(column="InfState", from_st="I", to_st="R", rate=0.2, stochastic=False)],
    ]
    return EpiModel(init_state=_sir_init_state(), rules=rules)


def _build_vec_sir_model() -> EpiModel_Vec_Encode_1_5:
    rules = [
        [SimpleInfection_Vec_Encode(
            beta=0.5, column="InfState", s_st="S", i_st="I", inf_to="I", freq_dep=True, stochastic=False,
            infstate_compartments=infstate_compartments, column_categories=infstate_compartments,
        )],
        [SimpleTransition_Vec_Encode(
            column="InfState", from_st="I", to_st="R", rate=0.2, stochastic=False,
            infstate_compartments=infstate_compartments, column_categories=infstate_compartments,
        )],
    ]
    return EpiModel_Vec_Encode_1_5(init_state=_sir_init_state(), rules=rules)


def _comparable_state(state: pd.DataFrame) -> pd.DataFrame:
    """Normalize a cur_state/current_state() DataFrame to a common, order-independent shape for comparison."""
    return (
        state[["InfState", "N", "T"]]
        .astype({"N": "float64", "T": "float64"})
        .sort_values("InfState")
        .reset_index(drop=True)
    )


def test_pandas_and_numpy_engines_agree_at_initial_state():
    """Sanity check: before looping any timestep, both engines should already report the same initial state.
    This is to confirm init_state validation/grouping/column-shuffle behave consistently between engines"""
    pandas_model = _build_pandas_sir_model()
    vec_model = _build_vec_sir_model()

    pandas_state = _comparable_state(pandas_model.cur_state)
    vec_state = _comparable_state(vec_model.current_state())

    pd.testing.assert_frame_equal(pandas_state, vec_state, check_exact=False, atol=1e-6, rtol=1e-6)

@pytest.mark.parametrize("n_steps", [1, 5, 20])
def test_pandas_and_numpy_engines_agree_on_sir_trajectory(n_steps):
    """Both engines should compute the same S/I/R population counts at every step of a deterministic SIR run."""
    pandas_model = _build_pandas_sir_model()
    vec_model = _build_vec_sir_model()

    for _ in range(n_steps):
        pandas_model.do_timestep(dt=1.0)
        vec_model.do_timestep(dt=1.0)

    pandas_state = _comparable_state(pandas_model.cur_state)
    vec_state = _comparable_state(vec_model.current_state())

    pd.testing.assert_frame_equal(pandas_state, vec_state, check_exact=False, atol=1e-6, rtol=1e-6)

