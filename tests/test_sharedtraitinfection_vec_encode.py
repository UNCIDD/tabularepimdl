"""
Unit test for SharedTraitInfection_Vec_Encode.py.

Reconstructs the exact grouped state produced by tests/test_sharedtraitinfection.py's dummy_state
fixture (same seed, same np.random.poisson draw) and reuses its independently-validated expected
N values (0.009576996819805172, 0.028730990459415517, 0.004788498409902586, 0.24572631546691115,
0.36711697319890346) as ground truth here -- a strong cross-check that the two implementations
agree.
"""
from unittest import mock

import numpy as np
import pytest

from tabularepimdl.SharedTraitInfection_Vec_Encode import SharedTraitInfection_Vec_Encode

# columns: HH_Number (trait_col, unencoded), InfState, N. Encoding is sorted(['S','I','R']) -> I=0, R=1, S=2.
COL_IDX_MAP = {"HH_Number": 0, "InfState": 1, "N": 2}
I_, R, S = 0, 1, 2
infstate_compartments = ["S", "I", "R"]


@pytest.fixture
def dummy_state():
    """The exact grouped state from test_sharedtraitinfection.py's dummy_state fixture (seed=3,
    np.random.poisson(2, 5)), re-encoded: HH0-S=2, HH1-S=6, HH2-S=1, HH3-S=1, HH3-I=7, HH3-R=1,
    HH4-S=2, HH4-I=5."""
    return np.array([
        [0, S,  2.0],
        [1, S,  6.0],
        [2, S,  1.0],
        [3, S,  1.0],
        [3, I_, 7.0],
        [3, R,  1.0],
        [4, S,  2.0],
        [4, I_, 5.0],
    ])


@pytest.fixture
def result_buffer():
    return np.empty((20, 3))


@pytest.fixture
def sharedtrait_infection():
    return SharedTraitInfection_Vec_Encode(
        in_beta=0.2 / 5, out_beta=0.002 / 5, inf_col="InfState", trait_col="HH_Number",
        trait_col_all_categories=[0, 1, 2, 3, 4], infstate_compartments=infstate_compartments,
    )


def test_initialization(sharedtrait_infection):
    assert sharedtrait_infection.in_beta == 0.2 / 5
    assert sharedtrait_infection.out_beta == 0.002 / 5
    assert sharedtrait_infection.inf_col == "InfState"
    assert sharedtrait_infection.trait_col == "HH_Number"
    assert sharedtrait_infection.s_st == "S"
    assert sharedtrait_infection.i_st == "I"
    assert sharedtrait_infection.inf_to == "I"
    assert sharedtrait_infection.stochastic is False
    assert sharedtrait_infection.infstate_all == ["S", "I", "R"]
    assert sharedtrait_infection.trait_col_all == [0, 1, 2, 3, 4]


def test_get_deltas_deterministic_matches_pandas_known_values(sharedtrait_infection, dummy_state, result_buffer):
    deltas = sharedtrait_infection.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    # cross-checked against tests/test_sharedtraitinfection.py::test_get_deltas_deterministic's exact values
    subtraction_n = np.array([
        0.009576996819805172, 0.028730990459415517, 0.004788498409902586,
        0.24572631546691115, 0.36711697319890346,
    ])
    expected = np.vstack([
        np.column_stack([[0, 1, 2, 3, 4], [S] * 5, -subtraction_n]),
        np.column_stack([[0, 1, 2, 3, 4], [I_] * 5, subtraction_n]),
    ])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(sharedtrait_infection, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=20):
        deltas = sharedtrait_infection.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.vstack([
            np.column_stack([[0, 1, 2, 3, 4], [S] * 5, [-20.0] * 5]),
            np.column_stack([[0, 1, 2, 3, 4], [I_] * 5, [20.0] * 5]),
        ])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_susceptible_returns_empty(sharedtrait_infection, result_buffer):
    all_infected = np.array([[3, I_, 7.0], [4, I_, 5.0]])
    deltas = sharedtrait_infection.get_deltas(current_state=all_infected, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 3)


def test_get_deltas_missing_n_column_raises(sharedtrait_infection, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        sharedtrait_infection.get_deltas(current_state=dummy_state, col_idx_map={"HH_Number": 0, "InfState": 1}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(sharedtrait_infection):
    result = sharedtrait_infection.to_dict()
    assert result == {
        "tabularepimdl.SharedTraitInfection_Vec_Encode": {
            "inf_col": "InfState",
            "in_beta": 0.2 / 5,
            "out_beta": 0.002 / 5,
            "trait_col": "HH_Number",
            "trait_col_all_categories": [0, 1, 2, 3, 4],
            "s_st": "S",
            "i_st": "I",
            "inf_to": "I",
            "stochastic": False,
            "infstate_compartments": infstate_compartments,
        }
    }
