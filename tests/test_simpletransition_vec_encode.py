"""
Unit test for SimpleTransition_Vec_Encode.py.

Mirrors tests/test_simpletransition.py's scenario (N counts, rate) on an encoded NumPy array.
"""

from unittest import mock

import numpy as np
import pytest

from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode

# columns: InfState, N. Encoding is sorted(['S','I']) -> I=0, S=2.
COL_IDX_MAP = {"InfState": 0, "N": 1}
S, I_ = 2, 0
infstate_compartments = ["S", "I", "R"]


@pytest.fixture
def dummy_state():
    """Mirrors test_simpletransition.py's dummy_state: N=[10,20,30,40], InfState=['S','I','S','I']."""
    return np.array([[S, 10.0], [I_, 20.0], [S, 30.0], [I_, 40.0]])


@pytest.fixture
def result_buffer():
    return np.empty((8, 2))


@pytest.fixture
def simple_transition():
    return SimpleTransition_Vec_Encode(column="InfState", from_st="S", to_st="I", rate=0.3, infstate_compartments=infstate_compartments, column_categories=infstate_compartments)


def test_initialization(simple_transition):
    assert simple_transition.column == "InfState"
    assert simple_transition.from_st == "S"
    assert simple_transition.to_st == "I"
    assert simple_transition.rate == 0.3
    assert simple_transition.stochastic is False
    assert simple_transition.infstate_all == ["S", "I", "R"]
    assert simple_transition.expansion_factor == 3


def test_get_deltas_deterministic(simple_transition, dummy_state, result_buffer):
    deltas = simple_transition.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    rate_const = 1 - np.exp(-1.0 * 0.3)
    expected = np.array([[S, -10 * rate_const], [S, -30 * rate_const], [I_, 10 * rate_const], [I_, 30 * rate_const]])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(simple_transition, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=10):
        deltas = simple_transition.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([[S, -10.0], [S, -10.0], [I_, 10.0], [I_, 10.0]])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_matching_from_state_returns_empty(simple_transition, result_buffer):
    all_infectious = np.array([[I_, 20.0], [I_, 40.0]])
    deltas = simple_transition.get_deltas(current_state=all_infectious, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 2)


def test_get_deltas_missing_n_column_raises(simple_transition, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        simple_transition.get_deltas(current_state=dummy_state, col_idx_map={"InfState": 0}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(simple_transition):
    result = simple_transition.to_dict()
    assert result == {
        "tabularepimdl.SimpleTransition_Vec_Encode": {
            "column": "InfState",
            "from_st": "S",
            "to_st": "I",
            "rate": 0.3,
            "stochastic": False,
            "column_categories": infstate_compartments,
            "infstate_compartments": infstate_compartments,
        }
    }
