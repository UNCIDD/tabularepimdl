"""
Unit test for SimpleInfection_Vec_Encode.py.

Mirrors tests/test_simpleinfection.py's scenario (N counts, beta, freq_dep) on an encoded NumPy
array.
"""

from unittest import mock

import numpy as np
import pytest

from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode

# columns: InfState, N. Encoding is sorted(['S','I']) -> I=0, S=2.
COL_IDX_MAP = {"InfState": 0, "N": 1}
S, I_ = 2, 0
infstate_compartments = ["S", "I", "R"]


@pytest.fixture
def dummy_state():
    """Mirrors test_simpleinfection.py's dummy_state: N=[10,20,30,40], InfState=['S','I','S','I']."""
    return np.array([[S, 10.0], [I_, 20.0], [S, 30.0], [I_, 40.0]])


@pytest.fixture
def result_buffer():
    return np.empty((8, 2))


@pytest.fixture
def simple_infection():
    return SimpleInfection_Vec_Encode(beta=0.3, column="InfState", infstate_compartments=infstate_compartments, column_categories=infstate_compartments)


def test_initialization(simple_infection):
    assert simple_infection.beta == 0.3
    assert simple_infection.column == "InfState"
    assert simple_infection.s_st == "S"
    assert simple_infection.i_st == "I"
    assert simple_infection.inf_to == "I"
    assert simple_infection.freq_dep is True
    assert simple_infection.stochastic is False
    assert simple_infection.infstate_all == ["S", "I", "R"]
    assert simple_infection.expansion_factor == 3


def test_get_deltas_deterministic_freq_dep(simple_infection, dummy_state, result_buffer):
    deltas = simple_infection.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    beta_effective = 0.3 / 100  # total N = 100, freq_dep=True
    infectious_sum = 60  # N where InfState==I
    rate_const = 1 - np.power(np.exp(-1.0 * beta_effective), infectious_sum)

    expected = np.array([[S, -10 * rate_const], [S, -30 * rate_const], [I_, 10 * rate_const], [I_, 30 * rate_const]])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_deterministic_not_freq_dep(dummy_state, result_buffer):
    rule = SimpleInfection_Vec_Encode(beta=0.3, column="InfState", freq_dep=False, infstate_compartments=infstate_compartments, column_categories=infstate_compartments)
    deltas = rule.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    infectious_sum = 60
    rate_const = 1 - np.power(np.exp(-1.0 * 0.3), infectious_sum)  # beta used directly, not divided by total N
    expected = np.array([[S, -10 * rate_const], [S, -30 * rate_const], [I_, 10 * rate_const], [I_, 30 * rate_const]])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(simple_infection, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=10):
        deltas = simple_infection.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([[S, -10.0], [S, -10.0], [I_, 10.0], [I_, 10.0]])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_infectious_returns_empty(simple_infection, result_buffer):
    all_susceptible = np.array([[S, 10.0], [S, 30.0]])
    deltas = simple_infection.get_deltas(current_state=all_susceptible, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 2)


def test_get_deltas_no_susceptible_returns_empty(simple_infection, result_buffer):
    all_infectious = np.array([[I_, 20.0], [I_, 40.0]])
    deltas = simple_infection.get_deltas(current_state=all_infectious, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 2)


def test_get_deltas_missing_n_column_raises(simple_infection, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        simple_infection.get_deltas(current_state=dummy_state, col_idx_map={"InfState": 0}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(simple_infection):
    result = simple_infection.to_dict()
    assert result == {
        "tabularepimdl.SimpleInfection_Vec_Encode": {
            "beta": 0.3,
            "column": "InfState",
            "s_st": "S",
            "i_st": "I",
            "inf_to": "I",
            "freq_dep": True,
            "stochastic": False,
            "column_categories": infstate_compartments,
            "infstate_compartments": infstate_compartments,
        }
    }
