"""
Unit test for EnvironmentalTransmission_Vec_Encode.py.
"""

from unittest import mock

import numpy as np
import pytest

from tabularepimdl.EnvironmentalTransmission_Vec_Encode import EnvironmentalTransmission_Vec_Encode

# columns: InfState, N. Encoding is sorted(['S','I']) -> I=0, S=2.
COL_IDX_MAP = {"InfState": 0, "N": 1}
S, I_ = 2, 0
infstate_compartments = ["S", "I", "R"]


@pytest.fixture
def dummy_state():
    return np.array([[S, 10.0], [I_, 20.0], [S, 30.0], [I_, 40.0]])


@pytest.fixture
def result_buffer():
    return np.empty((8, 2))


@pytest.fixture
def environmental_transmission():
    return EnvironmentalTransmission_Vec_Encode(beta=0.3, inf_col="InfState", infstate_compartments=infstate_compartments, inf_col_categories=infstate_compartments)


def test_initialization(environmental_transmission):
    assert environmental_transmission.beta == 0.3
    assert environmental_transmission.inf_col == "InfState"
    assert environmental_transmission.s_st == "S"
    assert environmental_transmission.inf_to == "I"
    assert environmental_transmission.stochastic is False
    assert environmental_transmission.infstate_all == ["S", "I", "R"]
    assert environmental_transmission.expansion_factor == 3


def test_get_deltas_deterministic(environmental_transmission, dummy_state, result_buffer):
    deltas = environmental_transmission.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    rate_const = 1 - np.exp(-1.0 * 0.3)
    expected = np.array([[S, -10 * rate_const], [S, -30 * rate_const], [I_, 10 * rate_const], [I_, 30 * rate_const]])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(environmental_transmission, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=10):
        deltas = environmental_transmission.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([[S, -10.0], [S, -10.0], [I_, 10.0], [I_, 10.0]])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_susceptible_returns_empty(environmental_transmission, result_buffer):
    all_infectious = np.array([[I_, 20.0], [I_, 40.0]])
    deltas = environmental_transmission.get_deltas(current_state=all_infectious, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 2)


def test_get_deltas_missing_n_column_raises(environmental_transmission, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        environmental_transmission.get_deltas(current_state=dummy_state, col_idx_map={"InfState": 0}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(environmental_transmission):
    result = environmental_transmission.to_dict()
    assert result == {
        "tabularepimdl.EnvironmentalTransmission_Vec_Encode": {
            "beta": 0.3,
            "inf_col": "InfState",
            "s_st": "S",
            "inf_to": "I",
            "stochastic": False,
            "inf_col_categories": infstate_compartments,
            "infstate_compartments": infstate_compartments,
        }
    }
