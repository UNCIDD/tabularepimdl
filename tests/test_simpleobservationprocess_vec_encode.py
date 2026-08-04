"""
Unit test for SimpleObservationProcess_Vec_Encode.py.

Mirrors tests/test_simpleobservationprocess.py's scenario (same rate, same population counts,
same source/obs states) on an encoded NumPy array.
"""
from unittest import mock

import numpy as np
import pytest

from tabularepimdl.SimpleObservationProcess_Vec_Encode import SimpleObservationProcess_Vec_Encode

# columns: InfState (source_col), Hosp (obs_col), N
COL_IDX_MAP = {"InfState": 0, "Hosp": 1, "N": 2}

# encodings (alphabetical): InfState sorted(['S','I']) -> I=0, S=1
#                           Hosp sorted(['U','I','P']) -> I=0, P=1, U=2
S, I_ = 1, 0
OBS_I, OBS_P, OBS_U = 0, 1, 2
infstate_compartments = ["S", "I", "R"]

@pytest.fixture
def dummy_state():
    """Mirrors test_simpleobservationprocess.py's dummy_state: N=[10,20,30,40,50],
    Infection_State=['S','I','S','I','S'], Hosp=['U','U','P','U','I']."""
    return np.array([
        [S,  OBS_U, 10.0],
        [I_, OBS_U, 20.0],
        [S,  OBS_P, 30.0],
        [I_, OBS_U, 40.0],
        [S,  OBS_I, 50.0],
    ])


@pytest.fixture
def result_buffer():
    return np.empty((10, 3))


@pytest.fixture
def simple_observation():
    return SimpleObservationProcess_Vec_Encode(
        source_col="InfState", source_state="I", obs_col="Hosp", rate=0.05,
        source_col_all_categories=["S", "I"], infstate_compartments=infstate_compartments,
        obs_col_all_categories=["U", "I", "P"],
    )


def test_initialization(simple_observation):
    assert simple_observation.source_col == "InfState"
    assert simple_observation.source_state == "I"
    assert simple_observation.obs_col == "Hosp"
    assert simple_observation.rate == 0.05
    assert simple_observation.unobs_state == "U"
    assert simple_observation.incobs_state == "I"
    assert simple_observation.prevobs_state == "P"
    assert simple_observation.stochastic is False
    assert simple_observation.infstate_all == ["S", "I", "R"]
    assert simple_observation.obs_col_all == ["U", "I", "P"]


def test_get_deltas_deterministic(simple_observation, dummy_state, result_buffer):
    deltas = simple_observation.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=False)

    rate_const = 1 - np.exp(-1.0 * 0.05)
    expected = np.array([
        [I_, OBS_U, -20 * rate_const],  # out_of_unobs (subtraction)
        [I_, OBS_U, -40 * rate_const],
        [I_, OBS_I,  20 * rate_const],   # into_incobs (addition)
        [I_, OBS_I,  40 * rate_const],
        [S,  OBS_I, -50.0],             # out_of_incobs (the pre-existing Hosp='I' row, N negated)
        [S,  OBS_P,  50.0],              # into_prev
    ])

    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(simple_observation, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=10):
        deltas = simple_observation.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)

        expected = np.array([
            [I_, OBS_U, -10.0],
            [I_, OBS_U, -10.0],
            [I_, OBS_I, 10.0],
            [I_, OBS_I, 10.0],
            [S, OBS_I, -50.0],
            [S, OBS_P, 50.0],
        ])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_matching_rows_returns_empty(simple_observation, result_buffer):
    """No one is in source_state + unobs_state -> nothing to observe."""
    state_without_unobserved_source = np.array([
        [S, OBS_U, 10.0],
        [S, OBS_P, 30.0],
    ])
    deltas = simple_observation.get_deltas(current_state=state_without_unobserved_source, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 3)


def test_get_deltas_missing_n_column_raises(simple_observation, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        simple_observation.get_deltas(current_state=dummy_state, col_idx_map={"InfState": 0, "Hosp": 1}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(simple_observation):
    result = simple_observation.to_dict()
    assert result == {
        "tabularepimdl.SimpleObservationProcess_Vec_Encode": {
            "source_col": "InfState",
            "source_state": "I",
            "source_col_all_categories": ["S", "I"],
            "obs_col": "Hosp",
            "rate": 0.05,
            "unobs_state": "U",
            "incobs_state": "I",
            "prevobs_state": "P",
            "stochastic": False,
            "infstate_compartments": infstate_compartments,
            "obs_col_all_categories": ["U", "I", "P"],
        }
    }
