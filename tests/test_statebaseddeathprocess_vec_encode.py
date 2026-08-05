"""
Unit test for StateBasedDeathProcess_Vec_Encode.py.

Note this rule's API is narrower than the pandas StateBasedDeathProcess.
These aren't the same feature, so this test is built around the NumPy version's actual (narrower)
semantics rather than trying to force equivalence with the pandas multi-column test.
"""
from unittest import mock

import numpy as np
import pytest

from tabularepimdl.StateBasedDeathProcess_Vec_Encode import StateBasedDeathProcess_Vec_Encode

COL_IDX_MAP = {"InfState": 0, "N": 1}
COLUMN_STATES = ["I1", "R1", "R2", "S1", "S2"]
# encoding is sorted(column_states): I1=0, R1=1, R2=2, S1=3, S2=4
I1, R1, R2, S1, S2 = 0, 1, 2, 3, 4
infstate_compartments = ["S", "I", "R"]

@pytest.fixture
def dummy_state():
    return np.array([
        [I1, 10.0],
        [R1, 20.0],  # target
        [R2, 30.0],
        [S1, 40.0],
        [S2, 50.0],  # target
    ])


@pytest.fixture
def result_buffer():
    return np.empty((10, 2))


@pytest.fixture
def statebased_deathprocess():
    return StateBasedDeathProcess_Vec_Encode(
        column="InfState", column_states=COLUMN_STATES, target_states=["S2", "R1"], rate=0.05,
        infstate_compartments=infstate_compartments,
    )


def test_initialization(statebased_deathprocess):
    assert statebased_deathprocess.column == "InfState"
    assert statebased_deathprocess.column_states == COLUMN_STATES
    assert statebased_deathprocess.target_states == ["S2", "R1"]
    assert statebased_deathprocess.rate == 0.05
    assert statebased_deathprocess.stochastic is False
    assert statebased_deathprocess.column_all == COLUMN_STATES
    assert statebased_deathprocess.infstate_all == ["S", "I", "R"]
    assert statebased_deathprocess.expansion_factor == max(5 * 3, 2 * 3)


def test_get_deltas_deterministic(statebased_deathprocess, dummy_state, result_buffer):
    deltas = statebased_deathprocess.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    rate_const = 1 - np.exp(-1.0 * 0.05)
    expected = np.array([
        [R1, -20 * rate_const],
        [S2, -50 * rate_const],
    ])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(statebased_deathprocess, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=15):
        deltas = statebased_deathprocess.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([
            [R1, -15.0],
            [S2, -15.0],
        ])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_matching_rows_returns_empty(statebased_deathprocess, result_buffer):
    state_without_targets = np.array([[I1, 10.0], [R2, 30.0]])
    deltas = statebased_deathprocess.get_deltas(current_state=state_without_targets, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 2)


def test_get_deltas_missing_n_column_raises(statebased_deathprocess, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        statebased_deathprocess.get_deltas(current_state=dummy_state, col_idx_map={"InfState": 0}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(statebased_deathprocess):
    result = statebased_deathprocess.to_dict()
    assert result == {
        "tabularepimdl.StateBasedDeathProcess_Vec_Encode": {
            "column": "InfState",
            "column_states": COLUMN_STATES,
            "target_states": ["S2", "R1"],
            "rate": 0.05,
            "stochastic": False,
            "infstate_compartments": infstate_compartments,
        }
    }
