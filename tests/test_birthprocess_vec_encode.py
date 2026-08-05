"""
Unit test for BirthProcess_Vec_Encode.py. Pytest package is used.
These tests cover both
the birth-rate math (mirroring the pandas BirthProcess tests) and that derive-then-cache behavior,
which has no equivalent in the pandas version.
"""
from unittest import mock

import numpy as np
import pytest

from tabularepimdl.BirthProcess_Vec_Encode import BirthProcess_Vec_Encode

COL_IDX_MAP = {"AgeCat": 0, "InfState": 1, "N": 2, "T": 3}


@pytest.fixture
def current_state():
    """Two age groups with susceptible status and different population size at time 0. AgeCat=0 is the youngest group."""
    return np.array([
        [1.0, 2.0, 500.0, 0.0],
        [0.0, 2.0, 300.0, 0.0],
    ])


@pytest.fixture
def result_buffer():
    return np.empty((4, 4))


@pytest.fixture
def birthprocess_non_stochastic():
    return BirthProcess_Vec_Encode(rate=0.1, column_to_sort="AgeCat", stochastic=False, infstate_compartments=["S", "I", "R"])


@pytest.fixture
def birthprocess_stochastic():
    return BirthProcess_Vec_Encode(rate=0.1, column_to_sort="AgeCat", stochastic=True, infstate_compartments=["S", "I", "R"])


def test_initialization(birthprocess_non_stochastic):
    assert birthprocess_non_stochastic.rate == 0.1
    assert birthprocess_non_stochastic.column_to_sort == "AgeCat"
    assert birthprocess_non_stochastic.stochastic is False
    assert birthprocess_non_stochastic.infstate_all == ["S", "I", "R"]
    assert birthprocess_non_stochastic.expansion_factor == 3


def test_start_state_sig_raises_ValueError_before_first_get_deltas_call(birthprocess_non_stochastic):
    """start_state_sig is only derived from data once get_deltas has run at least once."""
    with pytest.raises(ValueError):
        _ = birthprocess_non_stochastic.start_state_sig


def test_get_deltas_deterministic_derives_signature_from_youngest_row(birthprocess_non_stochastic, current_state, result_buffer):
    deltas = birthprocess_non_stochastic.get_deltas(current_state=current_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    total_N = current_state[:, COL_IDX_MAP["N"]].sum()
    expected_birth_N = total_N * (1 - np.exp(-1.0 * 0.1))

    assert deltas.shape == (1, 4)
    assert deltas[0, COL_IDX_MAP["AgeCat"]] == 0.0  # from the AgeCat=0 (youngest group) row
    assert deltas[0, COL_IDX_MAP["InfState"]] == 2.0
    assert deltas[0, COL_IDX_MAP["N"]] == pytest.approx(expected_birth_N)
    assert deltas[0, COL_IDX_MAP["T"]] == 0.0


def test_get_deltas_stochastic_uses_poisson_draw(birthprocess_stochastic, current_state, result_buffer):
    with mock.patch("numpy.random.poisson", return_value=7):
        deltas = birthprocess_stochastic.get_deltas(current_state=current_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
        assert deltas[0, COL_IDX_MAP["N"]] == 7.0


def test_start_state_signature_is_cached_across_calls(birthprocess_non_stochastic, current_state, result_buffer):
    """Once derived, the signature (all non-N/T values) must be reused even if later current_state differs,
    only N is recomputed each call."""
    # first call of get_deltas()
    birthprocess_non_stochastic.get_deltas(current_state=current_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    saved_signature = birthprocess_non_stochastic.start_state_sig.copy()

    # a state with the same AgeCat and differernt InfState and population N
    different_state = np.array([
        [1.0, 0.0, 50.0, 0.0],  
        [0.0, 0.0, 20.0, 0.0], # a different youngest-row InfState/N than the first call
    ])
    # second call of get_deltas()
    deltas_2 = birthprocess_non_stochastic.get_deltas(current_state=different_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    
    # AgeCat/InfState columns of the signature must be unchanged from the first call
    assert deltas_2[0, COL_IDX_MAP["AgeCat"]] == saved_signature[0, COL_IDX_MAP["AgeCat"]]
    assert deltas_2[0, COL_IDX_MAP["InfState"]] == saved_signature[0, COL_IDX_MAP["InfState"]]

    # N must be recomputed from the second call's data, not reused from the first.
    total_N_2 = different_state[:, COL_IDX_MAP["N"]].sum()
    expected_birth_N_2 = total_N_2 * (1 - np.exp(-1.0 * 0.1))
    assert deltas_2[0, COL_IDX_MAP["N"]] == pytest.approx(expected_birth_N_2)


def test_get_deltas_empty_current_state_returns_empty_array(birthprocess_non_stochastic, result_buffer):
    empty_state = np.empty((0, 4))
    deltas = birthprocess_non_stochastic.get_deltas(current_state=empty_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 4)


def test_get_deltas_missing_n_column_raises(birthprocess_non_stochastic, current_state, result_buffer):
    with pytest.raises(ValueError):
        birthprocess_non_stochastic.get_deltas(current_state=current_state, col_idx_map={"AgeCat": 0, "InfState": 1, "T": 3}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(birthprocess_non_stochastic):
    result = birthprocess_non_stochastic.to_dict()
    assert result == {
        "tabularepimdl.BirthProcess_Vec_Encode": {
            "rate": 0.1,
            "column_to_sort": "AgeCat",
            "stochastic": False,
            "infstate_compartments": ["S", "I", "R"],
        }
    }
