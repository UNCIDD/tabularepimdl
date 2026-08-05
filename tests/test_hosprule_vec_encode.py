"""
Unit test for HospRule_Vec_Encode.py.

Uses a 2-strain scenario so both of those behaviors -- "recovered in any column reduces the rate"
and "hospitalized in the strain-matching hosp column" -- are actually exercised, not just the
degenerate single-strain case.
"""
from unittest import mock

import numpy as np
import pytest

from tabularepimdl.HospRule_Vec_Encode import HospRule_Vec_Encode

# columns: Strain1, Strain2, Hosp1, Hosp2, N
COL_IDX_MAP = {"Strain1": 0, "Strain2": 1, "Hosp1": 2, "Hosp2": 3, "N": 4}
# encoding: strain sorted(['S','I','R']) -> I=0, R=1, S=2 ; hosp sorted(['U','H']) -> H=0, U=1
I_, R, S = 0, 1, 2
H, U = 0, 1
infstate_compartments = ["S", "I", "R"]

@pytest.fixture
def dummy_state():
    """Row0: infected via Strain1 only (no recovery anywhere) -> prim_hrate applies.
    Row1: infected via Strain2, but recovered via Strain1 -> sec_hrate applies.
    Row2: fully susceptible -> not eligible for hospitalization at all."""
    return np.array([
        [I_, S,  U, U, 100.0],
        [R,  I_, U, U, 50.0],
        [S,  S,  U, U, 30.0],
    ])


@pytest.fixture
def result_buffer():
    return np.empty((10, 5))


@pytest.fixture
def hosp_rule():
    return HospRule_Vec_Encode(
        strain_cols=["Strain1", "Strain2"], hosp_cols=["Hosp1", "Hosp2"],
        strain_cols_all_categories=["S", "I", "R"], hosp_cols_all_categories=["U", "H"],
        prim_hrate=0.2, sec_hrate=0.05, infstate_compartments=infstate_compartments,
    )


def test_initialization(hosp_rule):
    assert hosp_rule.strain_cols == ["Strain1", "Strain2"]
    assert hosp_rule.hosp_cols == ["Hosp1", "Hosp2"]
    assert hosp_rule.strain_cols_all_categories == ["S", "I", "R"]
    assert hosp_rule.hosp_cols_all_categories == ["U", "H"]
    assert hosp_rule.infect_status == "I"
    assert hosp_rule.recover_status == "R"
    assert hosp_rule.hosp_status == "H"
    assert hosp_rule.prim_hrate == 0.2
    assert hosp_rule.sec_hrate == 0.05
    assert hosp_rule.stochastic is False
    assert hosp_rule.strain_cols_all == ["S", "I", "R"]
    assert hosp_rule.hosp_cols_all == ["U", "H"]


def test_mismatched_strain_and_hosp_cols_length_raises():
    with pytest.raises(ValueError):
        HospRule_Vec_Encode(
            strain_cols=["Strain1", "Strain2"], hosp_cols=["Hosp1"],
            strain_cols_all_categories=["S", "I", "R"], hosp_cols_all_categories=["U", "H"],
            prim_hrate=0.2, sec_hrate=0.05, infstate_compartments=infstate_compartments,
        )


def test_expansion_factor(hosp_rule):
    """get_deltas always emits exactly 2 rows (one subtraction, one addition) per matching input
    row -- previously this raised TypeError (compared an int to a list)."""
    assert hosp_rule.expansion_factor == 3*3


def test_encode_categorical_states_by_engine(hosp_rule):
    """Exercises the engine-driven encoding path (previously indexed a dict with a list, and used
    the wrong attribute for the hospitalization status lookup)."""
    data_domains = {
        "Strain1": {"S": 2, "I": 0, "R": 1},
        "Strain2": {"S": 2, "I": 0, "R": 1},
        "Hosp1": {"U": 1, "H": 0},
        "Hosp2": {"U": 1, "H": 0},
    }
    hosp_rule._encode_categorical_states(data_domains)
    assert hosp_rule._infect_status_code == 0
    assert hosp_rule._recover_status_code == 1
    assert hosp_rule._hosp_status_code == 0
    assert hosp_rule._state_encoding_by_engine is True


def test_get_deltas_deterministic(hosp_rule, dummy_state, result_buffer):
    deltas = hosp_rule.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    rate_prim = 1 - np.exp(-1.0 * 0.2)
    rate_sec = 1 - np.exp(-1.0 * 0.05)
    expected = np.array([
        [I_, S, U, U, -100 * rate_prim],   # row0 leaves general pool at prim_hrate
        [R, I_, U, U, -50 * rate_sec],     # row1 leaves general pool at sec_hrate (recovered elsewhere)
        [I_, S, H, U, 100 * rate_prim],    # row0 -> hospitalized via Hosp1 (matches Strain1, the infected column)
        [R, I_, U, H, 50 * rate_sec],      # row1 -> hospitalized via Hosp2 (matches Strain2, the infected column)
    ])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(hosp_rule, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=15):
        deltas = hosp_rule.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([
            [I_, S, U, U, -15.0],
            [R, I_, U, U, -15.0],
            [I_, S, H, U, 15.0],
            [R, I_, U, H, 15.0],
        ])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_infection_returns_empty(hosp_rule, result_buffer):
    all_susceptible = np.array([[S, S, U, U, 100.0], [S, S, U, U, 50.0]])
    deltas = hosp_rule.get_deltas(current_state=all_susceptible, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 5)


def test_get_deltas_missing_n_column_raises(hosp_rule, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        hosp_rule.get_deltas(current_state=dummy_state, col_idx_map={"Strain1": 0, "Strain2": 1, "Hosp1": 2, "Hosp2": 3}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(hosp_rule):
    result = hosp_rule.to_dict()
    assert result == {
        "tabularepimdl.HospRule_Vec_Encode": {
            "strain_cols": ["Strain1", "Strain2"],
            "hosp_cols": ["Hosp1", "Hosp2"],
            "strain_cols_all_categories": ["S", "I", "R"],
            "hosp_cols_all_categories": ["U", "H"],
            "infect_status": "I",
            "recover_status": "R",
            "hosp_status": "H",
            "prim_hrate": 0.2,
            "sec_hrate": 0.05,
            "stochastic": False,
            "infstate_compartments": infstate_compartments,
        }
    }
