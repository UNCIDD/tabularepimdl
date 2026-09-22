"""
Unit test for MultiStrainInfectiousProcess_Vec_Encode.py.

Mirrors tests/test_multistraininfectiousprocess.py's exact scenario (N, Strain1/Strain2, betas,
cross_protect) and encode them as a NumPy array, and cross-checks against its independently-validated
prI values as ground truth.
"""

from unittest import mock

import numpy as np
import pytest

from tabularepimdl.MultiStrainInfectiousProcess_Vec_Encode import MultiStrainInfectiousProcess_Vec_Encode

# columns: Strain1, Strain2, N. Encoding is sorted(['S','I','R']) -> I=0, R=1, S=2.
COL_IDX_MAP = {"Strain1": 0, "Strain2": 1, "N": 2}
I_, R, S = 0, 1, 2
infstate_compartments = ["S", "I", "R"]

# cross-checked against tests/test_multistraininfectiousprocess.py's known expected_prI values
PRI_STRAIN1 = 0.043471260896961295
PRI_STRAIN2 = 0.01652854617837851


@pytest.fixture
def betas():
    return np.array([0.1, 0.05])


@pytest.fixture
def cross_protect():
    return np.array([[1.0, 0.5], [0.5, 1.0]])


@pytest.fixture
def dummy_state():
    """Mirrors test_multistraininfectiousprocess.py's dummy_state: N=[100,200,150],
    Strain1=['S','I','R'], Strain2=['S','S','I'].
    """
    return np.array([[S, S, 100.0], [I_, S, 200.0], [R, I_, 150.0]])


@pytest.fixture
def result_buffer():
    return np.empty((20, 3))


@pytest.fixture
def multistrain_infectiousprocess(betas, cross_protect):
    return MultiStrainInfectiousProcess_Vec_Encode(
        betas=betas, columns=["Strain1", "Strain2"], columns_all_categories=["S", "I", "R"], cross_protect=cross_protect, infstate_compartments=infstate_compartments
    )


def test_initialization(multistrain_infectiousprocess, betas, cross_protect):
    np.testing.assert_array_equal(multistrain_infectiousprocess.betas, betas)
    assert multistrain_infectiousprocess.columns == ["Strain1", "Strain2"]
    np.testing.assert_array_equal(multistrain_infectiousprocess.cross_protect, cross_protect)
    assert multistrain_infectiousprocess.s_st == "S"
    assert multistrain_infectiousprocess.i_st == "I"
    assert multistrain_infectiousprocess.r_st == "R"
    assert multistrain_infectiousprocess.inf_to == "I"
    assert multistrain_infectiousprocess.stochastic is False
    assert multistrain_infectiousprocess.freq_dep is True
    assert multistrain_infectiousprocess.columns_all == ["I", "R", "S"]


def test_initialization_mismatched_betas_and_columns_raises(cross_protect):
    with pytest.raises(ValueError):
        MultiStrainInfectiousProcess_Vec_Encode(
            betas=np.array([0.1, 0.05, 0.2]), columns=["Strain1", "Strain2"], columns_all_categories=["S", "I", "R"], cross_protect=cross_protect, infstate_compartments=infstate_compartments
        )


def test_initialization_non_square_cross_protect_raises(betas):
    with pytest.raises(ValueError):
        MultiStrainInfectiousProcess_Vec_Encode(
            betas=betas, columns=["Strain1", "Strain2"], columns_all_categories=["S", "I", "R"], cross_protect=np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]), infstate_compartments=infstate_compartments
        )


def test_get_deltas_deterministic_matches_pandas_known_values(multistrain_infectiousprocess, dummy_state, result_buffer):
    """Only dummy_state row0 (fully susceptible to both strains) can become newly infected -- rows 1 and 2 are
    already infected with one strain, so the 'no coinfections' rule zeroes their probability.
    """
    deltas = multistrain_infectiousprocess.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    neg_total = -100 * (1 - (1 - PRI_STRAIN1) * (1 - PRI_STRAIN2))
    add_strain1 = -neg_total * (PRI_STRAIN1 / (PRI_STRAIN1 + PRI_STRAIN2))
    add_strain2 = -neg_total * (PRI_STRAIN2 / (PRI_STRAIN1 + PRI_STRAIN2))

    expected = np.array(
        [
            [S, S, neg_total],  # subtraction: the original (S, S) row
            [I_, S, add_strain1],  # addition: infected by strain1
            [S, I_, add_strain2],  # addition: infected by strain2
        ]
    )
    np.testing.assert_allclose(deltas, expected)

    # additions should sum back to the magnitude of the subtraction (population conserved)
    assert add_strain1 + add_strain2 == pytest.approx(-neg_total)


def test_get_deltas_stochastic(multistrain_infectiousprocess, dummy_state, result_buffer):
    with mock.patch("numpy.random.multinomial", return_value=np.array([7, 2, 91])):
        deltas = multistrain_infectiousprocess.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([[S, S, -9.0], [I_, S, 7.0], [S, I_, 2.0]])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_no_infection_returns_empty(multistrain_infectiousprocess, result_buffer):
    all_susceptible = np.array([[S, S, 100.0], [S, S, 50.0]])
    deltas = multistrain_infectiousprocess.get_deltas(current_state=all_susceptible, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)
    assert deltas.shape == (0, 3)


def test_get_deltas_missing_n_column_raises(multistrain_infectiousprocess, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        multistrain_infectiousprocess.get_deltas(current_state=dummy_state, col_idx_map={"Strain1": 0, "Strain2": 1}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(multistrain_infectiousprocess, betas, cross_protect):
    result = multistrain_infectiousprocess.to_dict()
    d = result["tabularepimdl.MultiStrainInfectiousProcess_Vec_Encode"]
    np.testing.assert_array_equal(d.pop("betas"), betas)
    np.testing.assert_array_equal(d.pop("cross_protect"), cross_protect)
    assert d == {
        "columns": ["Strain1", "Strain2"],
        "columns_all_categories": ["I", "R", "S"],
        "s_st": "S",
        "i_st": "I",
        "r_st": "R",
        "inf_to": "I",
        "stochastic": False,
        "freq_dep": True,
        "infstate_compartments": infstate_compartments,
    }
