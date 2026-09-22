"""
Unit test for WAIFWTransmission_Vec_Encode_Bincount.py.

Uses the same 2-group scenario (N counts, infection states, waifw matrix) as tests/test_waifwtransmission.py
"""

from unittest import mock

import numpy as np
import pytest

from tabularepimdl.WAIFWTransmission_Vec_Encode_Bincount import WAIFWTransmission_Vec_Encode_Bincount

# columns: InfState, Group, N
COL_IDX_MAP = {"InfState": 0, "Group": 1, "N": 2}
# encodings (alphabetical): InfState sorted(['S','I']) -> I=0, S=2; Group sorted(['GroupA','GroupB']) -> GroupA=0, GroupB=1
S, I_ = 2, 0
GROUP_A, GROUP_B = 0, 1
infstate_compartments = ["S", "I", "R"]


@pytest.fixture
def dummy_waifw_matrix():
    return np.array([[0.1, 0.2], [0.3, 0.4]])


@pytest.fixture
def dummy_state():
    """Mirrors test_waifwtransmission.py's dummy_state: N=[50,5,40,10], S/I/S/I, GroupA/GroupA/GroupB/GroupB."""
    return np.array([[S, GROUP_A, 50.0], [I_, GROUP_A, 5.0], [S, GROUP_B, 40.0], [I_, GROUP_B, 10.0]])


@pytest.fixture
def result_buffer():
    return np.empty((10, 3))


@pytest.fixture
def waifw_transmission(dummy_waifw_matrix):
    return WAIFWTransmission_Vec_Encode_Bincount(
        waifw_matrix=dummy_waifw_matrix, inf_col="InfState", group_col="Group", group_col_all_categories=["GroupA", "GroupB"], infstate_compartments=infstate_compartments
    )


def test_initialization(waifw_transmission):
    # waifw_matrix is transposed on construction, same as the pandas WAIFWTransmission rule.
    np.testing.assert_array_equal(waifw_transmission.waifw_matrix, np.array([[0.1, 0.3], [0.2, 0.4]]))
    assert waifw_transmission.inf_col == "InfState"
    assert waifw_transmission.group_col == "Group"
    assert waifw_transmission.group_col_all_categories == ["GroupA", "GroupB"]
    assert waifw_transmission.s_st == "S"
    assert waifw_transmission.i_st == "I"
    assert waifw_transmission.inf_to == "I"
    assert waifw_transmission.stochastic is False
    assert waifw_transmission.infstate_all == ["S", "I", "R"]
    assert waifw_transmission.group_col_all == ["GroupA", "GroupB"]
    assert waifw_transmission.expansion_factor == 2 * 3


def test_get_deltas_deterministic(waifw_transmission, dummy_state, dummy_waifw_matrix, result_buffer):
    deltas = waifw_transmission.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)

    inf_array = np.array([5.0, 10.0])  # sum of N where InfState==I, per group (GroupA, GroupB)
    prI_per_group = 1 - np.power(np.exp(-1.0 * dummy_waifw_matrix.T), inf_array).prod(axis=1)
    # cross-check against the exact values asserted in test_waifwtransmission.py's equivalent test
    assert prI_per_group == pytest.approx([0.9698026165776815, 0.9932620530009145])

    expected = np.array([[S, GROUP_A, -50 * prI_per_group[0]], [S, GROUP_B, -40 * prI_per_group[1]], [I_, GROUP_A, 50 * prI_per_group[0]], [I_, GROUP_B, 40 * prI_per_group[1]]])
    np.testing.assert_allclose(deltas, expected)


def test_get_deltas_stochastic(waifw_transmission, dummy_state, result_buffer):
    with mock.patch("numpy.random.binomial", return_value=20):
        deltas = waifw_transmission.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0, stochastic=True)
        expected = np.array([[S, GROUP_A, -20.0], [S, GROUP_B, -20.0], [I_, GROUP_A, 20.0], [I_, GROUP_B, 20.0]])
        np.testing.assert_allclose(deltas, expected)


def test_get_deltas_raises_when_data_has_more_groups_than_declared(waifw_transmission, result_buffer):
    state_with_more_groups = np.array(
        [
            [S, 0, 50.0],
            [I_, 1, 5.0],
            [S, 2, 40.0],  # a third group code, but only 2 categories were declared
        ]
    )
    with pytest.raises(ValueError):
        waifw_transmission.get_deltas(current_state=state_with_more_groups, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)


def test_get_deltas_raises_when_categories_mismatch_matrix_size(dummy_waifw_matrix, dummy_state, result_buffer):
    rule = WAIFWTransmission_Vec_Encode_Bincount(
        waifw_matrix=dummy_waifw_matrix,
        inf_col="InfState",
        group_col="Group",
        group_col_all_categories=["GroupA", "GroupB", "GroupC"],  # 3 categories, but a 2x2 matrix
        infstate_compartments=infstate_compartments,
    )
    with pytest.raises(ValueError):
        rule.get_deltas(current_state=dummy_state, col_idx_map=COL_IDX_MAP, result_buffer=result_buffer, dt=1.0)


def test_get_deltas_missing_n_column_raises(waifw_transmission, dummy_state, result_buffer):
    with pytest.raises(ValueError):
        waifw_transmission.get_deltas(current_state=dummy_state, col_idx_map={"InfState": 0, "Group": 1}, result_buffer=result_buffer, dt=1.0)


def test_to_dict(waifw_transmission, dummy_waifw_matrix):
    result = waifw_transmission.to_dict()
    d = result["tabularepimdl.WAIFWTransmission_Vec_Encode_Bincount"]
    np.testing.assert_array_equal(d.pop("waifw_matrix"), dummy_waifw_matrix)  # transposed back to original order for saving
    assert d == {
        "inf_col": "InfState",
        "group_col": "Group",
        "group_col_all_categories": ["GroupA", "GroupB"],
        "s_st": "S",
        "i_st": "I",
        "inf_to": "I",
        "stochastic": False,
        "infstate_compartments": infstate_compartments,
    }
