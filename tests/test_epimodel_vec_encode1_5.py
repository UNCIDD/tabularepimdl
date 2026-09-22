"""
Unit tests for EpiModel_Vec_Encode1_5.py (EpiModel_Vec_Encode_1_5) -- focuses on the the NumPy engine's own mechanics.

The following tests exercise init_state validation, the encode/decode round-trip, Reset(), add_new_data_to_current_state(),
and the dynamic delta-buffer growth logic. Cross-engine numerical parity for do_timestep() itself is covered separately in
tests/test_engine_parity.py.
"""

import numpy as np
import pandas as pd
import pytest

from tabularepimdl.EpiModel_Vec_Encode1_5 import EpiModel_Vec_Encode_1_5
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode

infstate_compartments = ["S", "I", "R"]


def _init_state() -> pd.DataFrame:
    return pd.DataFrame({"InfState": ["S", "I"], "N": [990.0, 10.0], "T": [0, 0]})


def _transition_rule() -> SimpleTransition_Vec_Encode:
    return SimpleTransition_Vec_Encode(column="InfState", from_st="I", to_st="R", rate=0.2, stochastic=False, infstate_compartments=infstate_compartments, column_categories=infstate_compartments)


# ---------------------------------------------------------------------------
# init_state validation
# ---------------------------------------------------------------------------


def test_init_state_must_be_dataframe():
    with pytest.raises(TypeError):
        EpiModel_Vec_Encode_1_5(init_state={"N": [1], "T": [0]}, rules=[[_transition_rule()]])


def test_init_state_requires_n_and_t_columns():
    with pytest.raises(ValueError):
        EpiModel_Vec_Encode_1_5(init_state=pd.DataFrame({"InfState": ["S"]}), rules=[[_transition_rule()]])


def test_init_state_n_must_be_numeric():
    invalid_df = pd.DataFrame({"InfState": ["S"], "N": ["ten"], "T": [0]})
    with pytest.raises(ValueError):
        EpiModel_Vec_Encode_1_5(init_state=invalid_df, rules=[[_transition_rule()]])


def test_init_state_column_names_must_be_strings():
    invalid_df = pd.DataFrame({0: ["S"], "N": [1], "T": [0]})
    with pytest.raises(ValueError):
        EpiModel_Vec_Encode_1_5(init_state=invalid_df, rules=[[_transition_rule()]])


# ---------------------------------------------------------------------------
# rules-list normalization (mirrors EpiModel's own validate_rules_list
# ---------------------------------------------------------------------------


def test_single_rule_is_wrapped_in_nested_list():
    rule = _transition_rule()
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=rule)
    assert model.rules == [[rule]]


def test_flat_list_of_rules_becomes_one_group():
    r1, r2 = _transition_rule(), _transition_rule()
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[r1, r2])
    assert model.rules == [[r1, r2]]


def test_invalid_rules_type_raises():
    with pytest.raises(TypeError):
        EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules="not-a-rule")


# ---------------------------------------------------------------------------
# encode/decode round trip and column shuffling
# ---------------------------------------------------------------------------


def test_current_state_round_trips_to_original_values():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    state = model.current_state()
    assert set(state["InfState"]) == {"S", "I"}
    assert sorted(state["N"]) == [10.0, 990.0]
    assert (state["T"] == 0).all()


def test_model_post_init_moves_n_and_t_to_end_of_column_order():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    assert model._init_state_col_order[-2:] == ["N", "T"]


# ---------------------------------------------------------------------------
# do_timestep: manually-computed single-rule result
# ---------------------------------------------------------------------------


def test_do_timestep_matches_manually_computed_transition():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    model.do_timestep(dt=1.0)

    state = model.current_state().set_index("InfState")["N"]
    rate_const = 1 - np.exp(-1.0 * 0.2)
    assert state["S"] == pytest.approx(990.0)  # untouched by an I->R rule
    assert state["I"] == pytest.approx(10.0 * (1 - rate_const))
    assert state["R"] == pytest.approx(10.0 * rate_const)


def test_reset_to_restore_initial_state():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    model.do_timestep(dt=1.0)
    model.Reset()

    state = model.current_state()
    assert sorted(state["N"]) == [10.0, 990.0]
    assert (state["T"] == 0).all()


# ---------------------------------------------------------------------------
# add_new_data_to_current_state
# ---------------------------------------------------------------------------


def test_add_new_data_appends_a_row():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    new_data = pd.DataFrame({"InfState": ["R"], "N": [5.0], "T": [0]})
    updated = model.add_new_data_to_current_state(new_data)
    assert updated.shape[0] == 3  # 2 original rows + 1 new row


def test_add_new_data_wrong_column_count_raises():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    missing_column_T = pd.DataFrame({"InfState": ["R"], "N": [5.0]})
    with pytest.raises(ValueError):
        model.add_new_data_to_current_state(missing_column_T)


def test_add_new_data_wrong_column_names_raises():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    wrong_column_name = pd.DataFrame({"InfState": ["R"], "Count": [5.0], "T": [0]})
    with pytest.raises(ValueError):
        model.add_new_data_to_current_state(wrong_column_name)


# ---------------------------------------------------------------------------
# dynamic delta-buffer growth
# ---------------------------------------------------------------------------


def test_delta_buffer_grows_when_current_state_exceeds_initial_buffer_capacity():
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    initial_buffer_rows = model._delta_buffer.shape[0]

    # grow current_state_array well past the initial buffer size which is computed at construction time
    for i in range(10):
        model.add_new_data_to_current_state(pd.DataFrame({"InfState": ["S"], "N": [1.0 + i], "T": [0]}))

    assert model.current_state_array.shape[0] > initial_buffer_rows

    total_n_before = model.current_state_array[:, model._n_idx].sum()
    model.do_timestep(dt=1.0)  # mayt trigger a buffer resize rather than crash
    total_n_after = model.current_state_array[:, model._n_idx].sum()

    assert model._delta_buffer.shape[0] > initial_buffer_rows
    assert total_n_after == pytest.approx(total_n_before)  # a transition rule conserves total population
