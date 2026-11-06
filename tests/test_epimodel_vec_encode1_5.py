"""
Unit tests for EpiModel_Vec_Encode1_5.py (EpiModel_Vec_Encode_1_5) -- focuses on the the NumPy engine's own mechanics.

The following tests exercise init_state validation, the encode/decode round-trip, Reset(), add_new_data_to_current_state(),
the dynamic delta-buffer growth logic, stoch_policy validation, loading rules with from_yaml() (rules-only mode), and the
epidemic history when a rule produces no deltas. Cross-engine numerical parity for do_timestep() itself is covered separately in
tests/test_engine_parity.py.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tabularepimdl._validators.rule_validation import RuleDefinitionError
from tabularepimdl._validators.yaml_file_processing import YAMLFileNotFoundError
from tabularepimdl.EpiModel_Vec_Encode1_5 import EpiModel_Vec_Encode_1_5
from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
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


# ---------------------------------------------------------------------------
# stoch_policy validation
# ---------------------------------------------------------------------------
def test_invalid_stoch_policy_raises():
    with pytest.raises(ValidationError, match="stoch_policy"):
        EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]], stoch_policy="stocastic")  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["rule_based", "stochastic"])
def test_valid_stoch_policy_runs(policy):
    """The sum of N should be around 1000 after one timestep run."""
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]], stoch_policy=policy)
    model.do_timestep(dt=1.0)
    assert model.current_state()["N"].sum() == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# from_yaml() and rules-only mode
# ---------------------------------------------------------------------------
def _rule_yaml(from_st: str, to_st: str, indent: str = "  ") -> str:
    return (
        f"{indent}- tabularepimdl.SimpleTransition_Vec_Encode:\n"
        f"{indent}    column: InfState\n"
        f"{indent}    from_st: {from_st}\n"
        f"{indent}    to_st: {to_st}\n"
        f"{indent}    rate: 0.2\n"
        f"{indent}    infstate_compartments: [S, I, R]\n"
        f"{indent}    column_categories: [S, I, R]\n"
    )

# create a valid temporary yaml file with two rules, the two rules will be in one ruleset
@pytest.fixture
def rules_yaml_file(tmp_path):
    path = tmp_path / "rules.yml"
    path.write_text("rules:\n" + _rule_yaml("S", "I") + _rule_yaml("I", "R"), encoding="utf-8")
    return path


def test_from_yaml_builds_engine_with_rules_only(rules_yaml_file):
    """Build the engine with the fixture rule yaml file only, no init_data."""
    model = EpiModel_Vec_Encode_1_5.from_yaml(rules_yaml_file)

    assert model.init_state is None
    assert [[type(rule) for rule in ruleset] for ruleset in model.rules] == [[SimpleTransition_Vec_Encode, SimpleTransition_Vec_Encode]]  # a flat YAML list is one rule set


def test_from_yaml_accepts_a_string_path(rules_yaml_file):
    """Convert the fixture rule yaml file to a string and pass the string to the engine."""
    model = EpiModel_Vec_Encode_1_5.from_yaml(str(rules_yaml_file))
    assert len(model.rules[0]) == 2 # one ruleset with two rules in it


def test_from_yaml_nested_lists_make_separate_rulesets(tmp_path):
    """Create another yaml file with - and four spaces to form two nested lists, which will be separate rulesets."""
    path = tmp_path / "nested.yml"
    path.write_text("rules:\n  -\n" + _rule_yaml("S", "I", indent="    ") + "  -\n" + _rule_yaml("I", "R", indent="    "), encoding="utf-8")

    model = EpiModel_Vec_Encode_1_5.from_yaml(path)

    assert [len(ruleset) for ruleset in model.rules] == [1, 1] # two separate rulesets


def test_from_yaml_missing_file_raises(tmp_path):
    """No yaml file is provided, verify that the appropriate error is raised."""
    with pytest.raises(YAMLFileNotFoundError):
        EpiModel_Vec_Encode_1_5.from_yaml(tmp_path / "missing.yml")


def test_from_yaml_unknown_rule_class_raises(tmp_path):
    """Invalid rule name is used, verify that the appropriate error is raised."""
    path = tmp_path / "unknown.yml"
    path.write_text("rules:\n  - tabularepimdl.DoesNotExist:\n      rate: 0.1\n", encoding="utf-8")

    with pytest.raises(RuleDefinitionError, match="DoesNotExist"):
        EpiModel_Vec_Encode_1_5.from_yaml(path)


def test_rules_only_engine_cannot_do_timestep(rules_yaml_file):
    """"The engine is built with fixture rule yaml file only, no data input, 
    verify that the appropriate Runtime error is raised when doing timestep."""
    model = EpiModel_Vec_Encode_1_5.from_yaml(rules_yaml_file)
    with pytest.raises(RuntimeError, match="No initial state data"):
        model.do_timestep()


def test_rules_only_engine_cannot_add_new_data(rules_yaml_file):
    """"The engine is built with fixture rule yaml file only, no data input, 
    verify that the appropriate Runtime error is raised when adding new data to the engine."""
    model = EpiModel_Vec_Encode_1_5.from_yaml(rules_yaml_file)
    with pytest.raises(RuntimeError, match="No initial state data"):
        model.add_new_data_to_current_state(pd.DataFrame({"InfState": ["I"], "N": [1.0], "T": [0]}))


# ---------------------------------------------------------------------------
# add_new_data_to_current_state validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("category", ["Inf", "i"])  # a typo, and a case mismatch of an existing category
def test_add_new_data_unknown_category_raises(category):
    """The category value in new data's InfState column does not match the initial data's category value."""
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    rows_before = model.current_state_array.shape[0]

    with pytest.raises(ValueError, match="Invalid categories in column 'InfState'"):
        model.add_new_data_to_current_state(pd.DataFrame({"InfState": [category], "N": [5.0], "T": [0]}))

    assert model.current_state_array.shape[0] == rows_before  # nothing was added
    assert not np.isnan(model.current_state_array).any()


def test_add_new_data_accepts_a_subset_of_existing_categories():
    """The engine takes categories from both initial data and rules, R is known through the rule.
    Adding new data with category R satisfies the scope of engine's existing categories.
    """
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    model.add_new_data_to_current_state(pd.DataFrame({"InfState": ["R"], "N": [5.0], "T": [0]}))  # 'R' is only known through the rule, not the init_state data
    assert "R" in model.current_state()["InfState"].tolist()


def test_add_new_data_column_names_are_case_sensitive():
    """The engine data's column names are case-sensitive. Mis-matched column names (including case mismatch) should raise ValueError."""
    model = EpiModel_Vec_Encode_1_5(init_state=_init_state(), rules=[[_transition_rule()]])
    with pytest.raises(ValueError, match="case-sensitive"):
        model.add_new_data_to_current_state(pd.DataFrame({"infstate": ["I"], "n": [5.0], "t": [0]}))


def test_add_new_data_does_not_change_grouping_columns():
    """Grouping columns are determined and fixed when the engine is initialized. Adding new data should not change the grouping columns."""
    init_state = pd.DataFrame({"Age": ["child", "adult"], "InfState": ["S", "I"], "N": [50.0, 1.0], "T": [0, 0]})
    model = EpiModel_Vec_Encode_1_5(init_state=init_state, rules=[[_transition_rule()]])
    grouping_cols_before = list(model._grouping_cols)

    # new data lists its columns in a different order than the model's internal order
    model.add_new_data_to_current_state(pd.DataFrame({"InfState": ["I"], "Age": ["adult"], "N": [3.0], "T": [0]}))

    assert model._grouping_cols == grouping_cols_before
    assert model.current_state_array.shape[0] == 3


# ---------------------------------------------------------------------------
# epidemic history when a rule produces no deltas
# ---------------------------------------------------------------------------
def _all_susceptible() -> pd.DataFrame:
    """Create a dataframe with susceptible people only."""
    return pd.DataFrame({"InfState": ["S"], "N": [100.0], "T": [0]})


def _infection_rule() -> SimpleInfection_Vec_Encode:
    """With nobody infected this rule produces no deltas."""
    return SimpleInfection_Vec_Encode(beta=0.5, column="InfState", s_st="S", i_st="I", inf_to="I", infstate_compartments=infstate_compartments, column_categories=infstate_compartments)


def _susceptible_to_infected_rule() -> SimpleTransition_Vec_Encode:
    return SimpleTransition_Vec_Encode(column="InfState", from_st="S", to_st="I", rate=0.1, infstate_compartments=infstate_compartments, column_categories=infstate_compartments)


def test_history_time_advances_every_step_when_no_rule_produces_deltas():
    """With no infection happening in simulation, the time should still advance."""
    model = EpiModel_Vec_Encode_1_5(init_state=_all_susceptible(), rules=[[_infection_rule()]])

    for _ in range(4):
        model.do_timestep(dt=1.0)

    assert sorted(model.full_epi()["T"].unique()) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len({id(state) for state in model._full_epi_list}) == len(model._full_epi_list)  # no history entry is the same array object as another


@pytest.mark.parametrize("make_rules", [lambda: [_infection_rule(), _susceptible_to_infected_rule()], lambda: [_susceptible_to_infected_rule(), _infection_rule()]], ids=["no-operation-first", "no-operation-last"])
def test_rule_without_deltas_does_not_discard_other_rules_in_the_ruleset(make_rules):
    """
    A rule that produces no state changes (no deltas) does not accidentally prevent other rules from being executed, 
    regardless of whether that no-operation rule appears first or last in the ruleset.
    """
    reference_model = EpiModel_Vec_Encode_1_5(init_state=_all_susceptible(), rules=[_susceptible_to_infected_rule()])
    reference_model.do_timestep(dt=1.0) # advance the reference model by one timestep

    model = EpiModel_Vec_Encode_1_5(init_state=_all_susceptible(), rules=make_rules())
    model.do_timestep(dt=1.0) # advance the model by one timestep

    sort_cols = ["InfState"]
    pd.testing.assert_frame_equal(
        model.current_state().sort_values(sort_cols).reset_index(drop=True),
        reference_model.current_state().sort_values(sort_cols).reset_index(drop=True),
    )
    assert model.current_state()["N"].sum() == pytest.approx(100.0)
    assert not model.current_state()["InfState"].isna().any() # there must not be any missing values in the InfState column.


def test_stochastic_run_that_dies_out_keeps_the_correct_history():
    np.random.seed(0)
    init_state = pd.DataFrame({"InfState": ["S", "I"], "N": [99.0, 1.0], "T": [0, 0]})
    recovery = SimpleTransition_Vec_Encode(column="InfState", from_st="I", to_st="R", rate=5.0, infstate_compartments=infstate_compartments, column_categories=infstate_compartments)
    model = EpiModel_Vec_Encode_1_5(init_state=init_state, rules=[[_infection_rule(), recovery]], stoch_policy="stochastic")

    for _ in range(100):
        model.do_timestep(dt=0.25)

    assert "I" not in model.current_state()["InfState"].tolist()  # the outbreak went extinct, so later steps have no infected rows
    assert sorted(model.full_epi()["T"].unique()) == pytest.approx(np.arange(101) * 0.25) # verify the model's output contains exactly the expected 101 unique timestep values, starting at 0, increasing by 0.25, and ending at 25.
