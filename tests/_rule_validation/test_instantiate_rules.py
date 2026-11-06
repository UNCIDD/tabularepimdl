"""
Unit tests for instantiate_rules.py -- turning the dictionary form of a YAML 'rules' section into Rule objects.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from tabularepimdl._validators.rule_validation import RuleDefinitionError, instantiate_rules
from tabularepimdl._validators.rule_validation.instantiate_rules import from_dict
from tabularepimdl._validators.yaml_file_processing import EpiModelValidationError
from tabularepimdl.BirthProcess_Vec_Encode import BirthProcess_Vec_Encode
from tabularepimdl.Rule import Rule
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode

COMPARTMENTS = ["S", "I", "R"]
TRANSITION_KEY = "tabularepimdl.SimpleTransition_Vec_Encode"
BIRTH_KEY = "tabularepimdl.BirthProcess_Vec_Encode"


def _transition(**overrides) -> dict:
    """One rule dictionary the way it looks after the YAML file is parsed."""
    params = {"column": "InfState", "from_st": "S", "to_st": "I", "rate": 0.1, "infstate_compartments": COMPARTMENTS, "column_categories": COMPARTMENTS}
    params.update(overrides)
    return {TRANSITION_KEY: params} # {'tabularepimdl.SimpleTransition_Vec_Encode': {'column': 'InfState', 'from_st': 'S', 'to_st': 'I', 'rate': 0.1, 'infstate_compartments': ['S', 'I', 'R'], 'column_categories': ['S', 'I', 'R']}}


def _birth() -> dict:
    return {BIRTH_KEY: {"rate": 0.1, "stochastic": False, "column_to_sort": "InfState", "infstate_compartments": COMPARTMENTS}}
    # {"tabularepimdl.BirthProcess_Vec_Encode": {"rate": 0.1, "stochastic": False, "column_to_sort": "InfState", "infstate_compartments": COMPARTMENTS}}

# ---------------------------------------------------------------------------
# return shape: always a list of rule sets
# ---------------------------------------------------------------------------
def test_single_rule_dict_returns_one_ruleset():
    """Single dictionary is converted to a single ruleset."""
    result = instantiate_rules(_transition())
    assert len(result) == 1 and len(result[0]) == 1
    assert isinstance(result[0][0], SimpleTransition_Vec_Encode)


def test_flat_list_of_rules_is_one_ruleset():
    """A list of rules is converted to a single ruleset."""
    result = instantiate_rules([_transition(), _birth()])
    assert len(result) == 1
    assert [type(rule) for rule in result[0]] == [SimpleTransition_Vec_Encode, BirthProcess_Vec_Encode]


def test_list_of_lists_keeps_each_sublist_as_a_ruleset():
    """Two lists of rules are converted to two sublists of ruleset respectively."""
    result = instantiate_rules([[_transition(), _birth()], [_transition(from_st="I", to_st="R")]])
    assert [len(ruleset) for ruleset in result] == [2, 1]
    assert all(isinstance(rule, Rule) for ruleset in result for rule in ruleset)


def test_mixed_list_wraps_single_rules_and_keeps_sublists():
    """A single rule and a list of rules are converted to a single ruleset and a sublist of ruleset respectively."""
    result = instantiate_rules([_birth(), [_transition(), _transition(from_st="I", to_st="R")]])
    assert [len(ruleset) for ruleset in result] == [1, 2]


def test_return_shape_matches_what_the_engine_expects():
    """Every input shape comes back as list[list[Rule]], never a bare Rule or flat list."""
    sections: list[Any] = [_transition(), [_transition()], [[_transition()]], [_transition(), [_birth()]]]
    for section in sections:
        result = instantiate_rules(section)
        assert isinstance(result, list)
        assert all(isinstance(ruleset, list) and ruleset for ruleset in result)


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def test_optional_field_can_be_omitted():
    """'stochastic' has a default value, so a YAML rule does not need to list it."""
    rule = instantiate_rules(_transition())[0][0]
    assert rule.stochastic is False


def test_optional_field_can_be_provided():
    """'stochastic' default value can be changed."""
    rule = instantiate_rules(_transition(stochastic=True))[0][0]
    assert rule.stochastic is True


def test_missing_required_field_raises():
    """Remove key `rate` to trigger missed required fields error. .* matches zero or more of any character"""
    definition = _transition()
    del definition[TRANSITION_KEY]["rate"]
    with pytest.raises(RuleDefinitionError, match="missed required fields.*rate"):
        instantiate_rules(definition)


def test_unexpected_field_raises():
    """Add an extra key `not_a_field` to unexpected parameters error."""
    with pytest.raises(RuleDefinitionError, match="Unexpected parameters.*not_a_field"):
        instantiate_rules(_transition(not_a_field=1))


def test_parameters_must_be_a_mapping():
    """Pass an invalid dictionary, verify that the RuleDefinitionError (dict type) error is raised."""
    with pytest.raises(RuleDefinitionError, match="dictionary"):
        instantiate_rules({TRANSITION_KEY: "rate: 0.1"})  # type: ignore[dict-item]


def test_invalid_parameter_value_is_reported_as_rule_definition_error():
    """Assign an invalide value to rate to trigger RuleDefinitionError by cause of ValidationError."""
    with pytest.raises(RuleDefinitionError, match="Invalid parameter values for SimpleTransition_Vec_Encode") as exc_info:
        instantiate_rules(_transition(rate="fast"))
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_to_dict_round_trips_through_instantiate_rules():
    """Dump the original rule's parameters to a dictionary and rebuild the same rule with the dictionary, they should be the same rule."""
    original_rule = SimpleTransition_Vec_Encode(column="InfState", from_st="S", to_st="I", rate=0.3, infstate_compartments=COMPARTMENTS, column_categories=COMPARTMENTS)
    rebuilt_rule = instantiate_rules(original_rule.to_dict())[0][0]
    assert rebuilt_rule == original_rule


# ---------------------------------------------------------------------------
# one rule per entry
# ---------------------------------------------------------------------------
def test_two_rule_classes_in_one_entry_raises():
    """A missing '- ' in the YAML puts two classes under one list item; the second must not be dropped silently."""
    two_rules_in_one_entry = {**_transition(), **_birth()}
    with pytest.raises(RuleDefinitionError, match="exactly one rule class"):
        instantiate_rules([two_rules_in_one_entry])


def test_empty_rule_entry_raises():
    """Pass an empty rule to the instantiation process."""
    with pytest.raises(RuleDefinitionError, match="exactly one rule class"):
        instantiate_rules([{}])


# ---------------------------------------------------------------------------
# rule class name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("SimpleTransition_Vec_Encode", "package_name.class_name"),  # no package prefix
        ("otherpackage.SimpleTransition_Vec_Encode", "Expected package name 'tabularepimdl'"),
        ("tabularepimdl.DoesNotExist", "not a Rule subclass"),  # unknown class
        ("tabularepimdl.configure_logging", "not a Rule subclass"),  # exists, but is not a class
        ("tabularepimdl.Rule", "not a Pydantic-based Rule"),  # abstract base class
    ],
)
def test_bad_rule_class_name_raises(key, message):
    """Pass invalid rule name as the key, verify that the appropriate error is raised."""
    with pytest.raises(RuleDefinitionError, match=message):
        instantiate_rules({key: {}})


def test_non_string_rule_class_name_raises():
    """Pass non-string value as rule name, verify that the appropriate error is raised."""
    with pytest.raises(RuleDefinitionError, match="package_name.class_name"):
        instantiate_rules({1: {}})  # type: ignore[dict-item]


def test_from_dict_method_rejects_non_dict_data():
    """The from_dict() method should raise error for non-dict data."""
    with pytest.raises(RuleDefinitionError, match="mapping"):
        from_dict(["not", "dict", "data"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# structure of the rules section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", [None, "text", 5])
def test_unsupported_section_type_raises(section):
    """Pass and unsupported rule format, verify that the appropriate error is raised."""
    with pytest.raises(RuleDefinitionError, match="Unsupported rule format"):
        instantiate_rules(section)


def test_empty_rules_section_raises():
    """Pass an empty list, verify that the appropriate error is raised."""
    with pytest.raises(RuleDefinitionError, match="empty"):
        instantiate_rules([])


def test_empty_sublist_raises():
    """Pass an empty rule along with a valid rule in the list, verify that the appropriate error is raised."""
    with pytest.raises(RuleDefinitionError, match=r"rules\[1\] is empty"):
        instantiate_rules([[_transition()], []])


def test_nesting_deeper_than_two_levels_raises():
    """Pass a rule that has more than two nesting levels, verify that the appropriate error is raised."""
    with pytest.raises(RuleDefinitionError, match="Too much nesting"):
        instantiate_rules([[[_transition()]]])

def test_rule_definition_error_is_an_epimodel_validation_error():
    """Applications can catch EpiModelValidationError for both YAML-file and rule-definition problems."""
    with pytest.raises(EpiModelValidationError):
        instantiate_rules([[[_transition()]]])
