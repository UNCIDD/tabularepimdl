import importlib
import logging
from typing import Any, TypeAlias

from pydantic import BaseModel, ValidationError

from tabularepimdl._validators.rule_validation.rule_exceptions import RuleDefinitionError
from tabularepimdl.Rule import Rule

logger = logging.getLogger(__name__)


RuleDict: TypeAlias = dict[str, dict[str, Any]]
RulesSection: TypeAlias = RuleDict | list["RulesSection"]


def from_dict(rule_dict: dict) -> Rule:
    """
    Load a rule from its full dictionary definition.

    Args:
        rule_dict: a dictionary (key-value pair) defining one rule class read from a YAML file.
        It must contain exactly one key, in the form 'tabularepimdl.<RuleClass>'.

    Returns:
        An instantiated rule class object with parameter values initialized.

    Raises:
        RuleDefinitionError: If the definition is malformed, names an unknown or non-Pydantic rule class,
        or its parameters do not match the rule class.
    """
    if not isinstance(rule_dict, dict):
        raise RuleDefinitionError(f"Each rule must be a mapping of 'tabularepimdl.<RuleClass>' to its parameters, received {type(rule_dict).__name__}.")

    if len(rule_dict) != 1:
        raise RuleDefinitionError(
            f"Each rule entry must contain exactly one rule class, received {len(rule_dict)} keys: {list(rule_dict)}. If several rules were intended, start each one with '- ' in the YAML list."
        )

    key = next(iter(rule_dict))

    if not isinstance(key, str) or "." not in key:
        raise RuleDefinitionError(f"Expected format 'package_name.class_name', received {key!r}")

    mod_nm, cls_nm = key.rsplit(".", 1)
    logger.debug("mod_nm is %s, cls_nm is %s: ", mod_nm, cls_nm)

    if mod_nm != "tabularepimdl":  # Ensure the correct tabularepimdl module is imported
        raise RuleDefinitionError(f"Expected package name 'tabularepimdl', received {mod_nm}")

    mod = importlib.import_module(mod_nm)
    rule_cls = getattr(mod, cls_nm, None)  # rule_cls is expected to be a class defined in tabularepimdl

    if not isinstance(rule_cls, type) or not issubclass(rule_cls, Rule):
        raise RuleDefinitionError(f"{key} is not a Rule subclass defined in tabularepimdl")

    if not issubclass(rule_cls, BaseModel):
        raise RuleDefinitionError(f"{key} is not a Pydantic-based Rule and cannot be loaded from YAML")

    yaml_para_definition = rule_dict[key]  # obtain the parameter values from yaml's data
    _validate_definition(rule_cls, yaml_para_definition)  # validate parameter names first

    try:
        return rule_cls(**yaml_para_definition)  # parameter values are passed to class rule_cls, then the class is instantiated and returned
    except ValidationError as exc:
        raise RuleDefinitionError(f"Invalid parameter values for {rule_cls.__name__}:\n{exc}") from exc


def _validate_definition(tepi_rule: type[BaseModel], definition: dict) -> None:
    """
    Process an epidemic rule's class definition from a YAML file
    and validate its definition keys against the same Pydantic-integrated model's fields.

    Args:
        tepi_rule: a tabularepimdl rule class.
        definition: parameters defined from a YAML file for the above tabularepimdl class.

    Raises:
        RuleDefinitionError: If the definition is not a dictionary.
        RuleDefinitionError: If the YAML file misses fields the corresponding rule requires.
        RuleDefinitionError: If the YAML file includes fields the corresponding rule does not have.
    """
    # Expect the parameter definition to be a dictionary type data.
    if not isinstance(definition, dict):
        raise RuleDefinitionError(f"Epidemic rule's parameters must be in dictionary type. Received {type(definition).__name__}")

    rule_fields_mapping = tepi_rule.model_fields  # get all field items from Pydantic-integrated model
    logger.debug("tepi rule fields mapping: %s", rule_fields_mapping)

    rule_required_fields = {name for name, field in rule_fields_mapping.items() if field.is_required()}  # get required fields defined in a rule class
    logger.debug("rule required fields: %s", rule_required_fields)  # e.g. {'column', 'from_st', 'to_st', 'rate'}

    rule_all_fields = set(rule_fields_mapping.keys())  # get all field names defined in a rule class
    logger.debug("rule all fields: %s", rule_all_fields)  # e.g. {'to_st', 'column', 'from_st', 'rate', 'stochastic'}

    # Check whether the YAML rule definitions match the fields of the corresponding package rule class.
    yaml_provided_fields = set(definition.keys())  # get all the field names defined in a yaml file
    logger.debug("yaml provided fields: %s", yaml_provided_fields)

    missing_fields = rule_required_fields - yaml_provided_fields  # identify fields defined in the rule class that are missing from the yaml file
    logger.debug("missing fields: %s", missing_fields)
    extra_fields = yaml_provided_fields - rule_all_fields  # identify fields defined in the yaml file that do not belong to the rule class.
    logger.debug("extra fields: %s", extra_fields)

    if missing_fields:
        raise RuleDefinitionError(f"YAML file missed required fields for {tepi_rule.__name__}: {sorted(missing_fields)}")
    if extra_fields:
        raise RuleDefinitionError(f"Unexpected parameters in YAML file for {tepi_rule.__name__}: {sorted(extra_fields, key=str)}")


def _rule_build(rules_section: RulesSection) -> Any:
    """
    Recursively instantiate every rule dictionary in the 'rules' section, keeping the nesting of the input.

    Args:
        rules_section: the input rules in dictionary format: one rule, a list of rules, or a list of lists of rules.
    """
    if isinstance(rules_section, dict):
        return from_dict(rules_section)
    if isinstance(rules_section, list):
        return [_rule_build(item) for item in rules_section]
    raise RuleDefinitionError(f"Unsupported rule format, must be dict or list, received {type(rules_section).__name__}.")


def instantiate_rules(rules_section: RulesSection) -> list[list[Rule]]:
    """
    Instantiate the 'rules' section of a YAML configuration into rule sets.

    The grouping mirrors how the model engine reads its `rules` argument:
    a single rule becomes one rule set, a flat list of rules is one rule set, and
    in a list mixing rules and sub-lists each single rule and each sub-list is its own rule set.

    Args:
        rules_section: the input rules in dictionary format: one rule, a list of rules, or a list of lists of rules.

    Returns:
        A list of rule sets, each rule set being a non-empty list of Rule objects.

    Raises:
        RuleDefinitionError: If any rule is malformed, a list is empty, or lists are nested deeper than two levels.
    """
    rule_built = _rule_build(rules_section)

    if isinstance(rule_built, Rule):
        return [[rule_built]]

    if not rule_built:
        raise RuleDefinitionError("The 'rules' section is empty, at least one rule is required.")

    if all(isinstance(item, Rule) for item in rule_built):
        return [rule_built]

    rulesets: list[list[Rule]] = []
    for i, item in enumerate(rule_built):
        if isinstance(item, Rule):
            rulesets.append([item])
        elif not item:
            raise RuleDefinitionError(f"Rule set at rules[{i}] is empty, at least one rule is required.")
        elif all(isinstance(sub_item, Rule) for sub_item in item):
            rulesets.append(item)
        else:
            raise RuleDefinitionError(f"Too much nesting at rules[{i}]: rules can be a list of rules or a list of lists of rules, at most two levels deep.")

    return rulesets
