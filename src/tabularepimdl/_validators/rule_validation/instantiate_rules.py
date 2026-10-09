import importlib
import logging

from typing import Any, TypeAlias
from tabularepimdl.Rule import Rule

logger = logging.getLogger(__name__)


RuleDict: TypeAlias = dict[str, dict[str, Any]]
RuleGroup: TypeAlias = list[RuleDict]
RulesSection: TypeAlias = RuleDict | list[RuleDict | RuleGroup]


def from_dict(rule_dict: dict) -> Rule:
        """
        Load a rule from its full dictionary definition.

        Args:
            rule_dict: a dictionary (key-value pair) defining the rule class(es) read from a YAML file.

        Returns:
            An instantiated rule class object with parameter values initialized.
        """
        key = next(iter(rule_dict))
        
        if "." not in key:
            raise ValueError(f"Expected format 'pacakge_name.class_name', received {key!r}")
        
        mod_nm, cls_nm = key.rsplit(".", 1)
        logger.debug("mod_nm is %s, cls_nm is %s: ", mod_nm, cls_nm)

        if mod_nm != "tabularepimdl":  # Ensure the correct tabularepimdl module is imported
            raise ValueError(f"Expected pacakge name 'tabularepimdl', received {mod_nm}")
            
        mod = importlib.import_module(mod_nm)
        rule_cls = getattr(mod, cls_nm)  # rule_cls is expected to be a class defined in tabularepimdl

        if not isinstance(rule_cls, type) or not issubclass(rule_cls, Rule):
            raise TypeError(f"{key} is not a Rule subclass")

        yaml_para_definition = rule_dict[key]  # obtain the parameter values from yaml's data
        _validate_definition(rule_cls, yaml_para_definition)  # validate parameter types and names first

        return from_yaml_def(rule_cls, rule_dict[key])  # key's dict-type values are passed to class rule_cls, then the class is instantiated and returned


def _validate_definition(tepi_rule: type[Rule], definition: dict) -> None:
        """
        Process an epidemic rule's class definition from a YAML file
        and validate its definition keys against the same Pydantic-integrated model's fields.

        Args:
            tepi_rule: a tabularepimdl rule class.
            definition: parameters defined from a YAML file for the above tabularepimdl class.

        Raises:
            ValueError: If the YAML file misses fields the corresponding rule requires.
            ValueError: If the YAML file includes fields the corresponding rule does not require.
        """
        # Expect the parameter defition to be a dictioary type data.
        if not isinstance(definition, dict):
            raise TypeError(f"Epidemic rule's parameters must be in dictionary type. Received {type(definition)}")

        rule_fields_mapping = tepi_rule.model_fields  # get all field items from Pydantic-integrated model
        logger.debug("tepi rule fields mapping: %s", rule_fields_mapping)

        rule_required_fields = {name for name, field in rule_fields_mapping.items() if field.is_required}  # get required fields defined in a rule class
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
            raise ValueError(f"YAML file missed required fields for {tepi_rule.__name__}: {missing_fields}")
        if extra_fields:
            raise ValueError(f"Unexpected parameters in YAML file for {tepi_rule.__name__}: {extra_fields}")


def from_yaml_def(rule_cls: type[Rule], definition: dict) -> Rule:
        """
        Instantiate the specified Rule subclass using parameters from YAML.

        Args:
            rule_cls: The concrete rule class to instantiate.
            definition: Dictionary containing the rule's parameters.

        Returns:
            An instantiated rule class object with parameter values.
        """
        return rule_cls(**definition)


def instantiate_rules(rules_section: RulesSection) -> list[list[Rule]]:
    """
    Recursively process the 'rules' section and return instantiated rules by invoking from_dict.

    Args:
        rules_section: the input rule sets in dictionary format.

    Returns:
        instantiated rule sets.
    """
    if isinstance(rules_section, dict):
        return from_dict(rules_section)
    elif isinstance(rules_section, list):
        return [instantiate_rules(item) for item in rules_section]
    else:
        raise TypeError(f"Unsupported rule format, must be dict or list, received {type(rules_section).__name__}.")