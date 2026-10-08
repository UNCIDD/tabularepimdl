from typing import Any, TypeAlias

from tabularepimdl.Rule import Rule

RuleDict: TypeAlias = dict[str, dict[str, Any]]
RuleGroup: TypeAlias = list[RuleDict]
RulesSection: TypeAlias = RuleDict | list[RuleDict | RuleGroup]


def instantiate_rules(rules_section: RulesSection) -> list[list[Rule]]:
    """Recursively process the 'rules' section and return instantiated rules by invoking Rule's from_yaml."""
    if isinstance(rules_section, dict):
        return Rule.from_yaml(rules_section)
    elif isinstance(rules_section, list):
        return [instantiate_rules(item) for item in rules_section]
    else:
        raise TypeError(f"Unsupported rule format, must be dict or list, received {type(rules_section).__name__}.")
