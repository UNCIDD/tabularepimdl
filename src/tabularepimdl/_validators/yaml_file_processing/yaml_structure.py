from __future__ import annotations

from typing import Any

from .yaml_exceptions import YAMLStructureError


REQUIRED_TOP_LEVEL_KEYS = {"rules"}


def validate_yaml_structure(data: Any) -> dict[str, Any]:
    """
    Validate the basic structure of parsed YAML data.

    Requirements:
    - YAML document must not be empty.
    - Top-level object must be a mapping/dictionary.
    - There must be exactly one top-level key.
    - That key must be 'rules'.

    Args:
        data: Result returned by yaml.safe_load() from yaml_parser module.

    Returns:
        The validated YAML mapping.

    Raises:
        YAMLStructureError:
            If the YAML structure is invalid.
    """

     # Empty document
    if data is None:
        raise YAMLStructureError("YAML document is empty.")

    # Top level must be a mapping
    if not isinstance(data, dict):
        raise YAMLStructureError(
            "Expected the top-level YAML document to be a mapping "
            f"(dictionary), received {type(data).__name__}."
        )

    # Exactly one top-level key
    if len(data) != 1:
        actual_keys = list(data.keys())

        raise YAMLStructureError(
            "YAML document must contain exactly one "
            f"top-level key: 'rules'. "
            f"Found: {actual_keys!r}"
        )
    
    # For now that key must be 'rules'
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - data.keys()

    if missing_keys:
        raise ValueError(
            f"Missing required top-level keys: {sorted(missing_keys)}. "
            f"Allowed top-level keys are {sorted(REQUIRED_TOP_LEVEL_KEYS)}, "
            f"but found {sorted(data.keys())}."
        )
    
    return data
