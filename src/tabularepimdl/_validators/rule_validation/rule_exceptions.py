"""Exceptions raised while turning a rules definition (e.g. read from a YAML file) into Rule objects."""

from tabularepimdl._validators.yaml_file_processing.yaml_exceptions import EpiModelValidationError


class RuleDefinitionError(EpiModelValidationError):
    """Raised when a rules definition is malformed or cannot be turned into epidemic Rule objects."""
