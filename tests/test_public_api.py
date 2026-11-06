"""
Unit tests for the package-level public API exposed by tabularepimdl/__init__.py.
"""

import tabularepimdl


def test_load_yaml_and_instantiate_rules_are_importable_from_package_root():
    from tabularepimdl import instantiate_rules, load_yaml

    assert callable(load_yaml)
    assert callable(instantiate_rules)


def test_every_name_in_all_is_an_importable_attribute():
    """A name in __all__ that is missing or not a string breaks `from tabularepimdl import *`."""
    assert all(isinstance(name, str) for name in tabularepimdl.__all__)
    missing = [name for name in tabularepimdl.__all__ if not hasattr(tabularepimdl, name)]
    assert missing == []


def test_new_yaml_helpers_are_listed_in_all():
    assert {"load_yaml", "instantiate_rules"} <= set(tabularepimdl.__all__)
