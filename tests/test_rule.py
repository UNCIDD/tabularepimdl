"""
Unit test for Rule.py. Pytest package is used.
The Rule class defines a class that represents a dummy rule.
The unit tests ensure that the class behaves as expected.
"""

import numpy as np
import pytest

from my_module import DummyRule
from tabularepimdl.Rule import Rule


@pytest.fixture
def rule():
    rule = DummyRule()
    return rule

def test_rule_can_be_instantiated(rule):
    """
    Check the subclass satisfies the abstract interface and exposes the expected properties.
    """
    assert isinstance(rule, Rule)
    assert rule.stochastic is False
    assert rule.expansion_factor == 3


def test_abstract_rule_cannot_be_instantiated():
    """
    Rule contains abstract methods, Python should raise TypeError 
    when users attempt to instantiate it directly.
    """
    with pytest.raises(TypeError):
        Rule()  # type: ignore[abstract]


class IncompleteRule(Rule):
    """A subclasss missing an abstract method"""
    def get_deltas(self, current_state, col_idx_map, result_buffer, dt=1.0, stochastic=None):
        return result_buffer


def test_incomplete_subclass_cannot_be_instantiated():
    """
    A subclass missing an abstract method cannot be instantiated since it does not implement to_dict().
    """
    with pytest.raises(TypeError):
        IncompleteRule()  # type: ignore[abstract]


def test_get_deltas_returns_expected_values(rule):
    """Verify that the method calculates the expected deltas for the dummy rule."""
    current_state = np.array([[100.0, 10.0], [200.0, 20.0],])

    col_idx_map = {"N": 0, "InfState": 1,}

    result_buffer = np.zeros_like(current_state)

    original_buffer_id = id(result_buffer)

    result = rule.get_deltas(
        current_state=current_state,
        col_idx_map=col_idx_map,
        result_buffer=result_buffer,
        dt=2.0
    )

    expected = np.array([[-200.0, 0.0], [-400.0, 0.0],])
    
    assert result is result_buffer
    assert id(result) == original_buffer_id
    np.testing.assert_array_equal(result, expected)


def test_get_deltas_raises_when_N_missing(rule):
    """Check for a missing N entry in the column map"""
    current_state = np.array([[100.0, 10.0]])
    col_idx_map = {"InfState": 0}
    result_buffer = np.zeros_like(current_state)

    with pytest.raises(
        ValueError,
        match="Missing required column 'N'",
    ):
        rule.get_deltas(current_state=current_state, col_idx_map=col_idx_map, result_buffer=result_buffer,)


class LegacyRule(Rule):
    """Define a legacy-style subclass that implements only the abstract methods."""
    def get_deltas(self, current_state, col_idx_map, result_buffer, dt=1.0, stochastic=None,):
        return result_buffer

    def to_dict(self):
        return {"LegacyRule": {}}


def test_expansion_factor_not_implemented():
    """Base class intentionally raises NotImplementedError if a subclass does not override this property."""
    rule = LegacyRule()

    with pytest.raises(
        NotImplementedError,
        match="LegacyRule does not implement expansion_factor",
    ):
        _ = rule.expansion_factor


def test_encode_categorical_states_note_implemented():
    rule = LegacyRule()

    with pytest.raises(
        NotImplementedError,
        match="does not implement encode_categorical_states",
    ):
        rule.encode_categorical_states(data_domains={})


def test_to_dict(rule):
    """
    Tests the to_dict() returns the expected dictionary.
    """
    result = rule.to_dict()

    assert isinstance(result, dict)
    assert result == {
        "DummyRule": {
            "stochastic": False,
        }
    }
