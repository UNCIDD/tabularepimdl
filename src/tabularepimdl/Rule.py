import importlib
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Rule(ABC):
    """
    Set up an abstract base class `Rule` which defines a generic interface for
    epidemic rules that are used in epidemic model engine.
    """

    """@param stochastic: whether the process is stochastic or deterministic."""
    stochastic: bool

    @abstractmethod
    def get_deltas(self, current_state: np.ndarray, col_idx_map: dict[str, int], result_buffer: np.ndarray, dt: float = 1.0, stochastic: bool | None = None) -> np.ndarray:
        """
        Method takes in current state and return a series of deltas to that state.
        It computes the population deltas for the current state at a given time step.

        Args:
            current_state (np.ndarray): A structured array representing the current epidemic state. Must include a column `'N'`, which indicates the population count.
            col_idx_map (dict): mapping of column names to their index positions. e.g. {'N':0, 'InfState':1, 'Hosp':2}
            result_buffer (np.ndarray): A pre-allocated array that will be populated with the computed deltas. This array is modified in-place and returned.
            dt (float): The size of the time step. Defaults to 1.0.
            stochastic (bool, optional): Whether to apply stochastic modeling. If `None`, the class-level `self.stochastic` attribute is used.

        Returns:
            np.ndarray: A NumPy structured array containing the population deltas.

        Raises:
            ValueError: If the column `'N'` is missing in `current_state`.
        """

    @property
    def expansion_factor(self) -> int:
        """Maximum number of rows this rule can return per input row.

        Used by the model engine to size the shared delta buffer before running a timestep.
        Implemented by every NumPy rule; not declared `@abstractmethod` because
        the legacy pandas rules in `legacy/pandas_reference/` also subclass `Rule` and don't
        implement it (pandas has no preallocated buffer to size).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement expansion_factor.")

    def _encode_categorical_states(self, data_domains) -> None:
        """
        Use the fully updated data columns' domain mapping values to encode this rule's own
        column state values. Called by the model engine once per column domain update; not
        intended to be called directly by users (see `model_post_init` for the equivalent
        self-encoding path used when a rule is tested standalone).

        Implemented by every NumPy rule; not declared `@abstractmethod` for the
        same reason as `expansion_factor` above.

        Args:
            data_domains: mapping of column name to that column's category-to-code mapping.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _encode_categorical_states.")

    @classmethod
    def from_yaml(cls, rule_yaml):
        """
        Load a rule from its full yaml definition. Should not be overridden.

        Args:
            rule_yaml: a dictionary (key-value pair) defining the class(es) read from yaml file.

        Returns:
            An instantiated class object with parameter values initialized.
        """
        key = list(rule_yaml.keys())[0]

        if "." in key:
            mod_nm, cls_nm = key.split(".")
            logger.debug("mod_nm is %s, cls_nm is %s: ", mod_nm, cls_nm)

            if mod_nm != "tabularepimdl":  # Ensure the correct tabularepimdl module is imported
                raise ImportError(f"Expected pacakge 'tabularepimdl' but received {mod_nm}.")
            mod = importlib.import_module(mod_nm)
            rule_cls = getattr(mod, cls_nm)  # rule_cls is expected to be a class defined in tabularepimdl

        yaml_para_definition = rule_yaml[key]  # obtain the parameter values from yaml's dictionary
        Rule._validate_definition(rule_cls, yaml_para_definition)  # validate parameter types and names first

        return rule_cls.from_yaml_def(rule_yaml[key])  # key's dict-type values are passed to class rule_cls, then the class is instantiated and returned

    @classmethod
    def _validate_definition(cls, tepi_rule, definition: dict) -> None:
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

        rule_required_fields = {name for name, field in rule_fields_mapping.items() if field.is_required}  # get required fields defined in an epidemic rule
        logger.debug("rule required fields: %s", rule_required_fields)  # debug, e.g. {'column', 'from_st', 'to_st', 'rate'}

        rule_all_fields = set(rule_fields_mapping.keys())  # get all field names defined in an epidemic rule
        logger.debug("rule all fields: %s", rule_all_fields)  # debug, e.g. {'to_st', 'column', 'from_st', 'rate', 'stochastic'}

        # Check if the YAML definitions match the epidemic rule class's __init__ fields.
        yaml_provided_fields = set(definition.keys())  # get all the field names defined in yaml file
        logger.debug("yaml provided fields: %s", yaml_provided_fields)

        missing_fields = rule_required_fields - yaml_provided_fields  # fields that are in epidemic rule but missed in yaml definition
        logger.debug("missing fields: %s", missing_fields)
        extra_fields = yaml_provided_fields - rule_all_fields  # fields that are in yaml definition but not belonged to the epidemic rule
        logger.debug("extra fields: %s", extra_fields)

        if missing_fields:
            raise ValueError(f"YAML file missed required fields for {tepi_rule.__name__}: {missing_fields}")
        if extra_fields:
            raise ValueError(f"Unexpected parameters in YAML file for {tepi_rule.__name__}: {extra_fields}")

    @classmethod
    def from_yaml_def(cls, definition):
        """
        Do any special processing of a rule class definition from a yaml file.

        Args:
            definition: A dictionary giving the parameters required by an epidemic rule.

        Returns:
            An instantiated class object with parameter values intialized.
        """
        return cls(**definition)

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Return a dictionary object appropriate for inclusion in a yaml definition of an epidemic.
        Should be a dictionary with the class name (in form module.classname)
        being the outer key containing information needed for the class to run `from_yaml`

        Returns:
            A dictionary representation of this object appropriate to read in by method `from_yaml`.
        """
