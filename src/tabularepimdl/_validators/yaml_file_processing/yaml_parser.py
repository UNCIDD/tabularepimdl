"""
This module performs:
1. YAML parsing
2. duplicate-key detection
3. list-index tracking
4. detailed error reporting
"""

from __future__ import annotations

from typing import Any

import yaml

from .yaml_exceptions import DuplicateYAMLKeyError, YAMLParseError


class DuplicateKeyLoader(yaml.SafeLoader):
    """
    PyYAML SafeLoader that rejects duplicate mapping keys.
    """

    def __init__(
        self,
        stream: Any,
        *,
        filename: str = "<string>",
    ) -> None:
        self.filename = filename
        self._path_stack: list[str] = []

        super().__init__(stream)


def _format_path(path_parts: list[str]) -> str:
    """
    Convert YAML path components into a readable path.

    Example:
        ["rules", "foo", "rate"]
        -> "rules.foo.rate"
    """

    if not path_parts:
        return "<root>"

    result = ""

    for part in path_parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            if result:
                result += "."

            result += str(part)

    return result


def _construct_mapping(
    loader: DuplicateKeyLoader,
    node: yaml.SequenceNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """
    Construct a mapping while rejecting duplicate keys.
    """

    mapping: dict[Any, Any] = {}
    key_locations: dict[Any, tuple[int, int]] = {}

    current_path = loader._path_stack.copy()

    for key_node, value_node in node.value:

        # Convert YAML key into a Python object.
        key = loader.construct_object(
            key_node,
            deep=deep,
        )

        line = key_node.start_mark.line + 1
        column = key_node.start_mark.column + 1

        # ---------------------------------------------------------------
        # Duplicate-key check
        # ---------------------------------------------------------------

        if key in mapping:
            original_line, original_column = key_locations[key]

            raise DuplicateYAMLKeyError(
                filename=loader.filename,
                key=key,
                path=_format_path(current_path + [str(key)]),
                original_line=original_line,
                original_column=original_column,
                duplicate_line=line,
                duplicate_column=column,
            )

        key_locations[key] = (line, column)

        # ---------------------------------------------------------------
        # Construct the value while keeping track of the YAML path.
        # ---------------------------------------------------------------

        loader._path_stack.append(str(key))

        try:
            value = loader.construct_object(
                value_node,
                deep=deep,
            )
        finally:
            loader._path_stack.pop()

        mapping[key] = value

    return mapping

def _construct_sequence(
    loader: DuplicateKeyLoader,
    node: yaml.SequenceNode,
    deep: bool = False,
) -> list[Any]:
    """
    Construct a YAML sequence/list while tracking list indexes
    in the YAML path.
    """

    sequence: list[Any] = []

    for index, child_node in enumerate(node.value):

        # -----------------------------------------------------------
        # Add the list index to the current YAML path.
        #
        # Example:
        #
        # rules
        #   ↓
        # rules[2]
        # -----------------------------------------------------------

        loader._path_stack.append(index)

        try:
            value = loader.construct_object(
                child_node,
                deep=deep,
            )
        finally:
            loader._path_stack.pop()

        sequence.append(value)

    return sequence

# Register custom mapping constructor.
DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


# Register custom sequence constructor.
DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
    _construct_sequence,
)


def parse_yaml(content: str, *, filename: str = "<string>") -> Any:
    """
    Parse YAML text safely.

    Args:
        content: YAML document as text.

        filename: Human-readable filename name used in error messages.

    Returns:
        Python representation of the YAML document.

    Raises:
        YAMLParseError: if the YAML syntax is invalid.
    """

    loader = DuplicateKeyLoader(
        content,
        filename=filename,
    )

    try:
        return loader.get_single_data()

    except DuplicateYAMLKeyError:
        # Preserve the detailed duplicate-key exception.
        raise

    except yaml.YAMLError as exc:
        raise YAMLParseError(
            f"Invalid YAML syntax in {filename}: {exc}"
        ) from exc

    finally:
        loader.dispose()
