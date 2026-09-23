from __future__ import annotations
from typing import Any

import yaml

"""
Example.
rules:
  - tabularepimdl.BirthProcess_Vec_Encode:
      rate: 0.1
      stochastic: false
      rate: 0.2

The error can tell users:
Duplicate key 'rate'
File: <file_name.yml or .yaml>
Path: rules.tabularepimdl.BirthProcess_Vec_Encode.rate
Original occurrence: Line x, Column y
Duplicate occurrence: Line a, Column b
"""

class DuplicateYAMLKeyError(Exception):
    """
    Exception raised when a YAML mapping contains a duplicate key.

    Attributes:
        filename: name of the YAML file being loaded.
        key: the duplicate key of a key-value pair.
        path: the rule path identifying the location of the duplicate key.
        original_line: the line number of the key's first occurrence.
        original_column: the column number of the key's first occurrence.
        duplicate_line: the line number of the key's second occurrence.
        duplicate_column: the column number of the key's second occurrence.
    """
    def __init__(
        self,
        *,
        filename: str,
        key: Any,
        path: str,
        original_line: int,
        original_column: int,
        duplicate_line: int,
        duplicate_column: int,
    ) -> None:
        """
        Args:
            filename: name of the YAML file being loaded.
            key: the duplicate key of a key-value pair.
            path: the rule path identifying the location of the duplicate key.
            original_line: the line number of the key's first occurrence.
            original_column: the column number of the key's first occurrence.
            duplicate_line: the line number of the key's second occurrence.
            duplicate_column: the column number of the key's second occurrence.

        Returns:
            None
        """
        self.filename = filename
        self.key = key
        self.path = path
        self.original_line = original_line
        self.original_column = original_column
        self.duplicate_line = duplicate_line
        self.duplicate_column = duplicate_column

        #creates the text of the error
        message = (
            f"Duplicate key {key!r}\n"
            f"File: {filename}\n"
            f"Path: {path}\n"
            f"Original occurrence: "
            f"Line {original_line}, Column {original_column}\n"
            f"Duplicate occurrence: "
            f"Line {duplicate_line}, Column {duplicate_column}"
        )

        super().__init__(message)


class DuplicateKeyLoader(yaml.SafeLoader):
    """
    PyYAML SafeLoader that rejects duplicate keys within the same YAML mapping.

    Attributes:
        filename: name of the YAML file being loaded.
        _path_stack: stack of mapping keys representing the current YAML path during parsing.
    """
    def __init__(self, stream, *, filename: str = "<string>") -> None:
        """
        Args:
            stream: source from which PyYAML reads the YAML file.
            filename: name of the YAML file being loaded.

        Returns:
            None
        """
        self.filename = filename
        self._path_stack = []
        
        super().__init__(stream)


def _format_path(path_parts: list[str]) -> str: #Take a list describing a location in the YAML structure and turn it into readable text.
    """
    Convert path components into a readable path.

    Args:
        path_parts: a list of strings describing a location in the YAML structure.
            
    Returns:
        a string contains a rule's location. Example, ["rules", "foo", "rate"] -> "rules.foo.rate"
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


def _construct_mapping(loader: DuplicateKeyLoader, node: Any, deep: bool = False) -> dict:
    """
    Construct a Python dictionary while checking for duplicate
    keys within this particular YAML mapping.

    Args:
        loader: the PyYAML loader taking YAML text.
        node: the PyYAML internal structure holding key_node and value_node.
        deep: the PyYAML construction option controlling how deeply objects are constructed.

    Returns:
        the Python dictionay mapping of rules.

    Raises:
        DuplicateYAMLKeyError if a duplicate key is detected.
    """
    mapping = {}
    key_locations = {}

    # Get the current path from the loader.
    current_path = loader._path_stack.copy()
    
    for key_node, value_node in node.value:
        # Construct the key
        key = loader.construct_object(key_node, deep=deep) #turn this YAML key node into an ordinary Python object.

        # Location of this key in the YAML source.
        line = key_node.start_mark.line + 1
        column = key_node.start_mark.column + 1

        if key in mapping:
            original_line, original_column = key_locations[key]

            raise DuplicateYAMLKeyError(
                filename=loader.filename,
                key=key,
                path=_format_path(current_path + [key]),
                original_line=original_line,
                original_column=original_column,
                duplicate_line=line,
                duplicate_column=column,
            )

        # Remember where this key first occurred.
        key_locations[key] = (line, column)
        
        if isinstance(key, str):
            loader._path_stack.append(key)
        else:
            loader._path_stack.append(str(key))
        
        # Construct the value
        value = loader.construct_object(value_node, deep=deep)
        
        loader._path_stack.pop()
        
        mapping[key] = value

    return mapping

# Register the custom mapping constructor to detect duplicate keys
DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)

