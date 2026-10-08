from pathlib import Path
from typing import Any

from tabularepimdl._validators.yaml_file_processing import (
    parse_yaml,
    read_yaml_file,
    validate_model,
    validate_yaml_path,
    validate_yaml_structure,
)


    
def load_yaml(filename: str | Path) -> Any:
    """
    Load and structurally validate a YAML file.

    Args:
        filename: Path to the file, provided as a string or pathlib.Path object.

    Validates:
        1. filename argument
        2. path
        3. file existence
        4. regular-file type
        5. YAML extension
        6. file readability
        7. UTF-8 encoding
        8. YAML syntax
        9. duplicate YAML keys
        10. non-empty document
        11. top-level mapping
        12. exactly one top-level key
        13. required top-level key 'rules'

    Returns:
        Parsed YAML as a dictionary.

    Raises:
        EpiModelValidationError: if any validation step fails.
    """
    # 1–5. Validate and normalize path
    path = validate_yaml_path(filename)
    print('path:', path)

    # 6–7. Read YAML file
    content = read_yaml_file(path)
    #print('content:', content)

    # 8–9. Parse YAML and detect duplicate keys
    parsed_data = parse_yaml(content, filename=str(path))
    print('parsed data\n', parsed_data)

    # 10–13. Validate YAML structure
    dict_data = validate_yaml_structure(parsed_data)

    # For now, return the parsed dictionary.
    # Model validation will be added later.
        
    return dict_data