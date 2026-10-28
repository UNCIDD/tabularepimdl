import importlib

from tabularepimdl._validators.yaml_file_processing.yaml_load import load_yaml

yaml_load_module = importlib.import_module("tabularepimdl._validators.yaml_file_processing.yaml_load")



def test_load_yaml_with_valid_file(mocker, tmp_path):
    """Return the validated dictionary when all steps succeed."""

    # Create a temporary YAML file.
    data = """
        rules:
          SimpleTransition:
            column: InfState
            from_st: S
            to_st: I
            compartments:
              - S
              - I
              - R
        """
    yaml_file = tmp_path / "rules.yaml"
    yaml_file.write_text(data, encoding="utf-8")

    # Prepare the expected results from each dependency.
    expected_data = {
        "rules": {
            "SimpleTransition": {
                "column": "InfState",
                "from_st": "S",
                "to_st": "I",
                "compartments": ["S", "I", "R"]

            }
        }
    }

    # Mock the path validation function
    mock_validate_path = mocker.patch.object(
        yaml_load_module,
        "validate_yaml_path",
        return_value=yaml_file,
    )

    # Mock the file reading function
    # Return the YAML content that was written above
    mock_read_file = mocker.patch.object(
        yaml_load_module,
        "read_yaml_file",
        return_value=data,
    )

    # Mock the YAML parsing function
    mock_parse = mocker.patch.object(
        yaml_load_module,
        "parse_yaml",
        return_value=expected_data,
    )

    # Mock the structure validation function
    mock_validate_structure = mocker.patch.object(
        yaml_load_module,
        "validate_yaml_structure",
        return_value=expected_data,
    )

    # Call load_yaml() with the temporary file path
    result = load_yaml(yaml_file)

    # Check the returned dictionary
    assert result == expected_data

    # Verify each dependency was called correctly
    mock_validate_path.assert_called_once_with(yaml_file)

    mock_read_file.assert_called_once_with(yaml_file)
    
    mock_parse.assert_called_once_with(
        data,
        filename=str(yaml_file),
    )
    mock_validate_structure.assert_called_once_with(expected_data)