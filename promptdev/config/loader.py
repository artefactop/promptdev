"""Configuration file loading utilities."""

from pathlib import Path
from typing import Any

from ..utils.file_utils import read_config_file, resolve_file_path_string
from .models import PromptDevConfig


def load_config(config_path: Path) -> PromptDevConfig:
    """Load PromptDev configuration from YAML or JSON file."""

    try:
        # Read and parse configuration file
        data = read_config_file(config_path)

        if data is None:
            raise ValueError(f"Config file is empty or contains only null values: {config_path}")
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to read config file: {config_path}\nError: {e}") from e

    try:
        # Resolve relative paths relative to config file location
        data = _resolve_relative_paths(data, config_path.parent)

        # Resolve $ref references in the configuration
        data = _resolve_refs(data)

        # Convert promptfoo-style assertions to AssertionConfig format
        data = _convert_promptfoo_assertions(data)

        # Create and validate configuration
        return PromptDevConfig(**data)
    except Exception as e:
        # Provide more helpful error message for common validation errors
        error_msg = str(e)
        if "Field required" in error_msg:
            raise ValueError(
                f"Missing required field in config file: {config_path}\n"
                f"Error: {error_msg}\n"
                f"Hint: Make sure you have 'prompts', 'providers', and 'tests' sections defined."
            ) from e
        if "ValidationError" in str(type(e)):
            raise ValueError(
                f"Configuration validation failed for: {config_path}\n"
                f"Error: {error_msg}\n"
                f"Hint: Check the format of your configuration file against the documentation."
            ) from e
        raise ValueError(
            f"Failed to load configuration from: {config_path}\nError: {error_msg}"
        ) from e


def _resolve_relative_paths(data: dict[str, Any], base_path: Path) -> dict[str, Any]:
    """Resolve relative file paths in configuration relative to config file location."""

    def resolve_recursive(obj: Any) -> Any:
        """Recursively resolve file paths in the data structure."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == "prompts" and isinstance(value, list):
                    # Resolve prompt file paths
                    result[key] = [resolve_file_path_string(item, base_path) for item in value]
                elif key == "tests" and isinstance(value, list):
                    # Resolve test dataset file paths
                    resolved_tests = []
                    for test in value:
                        if isinstance(test, dict) and "file" in test:
                            test_copy = dict(test)
                            test_copy["file"] = resolve_file_path_string(test["file"], base_path)
                            resolved_tests.append(test_copy)
                        else:
                            resolved_tests.append(resolve_recursive(test))
                    result[key] = resolved_tests
                elif key == "assertionTemplates" and isinstance(value, dict):
                    # Resolve assertion template file paths
                    resolved_templates = {}
                    for template_name, template_config in value.items():
                        if isinstance(template_config, dict) and "value" in template_config:
                            template_copy = dict(template_config)
                            template_copy["value"] = resolve_file_path_string(
                                template_config["value"], base_path
                            )
                            resolved_templates[template_name] = template_copy
                        else:
                            resolved_templates[template_name] = resolve_recursive(template_config)
                    result[key] = resolved_templates
                else:
                    result[key] = resolve_recursive(value)
            return result
        if isinstance(obj, list):
            return [resolve_recursive(item) for item in obj]
        return obj

    return resolve_recursive(data)


def _resolve_refs(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref references in the configuration data.

    This handles JSON Schema/YAML $ref syntax like:
    - $ref: '#/assertionTemplates/myTemplate'
    - $ref: '#/schemas/mySchema'
    """

    def resolve_ref(ref_path: str, root_data: dict) -> Any:
        """Resolve a single $ref path like '#/assertionTemplates/myTemplate'."""
        if not ref_path.startswith("#/"):
            raise ValueError(
                f"Only local references starting with '#/' are supported, got: {ref_path}"
            )

        # Remove '#/' prefix and split path
        path_parts = ref_path[2:].split("/")

        # Navigate through the data structure
        current = root_data
        for part in path_parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Reference not found: {ref_path}")
            current = current[part]

        return current

    def resolve_recursive(obj: Any, root_data: dict) -> Any:
        """Recursively resolve $ref references in the data structure."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                # This is a reference - resolve it
                ref_path = obj["$ref"]
                resolved = resolve_ref(ref_path, root_data)

                # If the reference resolves to a dict, merge with any additional properties
                if isinstance(resolved, dict):
                    # Recursively resolve the resolved object first (to handle nested $refs)
                    result = resolve_recursive(resolved, root_data)
                    # Merge any additional properties from the referencing object
                    for key, value in obj.items():
                        if key != "$ref":
                            if isinstance(result, dict):
                                result[key] = resolve_recursive(value, root_data)
                    return result
                return resolve_recursive(resolved, root_data)
            # Regular dict - recursively resolve its values
            return {key: resolve_recursive(value, root_data) for key, value in obj.items()}
        if isinstance(obj, list):
            return [resolve_recursive(item, root_data) for item in obj]
        return obj

    return resolve_recursive(data, data)


def _convert_promptfoo_assertions(data: dict[str, Any]) -> dict[str, Any]:
    """Convert promptfoo-style assertions to PromptDev AssertionConfig format."""

    def convert_assertion_list(assertions):
        """Convert a list of assertions to proper format."""
        # $ref resolution is now handled by _resolve_refs, so we just pass through
        return assertions

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "defaultTest" and isinstance(value, dict):
                # Convert defaultTest assertions
                if "assert" in value:
                    result_value = dict(value)
                    result_value["assert"] = convert_assertion_list(value["assert"])
                    result[key] = result_value
                else:
                    result[key] = value
            elif key == "tests" and isinstance(value, list):
                # Convert test assertions
                converted_tests = []
                for test in value:
                    if isinstance(test, dict) and "assert" in test:
                        converted_test = dict(test)
                        converted_test["assert"] = convert_assertion_list(test["assert"])
                        converted_tests.append(converted_test)
                    else:
                        converted_tests.append(test)
                result[key] = converted_tests
            else:
                result[key] = (
                    _convert_promptfoo_assertions(value)
                    if isinstance(value, dict | list)
                    else value
                )
        return result
    if isinstance(data, list):
        return [_convert_promptfoo_assertions(item) for item in data]
    return data
