"""Configuration file loading utilities."""

import json
from pathlib import Path
from typing import Any

import yaml

from .models import PromptDevConfig


def load_config(config_path: Path) -> PromptDevConfig:
    """Load PromptDev configuration from YAML or JSON file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        # Read file content
        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}. "
                    f"Supported formats: .yaml, .yml, .json"
                )
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in config file: {config_path}\nError: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in config file: {config_path}\nError: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read config file: {config_path}\nError: {e}")

    if data is None:
        raise ValueError(f"Config file is empty or contains only null values: {config_path}")

    try:
        # Resolve relative paths relative to config file location
        data = _resolve_relative_paths(data, config_path.parent)

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
            )
        elif "ValidationError" in str(type(e)):
            raise ValueError(
                f"Configuration validation failed for: {config_path}\n"
                f"Error: {error_msg}\n"
                f"Hint: Check the format of your configuration file against the documentation."
            )
        else:
            raise ValueError(
                f"Failed to load configuration from: {config_path}\nError: {error_msg}"
            )


def _resolve_relative_paths(data: dict[str, Any], base_path: Path) -> dict[str, Any]:
    """Resolve relative file paths in configuration relative to config file location."""

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "prompts" and isinstance(value, list):
                # Resolve prompt file paths
                resolved_prompts = []
                for prompt in value:
                    if isinstance(prompt, str) and prompt.startswith("file://"):
                        # Remove file:// prefix and resolve relative to config
                        path_str = prompt[7:]  # Remove 'file://'
                        if not path_str.startswith("/"):  # Relative path
                            resolved_path = base_path / path_str
                            resolved_prompts.append(f"file://{resolved_path}")
                        else:
                            resolved_prompts.append(prompt)
                    else:
                        resolved_prompts.append(prompt)
                result[key] = resolved_prompts
            elif key == "tests" and isinstance(value, list):
                # Resolve test dataset file paths
                resolved_tests = []
                for test in value:
                    if isinstance(test, dict) and "file" in test:
                        file_path = test["file"]
                        if isinstance(file_path, str) and file_path.startswith("file://"):
                            path_str = file_path[7:]  # Remove 'file://'
                            if not path_str.startswith("/"):  # Relative path
                                resolved_path = base_path / path_str
                                test = dict(test)  # Create copy
                                test["file"] = resolved_path
                        resolved_tests.append(test)
                    else:
                        resolved_tests.append(test)
                result[key] = resolved_tests
            elif key == "assertionTemplates" and isinstance(value, dict):
                # Resolve assertion template file paths
                resolved_templates = {}
                for template_name, template_config in value.items():
                    if isinstance(template_config, dict) and "value" in template_config:
                        template_value = template_config["value"]
                        if isinstance(template_value, str) and template_value.startswith("file://"):
                            path_str = template_value[7:]  # Remove 'file://'
                            if not path_str.startswith("/"):  # Relative path
                                resolved_path = base_path / path_str
                                template_config = dict(template_config)  # Create copy
                                template_config["value"] = f"file://{resolved_path}"
                        resolved_templates[template_name] = template_config
                    else:
                        resolved_templates[template_name] = template_config
                result[key] = resolved_templates
            else:
                result[key] = _resolve_relative_paths(value, base_path)
        return result
    elif isinstance(data, list):
        return [_resolve_relative_paths(item, base_path) for item in data]
    else:
        return data


def _convert_promptfoo_assertions(data: dict[str, Any]) -> dict[str, Any]:
    """Convert promptfoo-style assertions to PromptDev AssertionConfig format."""

    def convert_assertion_list(assertions):
        """Convert a list of assertions to proper format."""
        converted = []
        for assertion in assertions:
            if isinstance(assertion, dict):
                if "$ref" in assertion and "type" not in assertion:
                    # Promptfoo-style $ref-only assertion
                    converted.append({"ref": assertion["$ref"]})
                else:
                    # Standard assertion format
                    converted.append(assertion)
            else:
                converted.append(assertion)
        return converted

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
    elif isinstance(data, list):
        return [_convert_promptfoo_assertions(item) for item in data]
    else:
        return data
