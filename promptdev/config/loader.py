from pathlib import Path
from typing import Any

from promptdev.config.schemas import PromptDevConfig
from promptdev.utils.file import read_file, resolve_file_path


def load_config(config_path: Path) -> PromptDevConfig:
    """Load config"""
    data = read_file(config_path)
    data = _resolve_refs(data)
    data = _resolve_relative_paths(data, config_path.parent)

    return PromptDevConfig(**data)


def _resolve_relative_paths(data: dict[str, Any], base_path: Path) -> dict[str, Any]:
    """Resolve relative file paths in configuration relative to config file location."""

    def resolve_recursive(obj: Any) -> Any:
        """Recursively resolve file paths in the data structure."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == "prompts" and isinstance(value, list):
                    resolved_prompts = []
                    for item in value:
                        if isinstance(item, str) and item.startswith("file://"):
                            item = item.removeprefix("file://")
                            prompt_path = resolve_file_path(item, base_path)
                            if not prompt_path.exists():
                                raise FileNotFoundError(f"File not found: {prompt_path}")
                            resolved_prompts.append(str(prompt_path))
                        elif isinstance(item, str):
                            resolved_prompts.append(item)
                        else:
                            raise ValueError(f"Invalid prompt: {item}")
                    result[key] = resolved_prompts
                elif key == "tests" and isinstance(value, list):
                    # Resolve test dataset file paths
                    resolved_tests = []
                    for test in value:
                        if isinstance(test, dict) and "file" in test:
                            test_copy = dict(test)
                            test_copy["file"] = str(resolve_file_path(test["file"], base_path))
                            resolved_tests.append(test_copy)
                        else:
                            resolved_tests.append(resolve_recursive(test))
                    result[key] = resolved_tests
                elif key == "value" and isinstance(value, str) and value.startswith("file://"):
                    resolved_value = resolve_file_path(value, base_path)
                    if not resolved_value.exists():
                        raise FileNotFoundError(f"File not found: {resolved_value}")
                    result[key] = str(resolved_value)
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
                        if key != "$ref" and isinstance(result, dict):
                            result[key] = resolve_recursive(value, root_data)
                    return result
                return resolve_recursive(resolved, root_data)
            # Regular dict - recursively resolve its values
            return {key: resolve_recursive(value, root_data) for key, value in obj.items()}
        if isinstance(obj, list):
            return [resolve_recursive(item, root_data) for item in obj]
        return obj

    return resolve_recursive(data, data)
