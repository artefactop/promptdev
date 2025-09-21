from pathlib import Path
from typing import Any

from promptdev.config.schemas import PromptDevConfig
from promptdev.utils.file import read_yaml_file


def load_config(config_path: Path) -> PromptDevConfig:
    """Load config"""
    data = read_yaml_file(config_path)
    data = _resolve_refs(data)
    data = _resolve_file_urls(data, config_path.parent)

    return PromptDevConfig(**data)


def _resolve_file_urls(obj: Any, base: Path) -> Any:
    """Recursively resolve file:// paths relative to a base directory and convert them to Path"""
    if isinstance(obj, str) and obj.startswith("file://"):
        rel_path = obj.removeprefix("file://")
        return (base / rel_path).resolve()

    if isinstance(obj, list):
        return [_resolve_file_urls(x, base) for x in obj]

    if isinstance(obj, dict):
        return {k: _resolve_file_urls(v, base) for k, v in obj.items()}

    return obj


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
