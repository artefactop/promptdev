from collections.abc import Generator
from pathlib import Path
from typing import Any

import yaml
from pydantic_core import from_json, to_json


def read_text_file(file_path: Path) -> str:
    """Read a text file with UTF-8 encoding.

    Args:
        file_path: Path to the text file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read file {file_path}: {e}") from e


def read_yaml_file(file_path: Path) -> Any:
    """Read and parse a YAML file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {file_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read YAML file {file_path}: {e}") from e


def read_json_file(file_path: Path) -> Any:
    """Read and parse a JSON file using pydantic_core.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return from_json(f.read())
    except Exception as e:
        raise ValueError(f"Invalid JSON syntax in {file_path}: {e}") from e


def read_jsonl_file(file_path: Path) -> Generator[dict[str, Any]]:
    """Read and parse a JSONL file line by line.

    Args:
        file_path: Path to the JSONL file

    Yields:
        Parsed JSON objects from each line

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If any line contains invalid JSON
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield from_json(line.encode())
                except Exception as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {file_path}: {e}") from e
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read JSONL file {file_path}: {e}") from e


def write_json_file(file_path: Path, data: Any, *, indent: int = 2) -> None:
    """Write data to a JSON file using pydantic_core.

    Args:
        file_path: Path to write the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation level

    Raises:
        ValueError: If data cannot be serialized or file cannot be written
    """
    try:
        json_bytes = to_json(data, indent=indent)
        with open(file_path, "wb") as f:
            f.write(json_bytes)
    except Exception as e:
        raise ValueError(f"Failed to write JSON file {file_path}: {e}") from e


def read_file(file_path: Path) -> Any:
    """Read a configuration file, auto-detecting format.

    Args:
        file_path: Path to the configuration file

    Returns:
        Parsed configuration data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or content is invalid
    """

    suffix = file_path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        return read_yaml_file(file_path)
    if suffix == ".json":
        return read_json_file(file_path)
    if suffix == ".jsonl":
        return read_jsonl_file(file_path)
    if suffix == ".txt":
        return read_text_file(file_path)

    raise ValueError(
        f"Unsupported config file format: {file_path.suffix}. Supported formats: .yaml, .yml, .json, .jsonl, .txt"
    )
