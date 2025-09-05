"""Dataset management for PromptDev."""

import json
from pathlib import Path
from typing import Any

from ..config.models import DatasetConfig, TestConfig


class PromptDevDataset:
    """Dataset container for test cases."""

    def __init__(self, test_cases: list[dict[str, Any]]):
        """Initialize dataset with test cases.

        Args:
            test_cases: List of test case dictionaries
        """
        self.test_cases = test_cases

    @classmethod
    def from_config(cls, config: DatasetConfig | TestConfig) -> "PromptDevDataset":
        """Create dataset from configuration.

        Args:
            config: Dataset or test configuration

        Returns:
            PromptDevDataset instance
        """
        if isinstance(config, TestConfig):
            # Single test case
            test_case = {
                "vars": config.vars or {},
                "assertions": [assertion.dict() for assertion in config.assert_],
                "name": "single_test",
            }
            return cls([test_case])

        elif isinstance(config, DatasetConfig):
            if config.file:
                # Load from JSONL file
                file_path = config.file
                if isinstance(file_path, str):
                    # Handle file:// prefix
                    if file_path.startswith("file://"):
                        file_path = Path(file_path[7:])
                    else:
                        file_path = Path(file_path)
                return cls._load_from_jsonl(file_path)
            elif config.inline:
                # Inline test cases
                test_cases = []
                for i, case_data in enumerate(config.inline):
                    # Extract variables from case_data
                    if "vars" in case_data:
                        # Case data has explicit vars field
                        variables = case_data["vars"]
                        name = case_data.get("name", f"inline_test_{i}")
                    else:
                        # Case data is the variables directly
                        variables = {
                            k: v for k, v in case_data.items() if k not in ["name", "assertions"]
                        }
                        name = case_data.get("name", f"inline_test_{i}")

                    test_case = {
                        "vars": variables,
                        "name": name,
                        "assertions": case_data.get("assertions", []),
                    }
                    test_cases.append(test_case)
                return cls(test_cases)
            elif config.vars:
                # Single vars dictionary
                test_case = {"vars": config.vars, "name": "vars_test", "assertions": []}
                return cls([test_case])

        # Empty dataset
        return cls([])

    @classmethod
    def _load_from_jsonl(cls, file_path: Path) -> "PromptDevDataset":
        """Load dataset from JSONL file (compatible with promptfoo format).

        Expected format (from your existing files):
        {"vars": {"expected_name": "John", "calendar_event_summary": "John - Vacation"}}
        {"vars": {"expected_name": "Jane", "calendar_event_summary": "Jane - Meeting"}}

        Args:
            file_path: Path to JSONL file

        Returns:
            PromptDevDataset instance
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        test_cases = []
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Extract test case data
                    test_case = {
                        "vars": data.get("vars", {}),
                        "name": f"test_{line_num}",
                        "assertions": [],
                        "expected": data.get("expected"),
                        "metadata": data.get("metadata", {}),
                    }

                    # Extract expected values from vars (promptfoo format)
                    variables = data.get("vars", {})
                    expected_values = {}
                    for key, value in variables.items():
                        if key.startswith("expected_"):
                            expected_key = key[9:]  # Remove 'expected_' prefix
                            expected_values[expected_key] = value

                    if expected_values:
                        test_case["expected"] = expected_values

                    test_cases.append(test_case)

                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {file_path}: {e}")

        return cls(test_cases)

    def filter_by_metadata(self, **filters) -> "PromptDevDataset":
        """Filter test cases by metadata attributes.

        Args:
            **filters: Metadata key-value pairs to filter by

        Returns:
            New dataset with filtered test cases
        """
        filtered_cases = []
        for case in self.test_cases:
            metadata = case.get("metadata", {})
            if all(metadata.get(k) == v for k, v in filters.items()):
                filtered_cases.append(case)

        return PromptDevDataset(filtered_cases)

    def add_assertions(self, assertions: list[dict[str, Any]]) -> None:
        """Add assertions to all test cases in dataset.

        Args:
            assertions: List of assertion configurations
        """
        for test_case in self.test_cases:
            test_case.setdefault("assertions", []).extend(assertions)

    def __len__(self) -> int:
        """Get number of test cases."""
        return len(self.test_cases)

    def __iter__(self):
        """Iterate over test cases."""
        return iter(self.test_cases)
