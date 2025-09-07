"""Dataset management for PromptDev."""

from pathlib import Path
from typing import Any

from ..config.models import DatasetConfig, TestConfig, PromptDevConfig
from ..utils.file_utils import read_jsonl_file_adaptive, resolve_file_path
from pydantic_evals import Case, Dataset


class PromptDevDataset:
    """Dataset container for test cases that properly integrates with pydantic_evals."""

    def __init__(self, cases: list[Case], evaluators: list[Any] | None = None):
        """Initialize dataset with pydantic_evals Cases.

        Args:
            cases: List of pydantic_evals Case objects
            evaluators: Optional list of evaluators for the dataset
        """
        self.cases = cases
        self.evaluators = evaluators or []
        # Create the actual pydantic_evals Dataset
        self.dataset = Dataset(cases=self.cases, evaluators=self.evaluators)

    @classmethod
    def from_config(cls, config: DatasetConfig | TestConfig) -> "PromptDevDataset":
        """Create dataset from configuration.

        Args:
            config: Dataset or test configuration

        Returns:
            PromptDevDataset instance
        """
        if isinstance(config, TestConfig):
            # Single test case - create proper Case object with unique name
            import uuid

            case = Case(
                name=f"test_{uuid.uuid4().hex[:8]}",
                inputs=config.vars or {},
                expected_output=None,
                metadata={"assertions": [assertion.model_dump() for assertion in config.assert_]},
            )
            return cls([case])

        if isinstance(config, DatasetConfig):
            if config.file:
                # Load from JSONL file
                file_path = config.file
                if isinstance(file_path, str):
                    # Use centralized file path resolution
                    file_path = resolve_file_path(file_path)
                return cls._load_from_jsonl(file_path)
            if config.tests:  # Updated from config.inline to config.tests
                # Inline test cases - create proper Case objects
                cases = []
                for i, test_config in enumerate(config.tests):
                    case = Case(
                        name=getattr(test_config, "description", None) or f"inline_test_{i}",
                        inputs=test_config.vars or {},
                        expected_output=None,
                        metadata={
                            "assertions": [
                                assertion.model_dump() for assertion in test_config.assert_
                            ],
                            "options": getattr(test_config, "options", {}),
                            "provider": getattr(test_config, "provider", None),
                        },
                    )
                    cases.append(case)
                return cls(cases)
            if config.vars:
                # Single vars dictionary
                import uuid

                case = Case(
                    name=f"vars_test_{uuid.uuid4().hex[:8]}",
                    inputs=config.vars,
                    expected_output=None,
                    metadata={},
                )
                return cls([case])

        # Empty dataset
        return cls([])

    @classmethod
    def from_test_cases_with_evaluators(
        cls,
        test_cases: list[dict[str, Any]],
        config: PromptDevConfig,
        error_collector: list | None = None,
        verbose: bool = False,
    ) -> "PromptDevDataset":
        """Create dataset from a list of test cases, creating evaluators from assertions."""
        # Import here to avoid circular imports
        from ..evaluators.factory import create_pydantic_evaluator
        from ..evaluators.evaluators import FailureEvaluator

        evaluators = []
        cases = []
        evaluator_assertion_map = {}  # Maps evaluator index to assertion template name
        evaluator_type_map = {}  # Maps evaluator index to assertion type

        # Create pydantic_evals Case objects
        for i, test_case in enumerate(test_cases):
            case_name = test_case.get("name", f"test_case_{i + 1}")
            inputs = test_case.get("vars", {})
            expected_output = test_case.get("expected")
            metadata = test_case.get("metadata", {})
            case = Case(
                name=case_name, inputs=inputs, expected_output=expected_output, metadata=metadata
            )
            cases.append(case)

        # Create evaluators from the assertions of the FIRST test case
        # This assumes all test cases in the batch share the same assertions
        if test_cases:
            first_test_case_assertions = test_cases[0].get("assertions", [])
            for assertion in first_test_case_assertions:
                try:
                    evaluator = create_pydantic_evaluator(assertion, config, verbose=False)
                    evaluator_index = len(evaluators)
                    evaluators.append(evaluator)

                    # Track original assertion name and type
                    assertion_name = cls._get_assertion_name(assertion)
                    assertion_type = cls._get_assertion_type(assertion, config)
                    evaluator_assertion_map[evaluator_index] = assertion_name
                    evaluator_type_map[evaluator_index] = assertion_type
                except Exception as e:
                    import traceback

                    error_msg = f"Failed to create evaluator for assertion {assertion}: {e}"
                    if verbose:
                        print(f"Warning: {error_msg}")
                        traceback.print_exc()

                    if error_collector is not None:
                        error_collector.append(error_msg)

                    # Add a failure evaluator to track this error
                    evaluators.append(FailureEvaluator(error_message=error_msg))

        # Create dataset with evaluators
        dataset = cls(cases, evaluators)

        # Store additional metadata for compatibility
        dataset.config = config
        dataset.test_cases = test_cases
        dataset._error_collector = error_collector
        dataset.evaluator_assertion_map = evaluator_assertion_map
        dataset.evaluator_type_map = evaluator_type_map

        return dataset

    @staticmethod
    def _get_assertion_name(assertion) -> str:
        """Extract the assertion template name from assertion config."""
        if hasattr(assertion, "model_dump"):
            assertion_dict = assertion.model_dump()
        elif isinstance(assertion, dict):
            assertion_dict = assertion
        else:
            return "unknown_assertion"

        # $ref resolution is now handled during configuration loading
        # No runtime template reference checking needed

        # Fallback to type or generic name
        return assertion_dict.get("type", "unknown_assertion")

    @staticmethod
    def _get_assertion_type(assertion, config) -> str:
        """Extract the assertion type from assertion config."""
        if hasattr(assertion, "model_dump"):
            assertion_dict = assertion.model_dump()
        elif isinstance(assertion, dict):
            assertion_dict = assertion
        else:
            return "unknown"

        # Check for direct type in assertion
        if assertion_dict.get("type"):
            return assertion_dict["type"]

        # $ref resolution is now handled during configuration loading
        # Template types are resolved at load time

        return "unknown"

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
        cases = []

        # Use adaptive JSONL reader for optimal performance
        jsonl_data = read_jsonl_file_adaptive(file_path)

        for line_num, data in enumerate(jsonl_data, 1):
            # Extract variables and expected values
            variables = data.get("vars", {})
            expected_output = data.get("expected")

            # Extract expected values from vars (promptfoo format)
            if not expected_output:
                expected_values = {}
                for key, value in variables.items():
                    if key.startswith("expected_"):
                        expected_key = key[9:]  # Remove 'expected_' prefix
                        expected_values[expected_key] = value

                if expected_values:
                    expected_output = expected_values

            # Create proper Case object
            case = Case(
                name=data.get("name", f"test_{line_num}"),
                inputs=variables,
                expected_output=expected_output,
                metadata=data.get("metadata", {}),
            )
            cases.append(case)

        return cls(cases)

    def filter_by_metadata(self, **filters) -> "PromptDevDataset":
        """Filter test cases by metadata attributes.

        Args:
            **filters: Metadata key-value pairs to filter by

        Returns:
            New dataset with filtered test cases
        """
        filtered_cases = [
            case
            for case in self.cases
            if all(case.metadata.get(k) == v for k, v in filters.items())
        ]

        return PromptDevDataset(filtered_cases, self.evaluators)

    def add_evaluators(self, evaluators: list[Any]) -> None:
        """Add evaluators to the dataset.

        Args:
            evaluators: List of pydantic_evals evaluators
        """
        self.evaluators.extend(evaluators)
        # Recreate the dataset with new evaluators
        self.dataset = Dataset(cases=self.cases, evaluators=self.evaluators)

    async def evaluate(self, task_function, **kwargs) -> Any:
        """Run evaluation using pydantic_evals.

        Args:
            task_function: Function to evaluate
            **kwargs: Additional arguments to pass to dataset.evaluate

        Returns:
            Evaluation report
        """
        return await self.dataset.evaluate(task_function, **kwargs)

    def evaluate_sync(self, task_function, **kwargs) -> Any:
        """Run evaluation synchronously using pydantic_evals.

        Args:
            task_function: Function to evaluate
            **kwargs: Additional arguments to pass to dataset.evaluate_sync

        Returns:
            Evaluation report
        """
        return self.dataset.evaluate_sync(task_function, **kwargs)

    def __len__(self) -> int:
        """Get number of test cases."""
        return len(self.cases)

    def __iter__(self):
        """Iterate over test cases."""
        return iter(self.cases)
