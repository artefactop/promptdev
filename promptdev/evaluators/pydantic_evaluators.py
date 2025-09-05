"""PydanticAI-based evaluators for PromptDev using pydantic_evals."""

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EqualsExpected,
    Evaluator,
    EvaluatorContext,
    IsInstance,
    LLMJudge,
)
from rich.console import Console

from ..config.models import AssertionConfig, PromptDevConfig

console = Console()


@dataclass
class JSONSchemaValidator(Evaluator[str, Any]):
    """Evaluator that validates JSON output against a schema using pydantic_evals."""

    schema: dict[str, Any]

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Evaluate JSON output against schema.

        Args:
            ctx: Evaluation context with output and expected values

        Returns:
            1.0 if valid JSON matching schema, 0.0 otherwise
        """
        try:
            # Extract JSON from output using regex (compatible with promptfoo)
            output_str = str(ctx.output)
            json_match = re.search(r"\{.*\}", output_str, re.DOTALL)

            if not json_match:
                return 0.0

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Basic schema validation
            required_fields = self.schema.get("required", [])
            properties = self.schema.get("properties", {})

            # Check required fields
            for field in required_fields:
                if field not in data:
                    return 0.0

            # Check field types (basic validation)
            for field, field_schema in properties.items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if (
                        expected_type == "string"
                        and not isinstance(data[field], str)
                        or expected_type == "boolean"
                        and not isinstance(data[field], bool)
                        or expected_type == "number"
                        and not isinstance(data[field], int | float)
                    ):
                        return 0.0

            return 1.0

        except (json.JSONDecodeError, Exception):
            return 0.0


@dataclass
class PythonAssertionEvaluator(Evaluator[str, Any]):
    """Evaluator that runs custom Python assertion functions using pydantic_evals."""

    assertion_file: str
    assert_function: Any | None = None
    last_detailed_results: list[dict[str, Any]] | None = None
    last_failure_reason: str | None = None

    def __post_init__(self):
        """Load assertion function after initialization."""
        self._load_assertion_function()

    def _load_assertion_function(self):
        """Load get_assert function from Python file."""
        assert_path = Path(self.assertion_file)
        if not assert_path.exists():
            # Show more context about where we're looking and what paths might be expected
            current_dir = Path.cwd()
            relative_path = (
                assert_path.relative_to(current_dir)
                if current_dir in assert_path.parents
                else assert_path
            )
            raise FileNotFoundError(
                f"Assertion file not found: {assert_path}\n"
                f"Current directory: {current_dir}\n"
                f"Looking for: {relative_path}\n"
                f"Absolute path: {assert_path.absolute()}\n"
                f"Parent directory exists: {assert_path.parent.exists()}"
            )

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("custom_assert", assert_path)
            if spec is None:
                raise ImportError(f"Could not create module spec for {assert_path}")

            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Module spec has no loader for {assert_path}")

            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Failed to load assertion module from {assert_path}: {e}") from e

        if not hasattr(module, "get_assert"):
            available_functions = [name for name in dir(module) if not name.startswith("_")]
            raise ValueError(
                f"Assertion file must define 'get_assert' function: {assert_path}\n"
                f"Available functions in module: {available_functions}"
            )

        self.assert_function = module.get_assert()

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Evaluate using custom Python function.

        Args:
            ctx: Evaluation context with output and expected values

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.assert_function:
            return 0.0

        # Prepare context for assertion function (promptfoo compatible)
        context = {
            "vars": ctx.inputs if hasattr(ctx, "inputs") else {},
            "expected": ctx.expected_output if hasattr(ctx, "expected_output") else {},
        }

        try:
            result = self.assert_function(str(ctx.output), context)

            # Handle different return types
            if isinstance(result, dict):
                # Promptfoo format: {'pass': bool, 'score': float, 'reason': str, 'details': list}
                score = result.get("score", 0.0)

                # Store detailed results for later access
                if "details" in result:
                    self.last_detailed_results = result["details"]
                else:
                    self.last_detailed_results = None

                # Store failure reason for later access
                if "reason" in result:
                    self.last_failure_reason = result["reason"]
                else:
                    self.last_failure_reason = None

                return float(score)
            elif isinstance(result, bool):
                # Clear stored data for non-dict results
                self.last_detailed_results = None
                self.last_failure_reason = "Boolean assertion failed" if not result else None
                return 1.0 if result else 0.0
            elif isinstance(result, int | float):
                # Clear stored data for non-dict results
                self.last_detailed_results = None
                self.last_failure_reason = (
                    "Numeric assertion failed" if float(result) < 1.0 else None
                )
                return float(result)
            else:
                # Clear stored data and set generic failure reason
                self.last_detailed_results = None
                self.last_failure_reason = "Invalid assertion return type"
                return 0.0

        except Exception as e:
            # Don't print error here - let it be collected for summary reporting
            self.last_detailed_results = None
            self.last_failure_reason = f"Assertion execution error: {str(e)}"
            return 0.0


@dataclass
class ContainsEvaluator(Evaluator[str, Any]):
    """Evaluator that checks if output contains a substring using pydantic_evals."""

    expected_substring: str

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Check if output contains expected substring.

        Args:
            ctx: Evaluation context with output

        Returns:
            1.0 if contains substring, 0.0 otherwise
        """
        return 1.0 if self.expected_substring in str(ctx.output) else 0.0


@dataclass
class ContainsJSONEvaluator(Evaluator[str, Any]):
    """Promptfoo-compatible contains-json evaluator that validates JSON content against schema."""

    schema: dict[str, Any]
    last_failure_reason: str | None = None

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Check if output contains valid JSON matching the schema.

        Args:
            ctx: Evaluation context with output

        Returns:
            1.0 if contains valid JSON matching schema, 0.0 otherwise
        """
        try:
            output_str = str(ctx.output)

            # Look for JSON in markdown code blocks first
            json_match = re.search(r"```json\s*\n(.*?)\n```", output_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Look for any JSON object
                json_match = re.search(r"\{.*\}", output_str, re.DOTALL)
                if not json_match:
                    return 0.0
                json_str = json_match.group(0)

            # Parse JSON
            data = json.loads(json_str)

            # Validate against schema (basic validation)
            required_fields = self.schema.get("required", [])

            # Check required fields exist and are not null
            for field in required_fields:
                if field not in data:
                    self.last_failure_reason = f"Required field '{field}' is missing"
                    return 0.0
                elif data[field] is None:
                    self.last_failure_reason = f"Required field '{field}' cannot be null"
                    return 0.0

            # Check field types
            if "properties" in self.schema:
                for field, field_schema in self.schema["properties"].items():
                    if field in data and data[field] is not None:
                        expected_type = field_schema.get("type")
                        if expected_type == "string" and not isinstance(data[field], str):
                            self.last_failure_reason = f"Field '{field}' must be a string, got {type(data[field]).__name__}"
                            return 0.0
                        elif expected_type == "boolean" and not isinstance(data[field], bool):
                            self.last_failure_reason = f"Field '{field}' must be a boolean, got {type(data[field]).__name__}"
                            return 0.0
                        elif expected_type == "number" and not isinstance(data[field], int | float):
                            self.last_failure_reason = f"Field '{field}' must be a number, got {type(data[field]).__name__}"
                            return 0.0
                        elif expected_type == "integer" and not isinstance(data[field], int):
                            self.last_failure_reason = f"Field '{field}' must be an integer, got {type(data[field]).__name__}"
                            return 0.0
                        elif expected_type == "array" and not isinstance(data[field], list):
                            self.last_failure_reason = f"Field '{field}' must be an array, got {type(data[field]).__name__}"
                            return 0.0
                        elif expected_type == "object" and not isinstance(data[field], dict):
                            self.last_failure_reason = f"Field '{field}' must be an object, got {type(data[field]).__name__}"
                            return 0.0

            self.last_failure_reason = None  # Success case
            return 1.0

        except json.JSONDecodeError as e:
            self.last_failure_reason = f"Invalid JSON: {str(e)}"
            return 0.0
        except Exception as e:
            self.last_failure_reason = f"Schema validation error: {str(e)}"
            return 0.0


@dataclass
class LLMRubricEvaluator(Evaluator[str, Any]):
    """Promptfoo-compatible llm-rubric evaluator using LLMJudge."""

    rubric: str
    model: str | None = None

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Evaluate using LLM rubric (delegated to LLMJudge).

        Args:
            ctx: Evaluation context with output

        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Create LLMJudge with the rubric
            judge_kwargs = {"rubric": self.rubric}
            if self.model:
                judge_kwargs["model"] = self.model

            judge = LLMJudge(**judge_kwargs)
            return judge.evaluate(ctx)
        except Exception as e:
            # Log the error and return 0.0 to fail the test
            console.print(f"[red]LLMRubricEvaluator failed: {e}[/red]")
            return 0.0


@dataclass
class GEvalEvaluator(Evaluator[str, Any]):
    """Promptfoo-compatible g-eval evaluator using LLMJudge with G-Eval methodology."""

    criteria: str
    model: str | None = None

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Evaluate using G-Eval methodology via LLMJudge.

        Args:
            ctx: Evaluation context with output

        Returns:
            Score between 0.0 and 1.0
        """
        # Create a G-Eval style rubric
        g_eval_rubric = f"""
You are an expert evaluator. Please evaluate the following output based on this criteria: {self.criteria}

Rate the output on a scale from 1 to 5, where:
1 = Very Poor
2 = Poor
3 = Fair
4 = Good
5 = Excellent

Consider the criteria carefully and provide your assessment.
"""

        try:
            judge_kwargs = {"rubric": g_eval_rubric}
            if self.model:
                judge_kwargs["model"] = self.model

            judge = LLMJudge(**judge_kwargs)
            score = judge.evaluate(ctx)

            # G-Eval typically returns scores 1-5, normalize to 0-1
            return score / 5.0 if score > 1.0 else score
        except Exception as e:
            # Log the error and return 0.0 to fail the test
            console.print(f"[red]GEvalEvaluator failed: {e}[/red]")
            return 0.0


@dataclass
class FailureEvaluator(Evaluator[str, Any]):
    """Evaluator that always fails - used when evaluator creation fails."""

    error_message: str

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> float:
        """Always return 0.0 to indicate failure.

        Args:
            ctx: Evaluation context

        Returns:
            Always 0.0 (failure)
        """
        return 0.0


def create_pydantic_evaluator(
    assertion_config: AssertionConfig | dict, config: PromptDevConfig, verbose: bool = False
) -> Evaluator:
    """Factory function to create pydantic_evals evaluators.

    Args:
        assertion_config: Assertion configuration (AssertionConfig object or dict)
        config: Full PromptDev configuration (for schema resolution)
        verbose: Enable verbose output

    Returns:
        Appropriate pydantic_evals evaluator instance
    """
    # Convert dict to AssertionConfig if needed
    if isinstance(assertion_config, dict):
        assertion_config = AssertionConfig(**assertion_config)

    evaluator_type = assertion_config.type
    evaluator_value = assertion_config.value

    # Handle promptfoo-style $ref-only assertions
    if not evaluator_type and assertion_config.ref:
        # We'll resolve the type from the template later
        evaluator_type = None

    # Resolve template references and schema $refs
    if assertion_config.ref and config.assertion_templates:
        # Remove the prefix properly (not using lstrip which removes individual characters)
        # Support both camelCase and snake_case patterns
        prefixes = ["#/assertionTemplates/", "#/assertion_templates/"]
        ref_name = assertion_config.ref
        for prefix in prefixes:
            if assertion_config.ref.startswith(prefix):
                ref_name = assertion_config.ref[len(prefix) :]
                break

        if ref_name in config.assertion_templates:
            template = config.assertion_templates[ref_name]

            # Template is an AssertionConfig object, not a dict
            template_type = template.type if hasattr(template, "type") else None
            template_value = template.value if hasattr(template, "value") else None
            template_threshold = template.threshold if hasattr(template, "threshold") else None
            template_rubric = template.rubric if hasattr(template, "rubric") else None
            template_model = template.model if hasattr(template, "model") else None

            # Merge template with assertion config (assertion config takes precedence)
            evaluator_type = assertion_config.type or template_type
            evaluator_value = (
                assertion_config.value if assertion_config.value is not None else template_value
            )

            # Create a temporary AssertionConfig with resolved values for the rest of the function
            resolved_assertion = AssertionConfig(
                type=evaluator_type,
                value=evaluator_value,
                threshold=assertion_config.threshold or template_threshold,
                rubric=assertion_config.rubric or template_rubric,
                model=assertion_config.model or template_model,
                ref=assertion_config.ref,
            )
            assertion_config = resolved_assertion

            # Update local variables to use resolved values
            evaluator_type = resolved_assertion.type
            evaluator_value = resolved_assertion.value

            # TODO: maybe we can resolve this easier using yaml reader?
            # Resolve schema references within template value
            if isinstance(evaluator_value, dict) and "$ref" in evaluator_value:
                schema_ref = evaluator_value["$ref"]

                # Handle different schema reference formats
                if schema_ref and schema_ref.startswith("#/schemas/"):
                    # Standard schema reference format
                    schema_name = schema_ref[len("#/schemas/") :]
                    if config.schemas and schema_name in config.schemas:
                        evaluator_value = config.schemas[schema_name]
                elif schema_ref and schema_ref.startswith("#/"):
                    # Direct root-level schema reference (promptfoo style)
                    schema_name = schema_ref[len("#/") :]
                    # Check if the schema exists as a top-level attribute in config
                    if hasattr(config, schema_name):
                        evaluator_value = getattr(config, schema_name)
                    # Also check in the raw config dict if available
                    elif hasattr(config, "__dict__") and schema_name in config.__dict__:
                        evaluator_value = config.__dict__[schema_name]

                # Update the assertion_config with the resolved schema value
                if evaluator_value != assertion_config.value:
                    assertion_config = AssertionConfig(
                        type=assertion_config.type,
                        value=evaluator_value,
                        threshold=assertion_config.threshold,
                        rubric=assertion_config.rubric,
                        model=assertion_config.model,
                        ref=assertion_config.ref,
                    )

    # Create appropriate pydantic_evals evaluator
    if evaluator_type == "json_schema":
        if isinstance(evaluator_value, dict):
            return JSONSchemaValidator(schema=evaluator_value)
        else:
            raise ValueError(
                f"JSON schema evaluator requires dict value, got: {type(evaluator_value)}"
            )

    elif evaluator_type == "contains-json":
        # Promptfoo's contains-json evaluator
        if isinstance(evaluator_value, dict):
            return ContainsJSONEvaluator(schema=evaluator_value)
        elif isinstance(evaluator_value, str):
            # Handle file reference to schema
            try:
                from pathlib import Path

                schema_path = Path(evaluator_value.replace("file://", ""))
                if schema_path.exists():
                    with open(schema_path) as f:
                        schema = json.load(f)
                    return ContainsJSONEvaluator(schema=schema)
                else:
                    raise FileNotFoundError(f"Schema file not found: {schema_path}")
            except Exception as e:
                raise ValueError(f"Failed to load schema from file: {e}") from e
        else:
            raise ValueError(
                f"Contains-json evaluator requires dict or file path, got: {type(evaluator_value)}"
            )

    elif evaluator_type == "python":
        if isinstance(evaluator_value, str):
            # Handle file:// URLs and resolve relative paths
            assertion_file = evaluator_value
            if assertion_file.startswith("file://"):
                assertion_file = assertion_file[7:]  # Remove file:// prefix

            # If it's a relative path, make it absolute relative to the examples directory
            if not assertion_file.startswith("/"):
                # Get the absolute path relative to the current working directory
                from pathlib import Path

                assertion_file = str(Path(assertion_file).resolve())

            return PythonAssertionEvaluator(assertion_file=assertion_file)
        else:
            raise ValueError(
                f"Python evaluator requires file path string, got: {type(evaluator_value)}"
            )

    elif evaluator_type == "exact":
        return EqualsExpected()

    elif evaluator_type == "contains":
        if isinstance(evaluator_value, str):
            return ContainsEvaluator(expected_substring=evaluator_value)
        else:
            raise ValueError(
                f"Contains evaluator requires string value, got: {type(evaluator_value)}"
            )

    elif evaluator_type == "is_instance":
        if isinstance(evaluator_value, str):
            return IsInstance(type_name=evaluator_value)
        else:
            raise ValueError(
                f"IsInstance evaluator requires type name string, got: {type(evaluator_value)}"
            )

    elif evaluator_type == "llm_judge":
        # Use LLMJudge for advanced semantic evaluation
        rubric = assertion_config.rubric or evaluator_value
        if isinstance(rubric, str):
            judge_kwargs = {"rubric": rubric}
            # Add model if specified
            if assertion_config.model:
                judge_kwargs["model"] = assertion_config.model
            return LLMJudge(**judge_kwargs)
        else:
            default_rubric = "Evaluate if the output is accurate and helpful"
            return LLMJudge(rubric=default_rubric)

    elif evaluator_type == "llm-rubric":
        # Promptfoo's llm-rubric evaluator
        rubric = evaluator_value if isinstance(evaluator_value, str) else "Evaluate output quality"
        return LLMRubricEvaluator(rubric=rubric, model=assertion_config.model)

    elif evaluator_type == "g-eval":
        # Promptfoo's g-eval evaluator
        criteria = (
            evaluator_value if isinstance(evaluator_value, str) else "Evaluate overall quality"
        )
        return GEvalEvaluator(criteria=criteria, model=assertion_config.model)

    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


class PromptDevDataset:
    """Wrapper around pydantic_evals Dataset for PromptDev compatibility."""

    def __init__(
        self,
        test_cases: list[dict[str, Any]],
        config: PromptDevConfig,
        error_collector: list = None,
        verbose: bool = False,
    ):
        """Initialize PromptDev dataset.

        Args:
            test_cases: List of test case dictionaries
            config: PromptDev configuration
            error_collector: List to collect errors for summary reporting
        """
        self.config = config
        self.test_cases = test_cases
        self._error_collector = error_collector

        # Convert test cases to pydantic_evals Cases
        self.cases = []
        self.evaluators = []
        self.evaluator_assertion_map = {}  # Maps evaluator index to assertion template name
        self.evaluator_type_map = {}  # Maps evaluator index to assertion type

        for i, test_case in enumerate(test_cases):
            case_name = test_case.get("name", f"test_case_{i}")
            inputs = test_case.get("vars", {})
            expected_output = test_case.get("expected")
            metadata = test_case.get("metadata", {})

            # Create Case
            case = Case(
                name=case_name, inputs=inputs, expected_output=expected_output, metadata=metadata
            )
            self.cases.append(case)

            # Add case-specific evaluators
            assertions = test_case.get("assertions", [])
            for assertion in assertions:
                try:
                    evaluator = create_pydantic_evaluator(assertion, config, verbose=False)
                    evaluator_index = len(self.evaluators)
                    self.evaluators.append(evaluator)

                    # Track original assertion name and type
                    assertion_name = self._get_assertion_name(assertion)
                    assertion_type = self._get_assertion_type(assertion, config)
                    self.evaluator_assertion_map[evaluator_index] = assertion_name
                    self.evaluator_type_map[evaluator_index] = assertion_type
                except Exception as e:
                    import traceback

                    error_msg = f"Failed to create evaluator for {assertion}: {e}"
                    # Store error for later reporting instead of printing immediately
                    if hasattr(self, "_error_collector") and self._error_collector is not None:
                        from ..evaluation.results import EvaluationError

                        error = EvaluationError(
                            error_type="evaluator_creation",
                            message=error_msg,
                            details=traceback.format_exc(),
                            context={
                                "test_case": test_case,
                                "assertion_config": assertion.model_dump()
                                if hasattr(assertion, "model_dump")
                                else str(assertion),
                            },
                        )
                        self._error_collector.append(error)
                    # Add a failure evaluator to ensure test fails
                    self.evaluators.append(FailureEvaluator(error_message=error_msg))

        # Add default test evaluators if configured and not already included in test cases
        # Don't add default test assertions if any test case has non-empty assertions
        # (indicating they've been pre-processed and merged)
        should_add_defaults = True
        if config.default_test and config.default_test.assert_:
            for case in test_cases:
                case_assertions = case.get("assertions", [])
                if (
                    case_assertions
                ):  # If any test case has assertions, defaults are already included
                    should_add_defaults = False
                    if verbose:
                        console.print(
                            f"[blue]Skipping default test assertions - found {len(case_assertions)} case assertions[/blue]"
                        )
                    break

        if config.default_test and config.default_test.assert_ and should_add_defaults:
            for assertion in config.default_test.assert_:
                try:
                    evaluator = create_pydantic_evaluator(assertion, config, verbose=True)
                    evaluator_index = len(self.evaluators)
                    self.evaluators.append(evaluator)

                    # Track original assertion name and type
                    assertion_name = self._get_assertion_name(assertion)
                    assertion_type = self._get_assertion_type(assertion, config)
                    self.evaluator_assertion_map[evaluator_index] = assertion_name
                    self.evaluator_type_map[evaluator_index] = assertion_type
                except Exception as e:
                    import traceback

                    error_msg = f"Failed to create default evaluator for {assertion}: {e}"
                    # Store error for later reporting instead of printing immediately
                    if hasattr(self, "_error_collector") and self._error_collector is not None:
                        from ..evaluation.results import EvaluationError

                        error = EvaluationError(
                            error_type="evaluator_creation",
                            message=error_msg,
                            details=traceback.format_exc(),
                            context={
                                "assertion_config": assertion.model_dump()
                                if hasattr(assertion, "model_dump")
                                else str(assertion),
                                "available_templates": list(config.assertion_templates.keys())
                                if hasattr(config, "assertion_templates")
                                and config.assertion_templates
                                else None,
                            },
                        )
                        self._error_collector.append(error)
                    # Add a failure evaluator to ensure test fails
                    self.evaluators.append(FailureEvaluator(error_message=error_msg))

        # Create pydantic_evals Dataset
        self.dataset = Dataset(cases=self.cases, evaluators=self.evaluators)

    def _get_assertion_name(self, assertion) -> str:
        """Extract the assertion template name from assertion config.

        Args:
            assertion: Assertion configuration object

        Returns:
            The assertion template name or type
        """
        if hasattr(assertion, "model_dump"):
            assertion_dict = assertion.model_dump()
        elif isinstance(assertion, dict):
            assertion_dict = assertion
        else:
            return str(assertion)

        # Check for template reference (e.g., $ref: '#/assertionTemplates/assertExpected')
        if "ref" in assertion_dict:
            ref = assertion_dict["ref"]
            if ref and ref.startswith("#/assertionTemplates/"):
                return ref[len("#/assertionTemplates/") :]

        # Check for metric name
        if "metric" in assertion_dict:
            return assertion_dict["metric"]

        # Check for type
        if "type" in assertion_dict:
            return assertion_dict["type"]

        return "unknown_assertion"

    def _get_assertion_type(self, assertion, config) -> str:
        """Extract the assertion type from assertion config.

        Args:
            assertion: Assertion configuration object
            config: PromptDev configuration

        Returns:
            The assertion type from YAML
        """
        if hasattr(assertion, "model_dump"):
            assertion_dict = assertion.model_dump()
        elif isinstance(assertion, dict):
            assertion_dict = assertion
        else:
            return "unknown"

        # Check for direct type in assertion
        if "type" in assertion_dict and assertion_dict["type"]:
            return assertion_dict["type"]

        # Check for template reference and resolve to get type
        if "ref" in assertion_dict:
            ref = assertion_dict["ref"]
            if ref.startswith("#/assertionTemplates/"):
                template_name = ref[len("#/assertionTemplates/") :]
                if (
                    hasattr(config, "assertion_templates")
                    and config.assertion_templates
                    and template_name in config.assertion_templates
                ):
                    template = config.assertion_templates[template_name]
                    if hasattr(template, "model_dump"):
                        template_dict = template.model_dump()
                    elif isinstance(template, dict):
                        template_dict = template
                    else:
                        return "unknown"

                    if "type" in template_dict:
                        return template_dict["type"]

        return "unknown"

    async def evaluate_async(self, task_function, **kwargs) -> Any:
        """Run evaluation using pydantic_evals.

        Args:
            task_function: Function to evaluate
            **kwargs: Additional arguments to pass to dataset.evaluate

        Returns:
            Evaluation report
        """
        return await self.dataset.evaluate(task_function, **kwargs)

    def evaluate_sync(self, task_function) -> Any:
        """Run evaluation synchronously using pydantic_evals.

        Args:
            task_function: Function to evaluate

        Returns:
            Evaluation report
        """
        return self.dataset.evaluate_sync(task_function)


async def run_pydantic_evaluation(
    agent_function,
    test_cases: list[dict[str, Any]],
    config: PromptDevConfig,
    verbose: bool = False,
    progress: bool = True,
    error_collector: list = None,
) -> Any:
    """Run evaluation using PydanticAI's pydantic_evals system.

    Args:
        agent_function: Function that takes inputs and returns outputs
        test_cases: List of test case dictionaries
        config: PromptDev configuration
        verbose: Enable verbose output
        progress: Enable progress bar (default: True)
        error_collector: List to collect errors for summary reporting

    Returns:
        Evaluation report from pydantic_evals
    """
    # Create PromptDev dataset
    dataset = PromptDevDataset(test_cases, config, error_collector, verbose)

    if verbose:
        console.print(
            f"[green]Created dataset with {len(dataset.cases)} cases and {len(dataset.evaluators)} evaluators[/green]"
        )

    # Run evaluation
    report = await dataset.evaluate_async(agent_function, progress=progress)

    if verbose:
        console.print("[green]Evaluation completed[/green]")
        report.print(include_input=True, include_output=True)

    return report, dataset
