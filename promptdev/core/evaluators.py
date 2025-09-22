import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import jsonschema
from pydantic_core import from_json
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, EvaluatorOutput
from pydantic_evals.evaluators.common import LLMJudge


@dataclass
class IsJSON(Evaluator[str, Any]):
    """Evaluator that checks if a string is valid JSON and optionally validates against a schema."""

    schema: dict[str, Any] | None = None
    evaluation_name: str = "is_json"

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> EvaluationReason:
        """Validate JSON schema.

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
                return EvaluationReason(value=0.0, reason="No JSON found in output")

            json_str = json_match.group(0)
            data = from_json(json_str.encode())
            if self.schema:
                jsonschema.validate(instance=data, schema=self.schema)
            return EvaluationReason(value=1.0, reason="Validation successful")
        except ValueError as e:
            return EvaluationReason(value=0.0, reason=f"Invalid JSON: {e!s}")
        except jsonschema.exceptions.SchemaError as e:
            return EvaluationReason(value=0.0, reason=f"Schema error: {e!s}")
        except jsonschema.exceptions.ValidationError as e:
            return EvaluationReason(value=0.0, reason=f"Validation error: {e!s}")
        except Exception as e:
            return EvaluationReason(value=0.0, reason=f"Invalid JSON: {e!s}")


@dataclass
class PythonAssertion(Evaluator[str, Any]):
    """Evaluator that runs custom Python assertion functions using pydantic_evals.
    Warning: This evaluator is potentially dangerous as it allows the execution of arbitrary Python code.
    Only use it if you trust the source code of the assertion file.
    """

    class FunctionContext(TypedDict):
        vars: dict[str, Any]

    FunctionResponseDict = TypedDict(
        "FunctionResponseDict",
        {
            "pass": bool | None,
            "score": float | None,
            "reason": str | None,
        },
    )

    assert_path: Path
    evaluation_name: str = "python"
    assert_function: (
        Callable[[str, FunctionContext], bool | float | FunctionResponseDict] | None
    ) = None

    def __post_init__(self):
        """Load assertion function after initialization."""
        self._load_assertion_function()

    def _load_assertion_function(self):
        """Load get_assert function from Python file."""
        if not isinstance(self.assert_path, Path):
            raise TypeError(f"Assertion path must be a Path object, got: {type(self.assert_path)}")

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("custom_assert", self.assert_path)
            if spec is None:
                raise ImportError(f"Could not create module spec for {self.assert_path}")

            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Module spec has no loader for {self.assert_path}")

            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(
                f"Failed to load assertion module from {self.assert_path}: {e}"
            ) from e

        if not hasattr(module, "get_assert"):
            available_functions = [name for name in dir(module) if not name.startswith("_")]
            raise ValueError(
                f"Assertion file must define 'get_assert' function: {self.assert_path}\n"
                f"Available functions in module: {available_functions}"
            )

        self.assert_function = module.get_assert

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> EvaluatorOutput:
        """Evaluate using custom Python function.

        Args:
            ctx: Evaluation context with output and expected values

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.assert_function:
            return EvaluationReason(value=0.0, reason="No assertion function loaded")

        # Prepare context for assertion function (promptfoo compatible)
        context = PythonAssertion.FunctionContext(
            vars=ctx.inputs if hasattr(ctx, "inputs") else {},
        )

        try:
            result = self.assert_function(str(ctx.output), context)
            # Handle different return types according to the signature:
            # get_assert(output: str, context: dict) -> bool | float | dict[str, Any]
            if isinstance(result, bool):
                return EvaluationReason(value=result, reason=f"Boolean result: {result}")
            if isinstance(result, float):
                return EvaluationReason(value=result, reason=f"Numeric score: {result}")
            if isinstance(result, dict):
                # Dictionary format - extract score and reason
                pass_ = result.get("pass")
                if pass_ is not None:
                    return EvaluationReason(value=pass_, reason=result.get("reason"))
                return EvaluationReason(value=result.get("score", 0.0), reason=result.get("reason"))

            return EvaluationReason(
                value=False, reason=f"Invalid assertion return type: {type(result)}"
            )

        except Exception as e:
            return EvaluationReason(value=False, reason=f"Assertion execution error: {e!s}")


@dataclass
class ContainsJSON(Evaluator[str, Any]):
    """Evaluator that checks if the output contains a valid JSON and optionally validates against a schema."""

    schema: dict[str, Any] | None = None
    evaluation_name: str = "contains_json"

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> EvaluationReason:
        """Validate that output contains valid JSON matching the schema.

        Args:
            ctx: Evaluation context with output

        Returns:
            1.0 if contains valid JSON matching schema, 0.0 otherwise
        """
        try:
            output_str = str(ctx.output)

            # Look for JSON in Markdown code blocks first
            json_match = re.search(r"```json\s*\n(.*?)\n```", output_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Look for any JSON object
                json_match = re.search(r"\{.*\}", output_str, re.DOTALL)
                if not json_match:
                    return EvaluationReason(value=0.0, reason="No JSON found in output")
                json_str = json_match.group(0)

            # Parse JSON
            data = from_json(json_str.encode())
            if self.schema:
                jsonschema.validate(instance=data, schema=self.schema)
            return EvaluationReason(value=1.0, reason="Validation successful")

        except ValueError as e:
            return EvaluationReason(value=0.0, reason=f"Invalid JSON: {e!s}")
        except jsonschema.exceptions.SchemaError as e:
            return EvaluationReason(value=0.0, reason=f"Schema error: {e!s}")
        except jsonschema.exceptions.ValidationError as e:
            return EvaluationReason(value=0.0, reason=f"Validation error: {e!s}")


@dataclass
class GEval(Evaluator[str, Any]):
    """Promptfoo-compatible g-eval evaluator using LLM judge."""

    rubric: str
    evaluation_name: str | None = None
    provider: Any | None = None  # pydantic-ai provider
    model: str | None = None
    client: Any | None = None

    def evaluate(self, ctx: EvaluatorContext[str, Any]) -> EvaluationReason:
        """Evaluate using G-Eval methodology via LLMJudge.

        Args:
            ctx: Evaluation context with output

        Returns:
            Score between 0.0 and 1.0
        """
        # Create a G-Eval style rubric
        g_eval_rubric = f"""
You are an expert evaluator. Please evaluate the following output based on this criteria: {self.rubric}

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
            if isinstance(score, dict):
                score = next(iter(score.values()))
                score = getattr(score, "value", score)

            # G-Eval typically returns scores 1-5, normalize to 0-1
            return EvaluationReason(
                value=score / 5.0 if score > 1.0 else score, reason="GEvalEvaluator score"
            )
        except Exception as e:
            return EvaluationReason(value=0.0, reason=f"GEvalEvaluator failed: {e!s}")
