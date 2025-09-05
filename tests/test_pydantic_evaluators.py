"""Tests for PydanticAI-based evaluators."""

import pytest

from promptdev.config.models import AssertionConfig, PromptDevConfig
from promptdev.evaluators.pydantic_evaluators import (
    ContainsJSONEvaluator,
    GEvalEvaluator,
    JSONSchemaValidator,
    LLMRubricEvaluator,
    PromptDevDataset,
    create_pydantic_evaluator,
)


class TestJSONSchemaValidator:
    """Test JSON schema validation using pydantic_evals."""

    def test_valid_json_schema(self):
        """Test valid JSON schema validation."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }

        evaluator = JSONSchemaValidator(schema=schema)

        # Mock evaluation context
        class MockContext:
            def __init__(self, output):
                self.output = output

        # Valid JSON should pass
        valid_output = '{"name": "John", "age": 30}'
        ctx = MockContext(valid_output)
        score = evaluator.evaluate(ctx)
        assert score == 1.0

        # Invalid JSON should fail
        invalid_output = '{"name": "John"}'  # missing age
        ctx = MockContext(invalid_output)
        score = evaluator.evaluate(ctx)
        assert score == 0.0


class TestCreatePydanticEvaluator:
    """Test evaluator factory function."""

    def test_create_json_schema_evaluator(self):
        """Test creating JSON schema evaluator."""
        assertion_config = AssertionConfig(
            type="json_schema",
            value={
                "type": "object",
                "properties": {"test": {"type": "string"}},
                "required": ["test"],
            },
        )
        config = PromptDevConfig(prompts=[], providers=[], tests=[])

        evaluator = create_pydantic_evaluator(assertion_config, config)
        assert isinstance(evaluator, JSONSchemaValidator)

    def test_schema_reference_resolution(self):
        """Test that schema references are properly resolved."""
        assertion_config = AssertionConfig(
            type="json_schema", ref="#/assertion_templates/testTemplate"
        )

        config = PromptDevConfig(
            prompts=[],
            providers=[],
            tests=[],
            assertion_templates={
                "testTemplate": AssertionConfig(
                    type="json_schema", value={"$ref": "#/schemas/testSchema"}
                )
            },
            schemas={
                "testSchema": {
                    "type": "object",
                    "properties": {"resolved": {"type": "boolean"}},
                    "required": ["resolved"],
                }
            },
        )

        evaluator = create_pydantic_evaluator(assertion_config, config)
        assert isinstance(evaluator, JSONSchemaValidator)
        assert evaluator.schema["properties"]["resolved"]["type"] == "boolean"


class TestPromptDevDataset:
    """Test PromptDev dataset wrapper."""

    def test_create_dataset(self):
        """Test creating a dataset with test cases."""
        test_cases = [
            {
                "name": "test_case_1",
                "vars": {"input": "test"},
                "expected": {"output": "result"},
                "assertions": [AssertionConfig(type="exact", value="result")],
            }
        ]

        config = PromptDevConfig(prompts=[], providers=[], tests=[])
        dataset = PromptDevDataset(test_cases, config)

        assert len(dataset.cases) == 1
        assert dataset.cases[0].name == "test_case_1"
        assert dataset.cases[0].inputs == {"input": "test"}
        assert dataset.cases[0].expected_output == {"output": "result"}


class TestPromptfooCompatibility:
    """Test compatibility with promptfoo evaluator types."""

    def test_contains_json_evaluator(self):
        """Test promptfoo contains-json evaluator."""
        assertion_config = AssertionConfig(
            type="contains-json",
            value={
                "type": "object",
                "properties": {"test": {"type": "string"}},
                "required": ["test"],
            },
        )
        config = PromptDevConfig(prompts=[], providers=[], tests=[])

        evaluator = create_pydantic_evaluator(assertion_config, config)
        assert isinstance(evaluator, ContainsJSONEvaluator)

    def test_llm_rubric_evaluator(self):
        """Test promptfoo llm-rubric evaluator."""
        assertion_config = AssertionConfig(
            type="llm-rubric", value="Evaluate if the output is helpful and accurate"
        )
        config = PromptDevConfig(prompts=[], providers=[], tests=[])

        evaluator = create_pydantic_evaluator(assertion_config, config)
        assert isinstance(evaluator, LLMRubricEvaluator)
        assert evaluator.rubric == "Evaluate if the output is helpful and accurate"

    def test_g_eval_evaluator(self):
        """Test promptfoo g-eval evaluator."""
        assertion_config = AssertionConfig(
            type="g-eval", value="Ensure the response is factually accurate"
        )
        config = PromptDevConfig(prompts=[], providers=[], tests=[])

        evaluator = create_pydantic_evaluator(assertion_config, config)
        assert isinstance(evaluator, GEvalEvaluator)
        assert evaluator.criteria == "Ensure the response is factually accurate"

    def test_direct_schema_reference_resolution(self):
        """Test resolving direct schema references (promptfoo style)."""
        # This simulates the pattern: $ref: '#/calendarEventSummarySchema'
        assertion_config = AssertionConfig(
            type="contains-json", ref="#/assertionTemplates/testTemplate"
        )

        # Create a config with extra fields (promptfoo style)
        config_dict = {
            "prompts": [],
            "providers": [],
            "tests": [],
            "assertionTemplates": {
                "testTemplate": {
                    "type": "contains-json",
                    "value": {"$ref": "#/calendarEventSummarySchema"},
                }
            },
            # Direct schema definition (promptfoo style)
            "calendarEventSummarySchema": {
                "type": "object",
                "properties": {"resolved": {"type": "boolean"}},
                "required": ["resolved"],
            },
        }

        config = PromptDevConfig(**config_dict)

        # The extra field should be accessible
        assert hasattr(config, "calendarEventSummarySchema")

        evaluator = create_pydantic_evaluator(assertion_config, config)
        assert isinstance(evaluator, ContainsJSONEvaluator)


if __name__ == "__main__":
    pytest.main([__file__])
