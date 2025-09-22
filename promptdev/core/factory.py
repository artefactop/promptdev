import uuid
from pathlib import Path

import yaml
from pydantic_ai import ModelSettings, models
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.common import (
    Contains,
    Equals,
    IsInstance,
    MaxDuration,
)
from rich.console import Console

from promptdev.config.schemas import (
    AssertionConfig,
    PromptDevConfig,
    ProviderConfig,
    TestConfig,
)
from promptdev.core.evaluators import (
    ContainsJSON,
    IsJSON,
    PythonAssertion,
)
from promptdev.utils.file import read_file, read_jsonl_file
from promptdev.utils.llm_models import EchoModel

console = Console()


class EvaluatorFactory:
    @staticmethod
    def create_evaluator(assertion_config: AssertionConfig) -> Evaluator:
        """Factory function to create pydantic_evals evaluators.

        Args:
            assertion_config: Assertion configuration (AssertionConfig object)

        Returns:
            Appropriate Evaluator instance
        """
        evaluator_type = assertion_config.type.replace("-", "_")
        evaluator_value = assertion_config.value

        # Create appropriate pydantic_evals evaluator
        if evaluator_type == "is_json":
            if evaluator_value is None or isinstance(evaluator_value, dict):
                return IsJSON(schema=evaluator_value, evaluation_name=evaluator_type)
            if isinstance(evaluator_value, Path):
                # Handle file reference to schema
                try:
                    schema = read_file(evaluator_value)
                    return IsJSON(schema=schema, evaluation_name=evaluator_type)
                except Exception as e:
                    raise ValueError(f"Failed to load schema from file: {e}") from e
            else:
                raise ValueError(
                    f"JSON schema evaluator requires dict or file path, got: {type(evaluator_value)}"
                )

        if evaluator_type == "contains_json":
            # Promptfoo's contains_json evaluator
            if evaluator_value is None or isinstance(evaluator_value, dict):
                return ContainsJSON(schema=evaluator_value, evaluation_name=evaluator_type)
            if isinstance(evaluator_value, Path):
                # Handle file reference to schema
                try:
                    schema = read_file(evaluator_value)
                    return ContainsJSON(schema=schema, evaluation_name=evaluator_type)
                except Exception as e:
                    raise ValueError(f"Failed to load schema from file: {e}") from e
            else:
                raise ValueError(
                    f"Contains-json evaluator requires dict or file path, got: {type(evaluator_value)}"
                )

        elif evaluator_type == "python":
            console.print(
                "Python evaluator executes arbitrary Python code. Use only if you trust the source code of the assertion file.",
                style="yellow",
            )
            if isinstance(evaluator_value, Path):
                return PythonAssertion(assert_path=evaluator_value, evaluation_name=evaluator_type)
            raise ValueError(
                f"Python evaluator requires a Path to a file, got: {type(evaluator_value)}"
            )

        elif evaluator_type in ["llm_rubric", "llm_judge"]:
            # # Use the rubric from assertion_config or evaluator_value
            # rubric = assertion_config.rubric or evaluator_value
            # if isinstance(rubric, str):
            #     judge_kwargs = {"rubric": rubric}
            #     if assertion_config.model:
            #         judge_kwargs["model"] = assertion_config.model
            #     return LLMJudge(**judge_kwargs)
            # default_rubric = "Evaluate if the output is accurate and helpful"
            # return LLMJudge(rubric=default_rubric)
            raise ValueError(f"{evaluator_value} - not yet supported")

        elif evaluator_type == "g_eval":
            raise ValueError("G-Eval evaluator - not yet supported")

        # Fallback to pydantic_evals built-in evaluators
        elif evaluator_type == "equals":
            if evaluator_value is not None:
                return Equals(value=evaluator_value)
            raise ValueError(f"Equals evaluator requires a value, got: {evaluator_value}")

        elif evaluator_type == "contains":
            if isinstance(evaluator_value, str):
                return Contains(value=evaluator_value, evaluation_name=evaluator_type)
            raise ValueError(
                f"Contains evaluator requires string value, got: {type(evaluator_value)}"
            )

        elif evaluator_type == "is_instance":
            if isinstance(evaluator_value, str):
                return IsInstance(type_name=evaluator_value.lower())
            raise ValueError(
                f"IsInstance evaluator requires type name string, got: {type(evaluator_value)}"
            )

        elif evaluator_type == "max_duration":
            if isinstance(evaluator_value, int | float):
                return MaxDuration(seconds=evaluator_value)
            raise ValueError(
                f"MaxDuration evaluator requires numeric value, got: {type(evaluator_value)}"
            )

        elif evaluator_type == "has_matching_span":
            # This evaluator requires SpanQuery parameter which is complex to handle generically
            raise ValueError("HasMatchingSpan evaluator - not yet supported")

        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")


class PromptTemplate:
    """Runtime prompt template"""

    def __init__(self, system_prompt: str, user_template: str):
        self.system_prompt = system_prompt
        self.user_template = user_template

    @classmethod
    def from_file(cls, prompt_path: Path) -> "PromptTemplate":
        """Load prompt from YAML file"""
        # TODO allow multiple prompts in the same file
        with open(prompt_path, encoding="utf-8") as f:
            messages = yaml.safe_load(f)  # TODO: Fix it, can return dict, list or None

        system_content = ""
        user_content = ""
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
            elif message["role"] == "user":
                user_content = message["content"]

        # Handle double-brace conversion
        system_content = system_content.replace("{{", "{").replace("}}", "}")
        user_content = user_content.replace("{{", "{").replace("}}", "}")

        return cls(system_content, user_content)

    @classmethod
    def from_string(cls, prompt: str) -> "PromptTemplate":
        """Create prompt template from string"""
        return cls(system_prompt="", user_template=prompt)

    def __repr__(self):
        return f"system: {self.system_prompt}, user: {self.user_template}"


class ModelFactory:
    """Builds PydanticAI models from provider config"""

    @staticmethod
    def build_model(provider_config: ProviderConfig) -> Model:
        """Create PydanticAI model from config"""
        if provider_config.id.startswith("ollama:"):
            actual_model_name = provider_config.id[7:]
            base_url = provider_config.config.get("base_url")
            provider = OllamaProvider(base_url=base_url)
            return OpenAIChatModel(model_name=actual_model_name, provider=provider)
        if provider_config.id == "promptdev:echo":
            return EchoModel()
        return models.infer_model(provider_config.id)

    @staticmethod
    def build_model_settings(provider_config: ProviderConfig) -> ModelSettings:
        """Create model settings from config"""
        return ModelSettings(**provider_config.config)


class DatasetFactory:
    """Builds pydantic_evals Dataset from config"""

    @staticmethod
    def build_dataset(config: PromptDevConfig) -> Dataset:
        """Build dataset from configuration"""

        cases = []
        common_evaluators = []

        # Handle default test evaluators
        if config.default_test:
            for assertion_config in config.default_test.assert_:
                evaluator = EvaluatorFactory.create_evaluator(assertion_config)
                common_evaluators.append(evaluator)

        # Process test configurations
        if config.tests:
            tests = config.tests
            if isinstance(tests, Path):
                cases.extend(DatasetFactory._build_cases_from_dataset_config(tests))
            else:
                for test_config in tests:
                    if isinstance(test_config, TestConfig):
                        cases.append(DatasetFactory._build_cases_from_test_config(test_config))
                    elif isinstance(test_config, Path):
                        cases.extend(DatasetFactory._build_cases_from_dataset_config(test_config))
                    else:
                        raise TypeError(f"Unsupported test config entry: {type(test_config)}")
        if not cases:
            raise ValueError("No test cases specified")

        return Dataset(cases=cases, evaluators=common_evaluators)

    @staticmethod
    def _build_cases_from_test_config(test_config: TestConfig) -> Case:
        """Build cases from TestConfig"""

        evaluators = []
        for assertion_config in test_config.assert_:
            if isinstance(assertion_config, Path):
                loaded = read_file(assertion_config)
                if isinstance(loaded, dict):
                    assertion_objs = [AssertionConfig(**loaded)]
                elif isinstance(loaded, list):
                    assertion_objs = [AssertionConfig(**item) for item in loaded]
                else:
                    raise TypeError(f"Unsupported assertion file format: {type(loaded)}")
            else:
                assertion_objs = [assertion_config]

            for ac in assertion_objs:
                evaluator = EvaluatorFactory.create_evaluator(ac)
                evaluators.append(evaluator)

        return Case(
            name=test_config.description or f"test_{uuid.uuid4().hex[:8]}",
            inputs=test_config.vars or {},
            metadata=test_config.metadata,
            expected_output=None,
            evaluators=tuple(evaluators),
        )

    @staticmethod
    def _build_cases_from_dataset_config(dataset_config: Path) -> list[Case]:
        """Create a pydantic Dataset instance from a Path."""
        file_path = Path(dataset_config)
        if file_path.suffix == ".jsonl":
            # Load from JSONL file
            return DatasetFactory._load_from_jsonl(file_path)
            # TODO support json and yaml formats too
        raise NotImplementedError("Only JSONL files are supported for now")

    @staticmethod
    def _load_from_jsonl(file_path: Path) -> list[Case]:
        """Load dataset from JSONL file (compatible with promptfoo format).

        Args:
            file_path: Path to JSONL file

        Returns:
            list[Case]
        """
        cases = []
        jsonl_data = read_jsonl_file(file_path)

        for line_num, data in enumerate(jsonl_data, 1):
            test_config = TestConfig(**data)
            if not test_config.description:
                test_config.description = f"{file_path.name}:{line_num}"
            test_case = DatasetFactory._build_cases_from_test_config(test_config)
            cases.append(test_case)

        return cases
