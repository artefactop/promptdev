from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_evals import Dataset

from promptdev.config.schemas import PromptDevConfig, ProviderConfig
from promptdev.core.factory import DatasetFactory, ModelFactory, PromptTemplate


@dataclass
class ProviderFactory:
    """A provider with all components built and ready"""

    config: ProviderConfig
    model: Model
    model_settings: ModelSettings

    @classmethod
    def from_config(cls, config: ProviderConfig) -> "ProviderFactory":
        return cls(
            config=config,
            model=ModelFactory.build_model(config),
            model_settings=ModelFactory.build_model_settings(config),
        )


@dataclass
class EvaluationContext:
    """Complete evaluation context with all built components"""

    config: PromptDevConfig
    prompt_templates: list[PromptTemplate]
    built_providers: list[ProviderFactory]
    dataset: Dataset

    @classmethod
    def from_config(cls, config: PromptDevConfig) -> "EvaluationContext":
        """Build complete evaluation context from config"""

        # Build prompt templates
        prompt_templates = []
        for prompt in config.prompts:
            if isinstance(prompt, Path) and prompt.suffix in [".yaml", ".yml"]:
                template = PromptTemplate.from_file(prompt)
            elif isinstance(prompt, Path) and prompt.suffix in [".txt"]:
                template = PromptTemplate.from_string(prompt.read_text())
            elif isinstance(prompt, str):
                template = PromptTemplate.from_string(prompt)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")
            prompt_templates.append(template)

        if not prompt_templates:
            raise ValueError("No prompts specified in configuration")

        # Build providers
        built_providers = []
        for provider_config in config.providers:
            built_provider = ProviderFactory.from_config(provider_config)
            built_providers.append(built_provider)

        # Build dataset
        dataset = DatasetFactory.build_dataset(config)

        return cls(
            config=config,
            prompt_templates=prompt_templates,
            built_providers=built_providers,
            dataset=dataset,
        )
