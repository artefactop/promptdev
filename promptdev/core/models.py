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
            print(f"loading prompt: {prompt} ({type(prompt)})")
            if isinstance(prompt, Path):
                template = PromptTemplate.from_file(prompt)
            else:
                # It's an inline template string
                template = PromptTemplate.from_string(prompt)
            prompt_templates.append(template)

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
