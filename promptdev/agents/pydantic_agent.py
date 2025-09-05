"""PydanticAI agent wrapper for PromptDev."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_ai import Agent
from rich.console import Console

from ..config.models import ProviderConfig

console = Console()


class PromptDevAgent:
    """Wrapper around PydanticAI Agent with PromptDev-specific functionality."""

    def __init__(
        self,
        prompt_path: str | Path,
        provider_config: ProviderConfig,
        output_type: type[BaseModel] | None = None,
    ):
        """Initialize PromptDev agent.

        Args:
            prompt_path: Path to YAML prompt file (compatible with existing format)
            provider_config: Provider configuration
            output_type: Expected output type (Pydantic model)
        """
        self.prompt_path = Path(prompt_path)
        self.provider_config = provider_config
        self.output_type = output_type

        # Load YAML prompt (compatible with existing promptfoo format)
        self.system_prompt, self.user_template = self._load_yaml_prompt()

        # Create PydanticAI agent with proper model setup
        model = self._create_model(provider_config)

        # Extract model parameters for the Agent
        self.model_params = self._get_model_params(provider_config)

        self.agent = Agent(
            model=model,
            system_prompt=self.system_prompt,
            output_type=output_type,
            **self.model_params,
        )

    def _load_yaml_prompt(self) -> tuple[str, str]:
        """Load YAML prompt file compatible with existing format.

        Expected format (from your existing prompts):
        - role: system
          content: |
            System prompt content here
        - role: user
          content: |-
            User prompt with {{variables}}

        Returns:
            Tuple of (system_prompt, user_template)
        """
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        with open(self.prompt_path, encoding="utf-8") as f:
            messages = yaml.safe_load(f)

        system_content = ""
        user_content = ""

        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
            elif message["role"] == "user":
                user_content = message["content"]

        # Handle double-brace conversion ({{var}} -> {var}) for compatibility
        system_content = system_content.replace("{{", "{").replace("}}", "}")
        user_content = user_content.replace("{{", "{").replace("}}", "}")

        return system_content, user_content

    async def run_test(self, variables: dict[str, Any]) -> Any:
        """Run test with variable substitution.

        Args:
            variables: Dictionary of variables for template substitution

        Returns:
            Agent output (structured if output_type is specified)
        """
        # Format both system and user templates with variables
        try:
            formatted_system = self.system_prompt.format(**variables)
            formatted_prompt = self.user_template.format(**variables)

            # Debug output removed - schema formatting is working correctly

        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing variable '{missing_var}' in prompt template") from e

        # Create a new agent with the formatted system prompt for this specific run
        model = self._create_model(self.provider_config)

        # Create agent with formatted system prompt
        from pydantic_ai import Agent

        dynamic_agent = Agent(
            model=model,
            system_prompt=formatted_system,
            output_type=self.output_type,
            **self.model_params,
        )

        # Run PydanticAI agent with model parameters
        model_settings = self._get_run_settings()
        if model_settings:
            result = await dynamic_agent.run(formatted_prompt, model_settings=model_settings)
        else:
            result = await dynamic_agent.run(formatted_prompt)
        return result.output

    def validate_template_variables(self, variables: dict[str, Any]) -> list[str]:
        """Validate that all required template variables are provided.

        Args:
            variables: Variables to check

        Returns:
            List of missing variable names
        """
        import re

        # Extract variable names from user template
        template_vars = set(re.findall(r"\{(\w+)\}", self.user_template))
        provided_vars = set(variables.keys())

        missing_vars = template_vars - provided_vars
        return list(missing_vars)

    def get_template_variables(self) -> list[str]:
        """Get list of all variables used in the prompt template.

        Returns:
            List of variable names found in template
        """
        import re

        # Extract variables from both system and user templates
        system_vars = re.findall(r"\{(\w+)\}", self.system_prompt)
        user_vars = re.findall(r"\{(\w+)\}", self.user_template)

        # Return unique variables
        return list(set(system_vars + user_vars))

    def _create_model(self, provider_config: ProviderConfig):
        """Create appropriate PydanticAI model based on provider configuration."""
        model_name = provider_config.model
        config = provider_config.config

        if model_name.startswith("ollama:"):
            # Handle Ollama models using OpenAI-compatible API
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.ollama import OllamaProvider

            # Extract the actual model name (remove ollama: prefix)
            actual_model_name = model_name[7:]  # Remove "ollama:" prefix

            # Create Ollama provider
            base_url = config.get("base_url", "http://localhost:11434/v1")
            provider = OllamaProvider(base_url=base_url)

            # Create OpenAI-compatible model with Ollama provider
            # Note: temperature and other model params are handled by the Agent, not the model
            return OpenAIChatModel(
                model_name=actual_model_name,
                provider=provider,
            )

        if model_name.startswith("openai:"):
            # Handle OpenAI models
            from pydantic_ai.models.openai import OpenAIChatModel

            actual_model_name = model_name[7:]  # Remove "openai:" prefix
            return OpenAIChatModel(actual_model_name, **config)

        if model_name.startswith("anthropic:"):
            # Handle Anthropic models
            from pydantic_ai.models.anthropic import AnthropicChatModel

            actual_model_name = model_name[10:]  # Remove "anthropic:" prefix
            return AnthropicChatModel(actual_model_name, **config)

        if model_name.startswith("test"):
            # Handle test models
            from pydantic_ai.models.test import TestModel

            return TestModel()

        # Fallback: try to infer the model using PydanticAI's auto-detection
        from pydantic_ai import models

        return models.infer_model(model_name)

    def _get_model_params(self, provider_config: ProviderConfig) -> dict:
        """Extract model parameters that should be passed to Agent, not Model."""
        return {}

        # For now, we'll handle temperature through the run_sync call
        # rather than the Agent initialization, as that's more flexible

    def _get_run_settings(self) -> dict:
        """Get model settings to pass to the run method."""
        config = self.provider_config.config
        settings = {}

        # Extract common model parameters
        if "temperature" in config:
            settings["temperature"] = config["temperature"]
        if "max_tokens" in config:
            settings["max_tokens"] = config["max_tokens"]

        return settings
