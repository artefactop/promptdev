"""Provider configuration utilities."""

from typing import Any

from ..config.models import ProviderConfig


def get_provider_config(provider_id: str, providers: list[ProviderConfig]) -> ProviderConfig:
    """Get provider configuration by ID.

    Args:
        provider_id: Provider identifier
        providers: List of available providers

    Returns:
        Provider configuration

    Raises:
        ValueError: If provider not found
    """
    for provider in providers:
        if provider.id == provider_id:
            return provider

    available_ids = [p.id for p in providers]
    raise ValueError(f"Provider '{provider_id}' not found. Available: {available_ids}")


def normalize_model_name(model: str) -> str:
    """Normalize model name to PydanticAI format.

    Converts various model name formats to PydanticAI format:
    - 'openai:gpt-4' -> 'openai:gpt-4' (already correct)
    - 'togetherai:chat:meta-llama/Meta-Llama-3.1-8B' -> 'together:meta-llama/Meta-Llama-3.1-8B'
    - 'ollama:chat:llama3.1:8b' -> 'ollama:llama3.1:8b'

    Args:
        model: Model name in various formats

    Returns:
        Normalized model name for PydanticAI
    """
    # Handle togetherai -> together conversion
    if model.startswith("togetherai:"):
        # togetherai:chat:model -> together:model
        parts = model.split(":", 2)
        if len(parts) >= 3:
            return f"together:{parts[2]}"
        elif len(parts) == 2:
            return f"together:{parts[1]}"

    # Handle ollama chat format
    if model.startswith("ollama:chat:"):
        # ollama:chat:model -> ollama:model
        parts = model.split(":", 2)
        if len(parts) >= 3:
            return f"ollama:{parts[2]}"

    # Handle bedrock format
    if model.startswith("bedrock:"):
        # bedrock:model -> bedrock:model (keep as is)
        return model

    # Default: return as-is
    return model


def get_provider_defaults(provider_type: str) -> dict[str, Any]:
    """Get default configuration for provider type.

    Args:
        provider_type: Provider type (openai, together, ollama, etc.)

    Returns:
        Default configuration dict
    """
    defaults = {
        "openai": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        "together": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        "ollama": {
            "temperature": 0.0,
        },
        "bedrock": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        "anthropic": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    }

    return defaults.get(provider_type, {})
