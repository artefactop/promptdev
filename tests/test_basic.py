"""Basic tests for PromptDev functionality."""

from pathlib import Path

import pytest

from promptdev.config.loader import load_config
from promptdev.config.models import PromptDevConfig, ProviderConfig


def test_provider_config():
    """Test provider configuration model."""
    provider = ProviderConfig(id="test-provider", model="openai:gpt-4", config={"temperature": 0.0})

    assert provider.id == "test-provider"
    assert provider.model == "openai:gpt-4"
    assert provider.config["temperature"] == 0.0


def test_config_model():
    """Test main configuration model."""
    config_data = {
        "description": "Test config",
        "prompts": ["test_prompt.yaml"],
        "providers": [
            {"id": "test-provider", "model": "openai:gpt-4", "config": {"temperature": 0.0}}
        ],
        "tests": [{"vars": {"test_var": "test_value"}}],
    }

    config = PromptDevConfig(**config_data)

    assert config.description == "Test config"
    assert len(config.providers) == 1
    assert config.providers[0].id == "test-provider"
    assert len(config.tests) == 1


def test_yaml_config_parsing():
    """Test YAML configuration file parsing."""
    # Create a temporary config file
    config_content = """
description: "Test YAML config"
prompts:
  - "test_prompt.yaml"
providers:
  - id: "test-provider"
    model: "openai:gpt-4"
    config:
      temperature: 0.0
tests:
  - vars:
      test_var: "test_value"
"""

    # Write to temp file
    import tempfile


    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    try:
        # Test loading
        config = load_config(temp_path)

        assert config.description == "Test YAML config"
        assert len(config.providers) == 1
        assert config.providers[0].model == "openai:gpt-4"

    finally:
        # Cleanup
        temp_path.unlink()


def test_cli_import():
    """Test that CLI can be imported without errors."""
    from promptdev.cli import cli

    assert cli is not None


if __name__ == "__main__":
    pytest.main([__file__])
