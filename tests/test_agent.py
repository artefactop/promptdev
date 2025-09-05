"""Tests for the PydanticAI agent wrapper."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from promptdev.agents.pydantic_agent import PromptDevAgent
from promptdev.config.models import ProviderConfig


class TestPromptDevAgent:
    """Test the PydanticAI agent wrapper."""

    @pytest.fixture
    def sample_prompt_file(self):
        """Create a sample prompt file for testing."""
        prompt_content = [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in {{task_type}}.",
            },
            {
                "role": "user",
                "content": "Process this input: {{user_input}}\nReturn the result in {{output_format}} format.",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(prompt_content, f)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def provider_config(self):
        """Create a sample provider configuration."""
        return ProviderConfig(
            id="test-provider", model="test:model", config={"temperature": 0.0, "max_tokens": 100}
        )

    def test_agent_initialization(self, sample_prompt_file, provider_config):
        """Test agent initialization."""
        agent = PromptDevAgent(
            prompt_path=sample_prompt_file, provider_config=provider_config, output_type=str
        )

        assert agent.prompt_path == sample_prompt_file
        assert agent.provider_config == provider_config
        assert agent.output_type is str
        assert agent.system_prompt is not None
        assert agent.user_template is not None

    def test_yaml_prompt_loading(self, sample_prompt_file, provider_config):
        """Test YAML prompt file loading."""
        agent = PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

        # Check that prompts were loaded and variables converted from {{}} to {}
        assert "You are a helpful assistant specialized in {task_type}" in agent.system_prompt
        assert "Process this input: {user_input}" in agent.user_template
        assert "Return the result in {output_format} format" in agent.user_template

    def test_template_variable_detection(self, sample_prompt_file, provider_config):
        """Test detection of template variables."""
        agent = PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

        variables = agent.get_template_variables()
        expected_vars = {"task_type", "user_input", "output_format"}

        assert set(variables) == expected_vars

    def test_template_variable_validation(self, sample_prompt_file, provider_config):
        """Test template variable validation."""
        agent = PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

        # Test with all required variables
        complete_vars = {
            "task_type": "data processing",
            "user_input": "test data",
            "output_format": "JSON",
        }
        missing = agent.validate_template_variables(complete_vars)
        assert len(missing) == 0

        # Test with missing variables
        incomplete_vars = {
            "task_type": "data processing",
            "user_input": "test data",
            # missing output_format
        }
        missing = agent.validate_template_variables(incomplete_vars)
        assert "output_format" in missing

    @pytest.mark.asyncio
    async def test_run_test_with_mock(self, sample_prompt_file, provider_config):
        """Test running a test with mocked PydanticAI agent."""
        agent = PromptDevAgent(
            prompt_path=sample_prompt_file, provider_config=provider_config, output_type=str
        )

        test_variables = {
            "task_type": "text analysis",
            "user_input": "Analyze this text",
            "output_format": "summary",
        }

        with patch("pydantic_ai.Agent") as mock_agent_class:
            # Setup mock
            mock_agent_instance = AsyncMock()
            mock_result = Mock()
            mock_result.output = "Test output result"
            mock_agent_instance.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent_instance

            # Run test
            result = await agent.run_test(test_variables)

            # Verify results
            assert result == "Test output result"

            # Verify agent was created and called
            mock_agent_class.assert_called()
            mock_agent_instance.run.assert_called_once()

    def test_missing_variable_error(self, sample_prompt_file, provider_config):
        """Test error handling for missing template variables."""
        agent = PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

        incomplete_vars = {
            "task_type": "data processing",
            # missing user_input and output_format
        }
        # This should fail during template formatting
        with pytest.raises(ValueError, match="Missing variable"), patch("pydantic_ai.Agent"):
            import asyncio

            asyncio.run(agent.run_test(incomplete_vars))

    def test_model_creation_openai(self, sample_prompt_file):
        """Test OpenAI model creation."""
        provider_config = ProviderConfig(
            id="openai-provider", model="openai:gpt-4", config={"temperature": 0.5}
        )

        with patch("pydantic_ai.models.openai.OpenAIChatModel") as mock_model:
            PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

            # Verify the model was created (the exact call signature depends on PydanticAI internals)
            mock_model.assert_called()

    def test_model_creation_ollama(self, sample_prompt_file):
        """Test Ollama model creation."""
        provider_config = ProviderConfig(
            id="ollama-provider",
            model="ollama:llama2",
            config={"base_url": "http://localhost:11434/v1"},
        )

        with (
            patch("pydantic_ai.models.openai.OpenAIChatModel") as mock_model,
            patch("pydantic_ai.providers.ollama.OllamaProvider") as mock_provider,
        ):
            PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

            mock_provider.assert_called_with(base_url="http://localhost:11434/v1")
            mock_model.assert_called()

    def test_model_creation_test(self, sample_prompt_file):
        """Test test model creation."""
        provider_config = ProviderConfig(id="test-provider", model="test", config={})

        # Set environment variable to avoid OpenAI API key error
        import os

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("pydantic_ai.models.test.TestModel"),
        ):
            agent = PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

            # The test model creation might be handled differently by PydanticAI
            # Just verify no errors during agent creation
            assert agent is not None

    def test_run_settings_extraction(self, sample_prompt_file, provider_config):
        """Test extraction of model settings for run method."""
        provider_config.config = {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9,
            "other_param": "ignored",
        }

        agent = PromptDevAgent(prompt_path=sample_prompt_file, provider_config=provider_config)

        settings = agent._get_run_settings()

        # Should extract only known model parameters
        assert settings["temperature"] == 0.7
        assert settings["max_tokens"] == 150
        assert "other_param" not in settings

    def test_prompt_file_not_found(self, provider_config):
        """Test error handling for missing prompt files."""
        non_existent_path = Path("/non/existent/prompt.yaml")

        with pytest.raises(FileNotFoundError):
            PromptDevAgent(prompt_path=non_existent_path, provider_config=provider_config)

    def test_complex_prompt_structure(self, provider_config):
        """Test handling of complex prompt structures."""
        # Create a more complex prompt with multiple variables and formatting
        complex_prompt = [
            {
                "role": "system",
                "content": "You are {{role_type}} working on {{project_name}}.\nYour expertise: {{expertise_areas}}",
            },
            {
                "role": "user",
                "content": "Task: {{task_description}}\n\nContext:\n{{context_data}}\n\nPlease provide {{output_type}} response.",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(complex_prompt, f)
            prompt_path = Path(f.name)

        try:
            agent = PromptDevAgent(prompt_path=prompt_path, provider_config=provider_config)

            # Test variable detection
            variables = set(agent.get_template_variables())
            expected = {
                "role_type",
                "project_name",
                "expertise_areas",
                "task_description",
                "context_data",
                "output_type",
            }
            assert variables == expected

            # Test system prompt formatting
            assert "You are {role_type} working on {project_name}" in agent.system_prompt
            assert "Your expertise: {expertise_areas}" in agent.system_prompt

        finally:
            prompt_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
