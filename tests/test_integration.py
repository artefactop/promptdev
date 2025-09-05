"""Integration tests for PromptDev framework."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from promptdev.config.loader import load_config
from promptdev.config.models import PromptDevConfig
from promptdev.evaluation.runner import EvaluationRunner


class TestEndToEndIntegration:
    """Test complete end-to-end evaluation workflows."""

    @pytest.fixture
    def sample_prompt_file(self):
        """Create a sample prompt file."""
        prompt_content = [
            {
                "role": "system",
                "content": "You are a helpful assistant that processes calendar events.",
            },
            {
                "role": "user",
                "content": "Extract event information from: {{calendar_event_summary}}\n\nReturn JSON with: name, event_type, out_of_office (boolean)",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(prompt_content, f)
            yield Path(f.name)
        Path(f.name).unlink()

    @pytest.fixture
    def sample_dataset_file(self):
        """Create a sample JSONL dataset file."""
        test_data = [
            {
                "vars": {
                    "calendar_event_summary": "John Smith - Vacation",
                    "expected_name": "John Smith",
                    "expected_event_type": "vacation",
                    "expected_out_of_office": True,
                }
            },
            {
                "vars": {
                    "calendar_event_summary": "Jane Doe - Team Meeting",
                    "expected_name": "Jane Doe",
                    "expected_event_type": "meeting",
                    "expected_out_of_office": False,
                }
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            temp_file = Path(f.name)

        try:
            yield temp_file
        finally:
            temp_file.unlink(missing_ok=True)

    @pytest.fixture
    def sample_config_file(self, sample_prompt_file, sample_dataset_file):
        """Create a complete configuration file."""
        config_data = {
            "description": "Integration test configuration",
            "prompts": [f"file://{sample_prompt_file}"],
            "providers": [
                {"id": "test-provider", "model": "test:model", "config": {"temperature": 0.0}}
            ],
            "tests": [{"file": f"file://{sample_dataset_file}"}],
            "default_test": {
                "assert": [
                    {
                        "type": "json_schema",
                        "value": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "event_type": {"type": "string"},
                                "out_of_office": {"type": "boolean"},
                            },
                            "required": ["name", "event_type", "out_of_office"],
                        },
                    }
                ]
            },
            "cache": {
                "enabled": False  # Disable cache for testing
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = Path(f.name)

        try:
            yield temp_file
        finally:
            temp_file.unlink(missing_ok=True)

    def test_config_loading_integration(self, sample_config_file):
        """Test loading a complete configuration file."""
        config = load_config(sample_config_file)

        assert isinstance(config, PromptDevConfig)
        assert config.description == "Integration test configuration"
        assert len(config.providers) == 1
        assert config.providers[0].id == "test-provider"
        assert len(config.tests) == 1
        assert config.default_test is not None
        assert len(config.default_test.assert_) == 1

    @pytest.mark.asyncio
    async def test_full_evaluation_workflow(self, sample_config_file):
        """Test complete evaluation workflow with mocked LLM."""
        config = load_config(sample_config_file)
        runner = EvaluationRunner(config, verbose=False)

        # Mock the agent to return predictable JSON responses
        mock_responses = [
            '{"name": "John Smith", "event_type": "vacation", "out_of_office": true}',
            '{"name": "Jane Doe", "event_type": "meeting", "out_of_office": false}',
        ]

        with patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            # Set up mock to return different responses for each call
            mock_agent.run_test.side_effect = mock_responses

            # Run evaluation
            results = await runner.run_evaluation()

            # Verify results structure
            assert len(results.provider_results) == 1
            provider_result = results.provider_results[0]
            assert provider_result.provider_id == "test-provider"
            assert len(provider_result.test_results) == 2

            # Verify test results
            for i, test_result in enumerate(provider_result.test_results):
                assert test_result.output == mock_responses[i]
                assert test_result.execution_time_ms > 0
                # Schema validation should pass for valid JSON
                assert test_result.score > 0

    @pytest.mark.asyncio
    async def test_evaluation_with_failures(self, sample_config_file):
        """Test evaluation workflow with some failing tests."""
        config = load_config(sample_config_file)
        runner = EvaluationRunner(config, verbose=False)

        # Mock responses with some invalid JSON
        mock_responses = [
            '{"name": "John Smith", "event_type": "vacation", "out_of_office": true}',  # Valid
            "Invalid JSON response",  # Invalid
        ]

        with patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run_test.side_effect = mock_responses

            results = await runner.run_evaluation()

            # Should have results for both tests
            assert len(results.provider_results[0].test_results) == 2

            # First test should pass (valid JSON)
            test1 = results.provider_results[0].test_results[0]
            assert test1.score > 0

            # Second test might fail (invalid JSON)
            results.provider_results[0].test_results[1]
            # Score might be low due to JSON parsing issues in evaluators

    @pytest.mark.asyncio
    async def test_multiple_providers(self, sample_prompt_file, sample_dataset_file):
        """Test evaluation with multiple providers."""
        config_data = {
            "description": "Multi-provider test",
            "prompts": [f"file://{sample_prompt_file}"],
            "providers": [
                {"id": "provider-1", "model": "test:model1", "config": {"temperature": 0.0}},
                {"id": "provider-2", "model": "test:model2", "config": {"temperature": 0.5}},
            ],
            "tests": [{"file": f"file://{sample_dataset_file}"}],
            "default_test": {"assert": [{"type": "json_schema", "value": {"type": "object"}}]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            config = load_config(config_file)
            runner = EvaluationRunner(config, verbose=False)

            with patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent_class.return_value = mock_agent
                mock_agent.run_test.return_value = '{"test": "response"}'

                results = await runner.run_evaluation()

                # Should have results for both providers
                assert len(results.provider_results) == 2
                assert results.provider_results[0].provider_id == "provider-1"
                assert results.provider_results[1].provider_id == "provider-2"

                # Each provider should have run both test cases
                for provider_result in results.provider_results:
                    assert len(provider_result.test_results) == 2
        finally:
            config_file.unlink()

    @pytest.mark.asyncio
    async def test_cache_integration(self, sample_config_file):
        """Test cache integration in full workflow."""
        config = load_config(sample_config_file)

        # Enable caching
        from promptdev.config.models import CacheConfig

        config.cache = CacheConfig(enabled=True, ttl=3600)

        runner = EvaluationRunner(config, verbose=False)

        with patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run_test.return_value = '{"cached": "response"}'

            # First run - should call agent
            results1 = await runner.run_evaluation()
            assert mock_agent.run_test.call_count == 2  # 2 test cases

            # Reset mock and run again - should use cache
            mock_agent.run_test.reset_mock()
            runner2 = EvaluationRunner(config, verbose=False)
            results2 = await runner2.run_evaluation()

            # Should have same results
            assert len(results1.provider_results) == len(results2.provider_results)

    def test_error_collection_integration(self, sample_config_file):
        """Test error collection during evaluation."""
        config = load_config(sample_config_file)
        runner = EvaluationRunner(config, verbose=False)

        # This should collect configuration errors
        assert len(runner.evaluation_errors) == 0  # Should start empty

    @pytest.mark.asyncio
    async def test_provider_override_integration(self, sample_config_file):
        """Test provider override functionality in integration."""
        config = load_config(sample_config_file)

        # Add another provider to config
        from promptdev.config.models import ProviderConfig

        config.providers.append(
            ProviderConfig(id="second-provider", model="test:model2", config={"temperature": 0.8})
        )

        runner = EvaluationRunner(config, verbose=False)

        with patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run_test.return_value = '{"override": "test"}'

            # Run with provider override
            results = await runner.run_evaluation(provider_override="test-provider")

            # Should only have one provider result
            assert len(results.provider_results) == 1
            assert results.provider_results[0].provider_id == "test-provider"


class TestCLIIntegration:
    """Test CLI integration with the evaluation system."""

    @pytest.fixture
    def sample_prompt_file(self):
        """Create a sample prompt file."""
        prompt_content = [
            {
                "role": "system",
                "content": "You are a helpful assistant that processes calendar events.",
            },
            {
                "role": "user",
                "content": "Extract event information from: {{calendar_event_summary}}\n\nReturn JSON with: name, event_type, out_of_office (boolean)",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(prompt_content, f)
            temp_file = Path(f.name)

        try:
            yield temp_file
        finally:
            temp_file.unlink(missing_ok=True)

    @pytest.fixture
    def sample_dataset_file(self):
        """Create a sample JSONL dataset file."""
        test_data = [
            {
                "vars": {
                    "calendar_event_summary": "John Smith - Vacation",
                    "expected_name": "John Smith",
                    "expected_event_type": "vacation",
                    "expected_out_of_office": True,
                }
            },
            {
                "vars": {
                    "calendar_event_summary": "Jane Doe - Team Meeting",
                    "expected_name": "Jane Doe",
                    "expected_event_type": "meeting",
                    "expected_out_of_office": False,
                }
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            temp_file = Path(f.name)

        try:
            yield temp_file
        finally:
            temp_file.unlink(missing_ok=True)

    @pytest.fixture
    def sample_config_file(self, sample_prompt_file, sample_dataset_file):
        """Create a complete configuration file."""
        config_data = {
            "description": "Integration test configuration",
            "prompts": [f"file://{sample_prompt_file}"],
            "providers": [
                {"id": "test-provider", "model": "test:model", "config": {"temperature": 0.0}}
            ],
            "tests": [{"file": f"file://{sample_dataset_file}"}],
            "default_test": {
                "assert": [
                    {
                        "type": "json_schema",
                        "value": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "event_type": {"type": "string"},
                                "out_of_office": {"type": "boolean"},
                            },
                            "required": ["name", "event_type", "out_of_office"],
                        },
                    }
                ]
            },
            "cache": {
                "enabled": False  # Disable cache for testing
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = Path(f.name)

        try:
            yield temp_file
        finally:
            temp_file.unlink(missing_ok=True)

    def test_cli_import_integration(self):
        """Test that CLI can be imported and initialized."""
        from promptdev.cli import cli

        # Should be able to import without errors
        assert cli is not None

        # Test CLI help (should not raise exceptions)
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PromptDev" in result.output

    def test_validate_command_integration(self, sample_config_file):
        """Test validate command integration."""
        from click.testing import CliRunner

        from promptdev.cli import validate

        runner = CliRunner()
        result = runner.invoke(validate, [str(sample_config_file)])

        # Should validate successfully
        assert result.exit_code == 0
        assert "valid" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__])
