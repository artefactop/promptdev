"""Tests for the evaluation runner."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from promptdev.config.models import (
    AssertionConfig,
    DatasetConfig,
    PromptDevConfig,
    ProviderConfig,
    TestConfig,
)
from promptdev.evaluation.results import EvaluationResults
from promptdev.evaluation.runner import EvaluationRunner


class TestEvaluationRunner:
    """Test the evaluation runner functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return PromptDevConfig(
            description="Test configuration",
            prompts=["test_prompt.yaml"],
            providers=[
                ProviderConfig(id="test-provider", model="test:model", config={"temperature": 0.0})
            ],
            tests=[
                TestConfig(
                    vars={"test_var": "test_value"},
                    assert_=[AssertionConfig(type="exact", value="expected_output")],
                )
            ],
        )

    @pytest.fixture
    def sample_prompt_file(self):
        """Create a temporary prompt file."""
        prompt_content = """
- role: system
  content: |
    You are a helpful assistant.
- role: user
  content: |-
    Process this input: {test_var}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(prompt_content)
            yield Path(f.name)
        Path(f.name).unlink()

    def test_runner_initialization(self, sample_config):
        """Test runner initialization."""
        runner = EvaluationRunner(sample_config, verbose=False)

        assert runner.config == sample_config
        assert runner.verbose is False
        assert runner.use_progress_bar is False
        assert runner.cache is not None

    def test_load_datasets(self, sample_config):
        """Test dataset loading."""
        runner = EvaluationRunner(sample_config, verbose=True)

        # Should have loaded the inline test configuration
        assert len(runner.datasets) == 1
        assert len(runner.datasets[0].test_cases) == 1
        assert runner.datasets[0].test_cases[0]["vars"]["test_var"] == "test_value"

    @pytest.mark.asyncio
    async def test_run_evaluation_mock(self, sample_config, sample_prompt_file):
        """Test evaluation with mocked components."""
        # Update config to use the real prompt file
        sample_config.prompts = [f"file://{sample_prompt_file}"]

        runner = EvaluationRunner(sample_config, verbose=False)

        # Mock the agent and evaluation methods
        with (
            patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class,
            patch.object(runner, "_evaluate_output") as mock_evaluate,
        ):
            # Setup mocks
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run_test.return_value = "test_output"
            mock_evaluate.return_value = 1.0

            # Run evaluation
            results = await runner.run_evaluation()

            # Verify results
            assert isinstance(results, EvaluationResults)
            assert len(results.provider_results) == 1
            assert results.provider_results[0].provider_id == "test-provider"
            assert len(results.provider_results[0].test_results) == 1
            assert results.provider_results[0].test_results[0].passed is True
            assert results.provider_results[0].test_results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_provider_override(self, sample_config, sample_prompt_file):
        """Test provider override functionality."""
        # Add multiple providers
        sample_config.providers.append(
            ProviderConfig(id="second-provider", model="test:model2", config={"temperature": 0.5})
        )
        sample_config.prompts = [f"file://{sample_prompt_file}"]

        runner = EvaluationRunner(sample_config, verbose=False)

        with (
            patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class,
            patch.object(runner, "_evaluate_output") as mock_evaluate,
        ):
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run_test.return_value = "test_output"
            mock_evaluate.return_value = 1.0

            # Run with provider override
            results = await runner.run_evaluation(provider_override="test-provider")

            # Should only have one provider result (the overridden one)
            assert len(results.provider_results) == 1
            assert results.provider_results[0].provider_id == "test-provider"

    def test_cache_configuration(self, sample_config):
        """Test cache configuration handling."""
        from promptdev.config.models import CacheConfig

        # Test with cache disabled
        sample_config.cache = CacheConfig(enabled=False)
        runner = EvaluationRunner(sample_config)
        assert runner.cache.enabled is False

        # Test with cache enabled and custom directory
        cache_dir = tempfile.mkdtemp()
        sample_config.cache = CacheConfig(enabled=True, cache_dir=cache_dir, ttl=3600)
        runner = EvaluationRunner(sample_config)
        assert runner.cache.enabled is True
        assert runner.cache.cache_dir == Path(cache_dir)

    @pytest.mark.asyncio
    async def test_error_handling(self, sample_config, sample_prompt_file):
        """Test error handling during evaluation."""
        sample_config.prompts = [f"file://{sample_prompt_file}"]
        runner = EvaluationRunner(sample_config, verbose=False)

        with patch("promptdev.evaluation.runner.PromptDevAgent") as mock_agent_class:
            # Setup agent to raise an exception
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run_test.side_effect = Exception("Test error")

            results = await runner.run_evaluation()

            # Should still return results but with failed tests
            assert isinstance(results, EvaluationResults)
            assert len(results.provider_results) == 1
            test_result = results.provider_results[0].test_results[0]
            assert test_result.passed is False
            assert test_result.score == 0.0
            assert "Test error" in test_result.error


class TestDatasetLoader:
    """Test dataset loading functionality."""

    def test_inline_dataset_loading(self):
        """Test loading inline test cases."""
        from promptdev.evaluation.dataset import PromptDevDataset

        test_config = TestConfig(
            vars={"input": "test", "expected": "output"},
            assert_=[AssertionConfig(type="exact", value="output")],
        )

        dataset = PromptDevDataset.from_config(test_config)
        assert len(dataset.test_cases) == 1
        assert dataset.test_cases[0]["vars"]["input"] == "test"
        assert dataset.test_cases[0]["vars"]["expected"] == "output"

    def test_jsonl_dataset_loading(self):
        """Test loading JSONL dataset files."""
        from promptdev.evaluation.dataset import PromptDevDataset

        # Create temporary JSONL file
        test_data = [
            {"vars": {"input": "test1", "expected": "output1"}},
            {"vars": {"input": "test2", "expected": "output2"}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            dataset_config = DatasetConfig(file=temp_path)
            dataset = PromptDevDataset.from_config(dataset_config)

            assert len(dataset.test_cases) == 2
            assert dataset.test_cases[0]["vars"]["input"] == "test1"
            assert dataset.test_cases[1]["vars"]["input"] == "test2"
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
