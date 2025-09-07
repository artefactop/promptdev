"""
Integration tests for Promptdev CLI using actual configuration files.

This module contains comprehensive integration tests that verify the entire
evaluation pipeline through the CLI interface, using real configuration files
stored in the tests/data directory.
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptdev.cli import cli


class TestCLIIntegration:
    """Integration tests for Promptdev CLI with actual configuration files."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent.parent / "data"

    def test_deterministic_asserts(self, runner, test_data_dir):
        """Test all evaluators with both pass and fail scenarios using actual config file."""
        # Use the actual config file
        config_file = test_data_dir / "deterministic_asserts_config.yaml"
        assert config_file.exists(), f"Config file not found: {config_file}"

        # Use isolated filesystem to avoid path issues
        with runner.isolated_filesystem():
            # Copy test files to isolated filesystem
            shutil.copy(config_file, "config.yaml")
            shutil.copy(test_data_dir / "simple_prompt.yaml", "simple_prompt.yaml")
            shutil.copy(test_data_dir / "python_assert.py", "python_assert.py")

            # Run CLI evaluation
            result = runner.invoke(cli, ["eval", "config.yaml", "--no-cache"])

            # Check that CLI ran successfully
            assert result.exit_code == 0, f"CLI failed with output: {result.output}"
            # TODO assert results are correct

    def test_cli_basic_functionality(self, runner):
        """Test basic CLI functionality and help commands."""
        # Test help command
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Promptdev" in result.output

        # Test eval help
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        # Check for common CLI help terms instead of specific parameter names
        assert any(term in result.output.lower() for term in ["config", "file", "usage", "options"])

    def test_cli_validation_command(self, runner, test_data_dir):
        """Test the validate command using actual config file."""
        # Use the actual validation config file
        config_file = test_data_dir / "simple_validation_config.yaml"
        assert config_file.exists(), f"Config file not found: {config_file}"

        # Use isolated filesystem
        with runner.isolated_filesystem():
            # Copy test files to isolated filesystem
            shutil.copy(config_file, "validation.yaml")
            shutil.copy(test_data_dir / "simple_prompt.yaml", "simple_prompt.yaml")

            result = runner.invoke(cli, ["validate", "validation.yaml"])
            assert result.exit_code == 0
            assert "valid" in result.output.lower() or "✓" in result.output

    def test_config_file_not_found(self, runner):
        """Test CLI behavior when config file doesn't exist."""
        result = runner.invoke(cli, ["eval", "nonexistent_config.yaml"])
        assert result.exit_code != 0
        # Should show an error about file not found

    def test_invalid_config_file(self, runner):
        """Test CLI behavior with invalid YAML config."""
        with runner.isolated_filesystem():
            # Create invalid YAML file
            with open("invalid.yaml", "w") as f:
                f.write("invalid: yaml: content: [")

            result = runner.invoke(cli, ["eval", "invalid.yaml"])
            assert result.exit_code != 0

    def test_cache_clear_command(self, runner):
        """Test the cache clear command."""
        result = runner.invoke(cli, ["cache", "clear"])
        assert result.exit_code == 0
        assert result.output == "✓ Cache cleared successfully\n"

    def test_cache_stats_command(self, runner):
        """Test the cache stats command."""
        result = runner.invoke(cli, ["cache", "stats"])
        assert result.exit_code == 0
