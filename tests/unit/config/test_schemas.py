from promptdev.config.schemas import PromptDevConfig, ProviderConfig, TestConfig


class TestProviderConfig:
    """Test cases for the Provider model."""

    def test_provider_with_config(self):
        data = {"id": "promptdev:echo", "config": {"temperature": 0.7}}
        provider = ProviderConfig(**data)
        assert provider.id == "promptdev:echo"
        assert provider.config["temperature"] == 0.7

    def test_provider_without_config(self):
        data = {"id": "promptdev:echo"}
        provider = ProviderConfig(**data)
        assert provider.id == "promptdev:echo"
        assert provider.config == {}  # Default value


class TestTestConfig:
    """Test cases for the TestConfig model."""

    def test_config_with_assertions(self):
        data = {
            "description": "Simple validation test",
            "vars": {"text": "test"},
            "assert": [{"type": "contains", "value": "test"}],
            "metadata": {"key": "value"},
        }
        test_config = TestConfig(**data)
        assert test_config.description == "Simple validation test"
        assert test_config.vars == {"text": "test"}
        assert test_config.assert_[0].type == "contains"
        assert test_config.assert_[0].value == "test"


class TestPromptDevConfig:
    """Test cases for the Config model."""

    def test_valid_config(self, tmp_path):
        # Create temporary files for testing
        prompt_file = tmp_path / "simple_prompt.yaml"
        prompt_file.write_text("system: test\nuser: test")

        dataset_dir = tmp_path / "simple"
        dataset_dir.mkdir()
        dataset_file = dataset_dir / "dataset.jsonl"
        dataset_file.write_text(
            '{"vars":{"text":"test"}, "assert":[{"type":"contains", "value":"test"}]}'
        )

        # NOTE: Using Path objects instead of file:// URLs
        data = {
            "description": "Simple validation test configuration",
            "prompts": [prompt_file],
            "providers": [
                {"id": "promptdev:echo", "config": {"temperature": 0.0}},
                "promptdev:echo",
            ],
            "tests": [
                {
                    "description": "Simple validation test",
                    "vars": {"text": "test"},
                    "assert": [{"type": "contains", "value": "test"}],
                },
                dataset_file,
            ],
        }
        config = PromptDevConfig(**data)
        assert config.description == "Simple validation test configuration"
        assert len(config.prompts) == 1
        assert config.prompts[0] == prompt_file
        assert len(config.providers) == 2
        assert len(config.tests) == 2
        assert config.tests[0].description == "Simple validation test"
        assert config.tests[1] == dataset_file
