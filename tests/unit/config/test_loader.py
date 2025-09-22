import tempfile
from pathlib import Path

import yaml

from promptdev.config.loader import _resolve_file_urls, load_config


class TestLoader:
    """Test cases for the loader module."""

    def test_loader(self, tmp_path):
        # Create a temporary YAML file for the prompt inside tmp_path
        prompt_file = tmp_path / "simple_prompt.yaml"
        with open(prompt_file, "w") as f:
            yaml.dump({"system": "system_prompt", "user": "user_prompt"}, f)

        # Create a temporary directory for the dataset
        dataset_dir = tmp_path / "simple"
        dataset_dir.mkdir()
        dataset_file = dataset_dir / "dataset.jsonl"
        with open(dataset_file, "w") as f:
            f.write('{"input": "test", "output": "test"}')

        # Create a temporary config file inside tmp_path
        config_file = tmp_path / "config.yaml"
        config_data = {
            "description": "Simple validation test configuration",
            "prompts": ["file://simple_prompt.yaml"],  # Relative to tmp_path
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
                "file://simple/dataset.jsonl",  # Relative to tmp_path
            ],
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load the config using the loader
        config = load_config(config_file)

        # Assert that the paths are resolved correctly
        assert config.prompts[0] == Path(prompt_file)
        assert config.tests[1] == Path(dataset_file)

    def test_resolve_file_urls(self):
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="simple_prompt", suffix=".yaml", delete=False
        ) as tmpfile:
            yaml.dump({"system": "system_prompt", "user": "user_prompt"}, tmpfile)
            tmpfile.flush()
            tmp_path = Path(tmpfile.name)

        try:
            # Update config_data to point to the temporary file
            config_data = {
                "description": "Simple validation test configuration",
                "prompts": [f"file://{tmp_path}"],
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
                    "file://./simple/dataset.jsonl",
                ],
            }

            base_path = Path(".")
            resolved_data = _resolve_file_urls(config_data, base_path)

            # Assert that the path is resolved correctly
            assert resolved_data["prompts"][0] == tmp_path.resolve()
            assert resolved_data["tests"][1] == Path("./simple/dataset.jsonl").resolve()
        finally:
            # Clean up the temporary file
            tmpfile.close()
            tmp_path.unlink()
