# Promptdev

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://github.com/astral-sh/ruff)
[![CI](https://img.shields.io/github/actions/workflow/status/artefactop/promptdev/ci.yml?style=for-the-badge)](https://github.com/artefactop/promptdev/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/artefactop/promptdev?style=for-the-badge)](https://codecov.io/gh/artefactop/promptdev)
[![Security](https://img.shields.io/github/actions/workflow/status/artefactop/promptdev/security.yml?style=for-the-badge)](https://github.com/artefactop/promptdev/actions/workflows/security.yml)


`promptdev` is a prompt evaluation framework that provides comprehensive testing for AI agents across multiple providers.

![Promptdev Demo](https://github.com/artefactop/promptdev/raw/main/docs/demo.gif)

> [!WARNING]
>
> promptdev is in preview and is not ready for production use.
>
> We're working hard to make it stable and feature-complete, but until then, expect to encounter bugs,
> missing features, and fatal errors.

## Features

- 🔒 **Type Safe** - Full Pydantic validation for inputs, outputs, and configurations  
- 🤖 **PydanticAI Integration** - Native support for PydanticAI agents (in progress) and [evaluation framework](https://ai.pydantic.dev/evals/)
- 📊 **Multi-Provider Testing** - Test across OpenAI, Together.ai, Ollama, Bedrock, and [more](https://ai.pydantic.dev/models/overview/)
- ⚡ **Performance Optimized** - File-based caching with TTL for faster repeated evaluations
- 📈 **Rich Reporting** - Beautiful console output with detailed failure analysis and provider comparisons
- 🧪 **Promptfoo Compatible** - Works with (some) existing promptfoo YAML configs and datasets
- 🎯 **Comprehensive Assertions** - Built-in evaluators plus custom Python assertion support

## Quick Start

### Installation

#### From PyPI (alpha version)
```bash
pip install promptdev --pre
```

#### From Source
```bash
git clone https://github.com/artefactop/promptdev.git
cd promptdev
pip install -e .
```

#### For Development
```bash
git clone https://github.com/artefactop/promptdev.git
cd promptdev
uv sync
uv run promptdev --help
```

### Basic Usage

#### If installed via pip:
```bash
# Run evaluation (simple demo)
promptdev eval examples/demo/config.yaml

# Run evaluation (advanced example)
promptdev eval examples/calendar_event_summary/config.yaml

# Disable caching for a run
promptdev eval examples/demo/config.yaml --no-cache

# Export results
promptdev eval examples/demo/config.yaml --output json
promptdev eval examples/demo/config.yaml --output html

# Validate configuration
promptdev validate examples/demo/config.yaml

# Cache management
promptdev cache stats
promptdev cache clear
```

#### If running from source:
```bash
uv run promptdev --help
```

## Assertion Types

Promptdev supports a comprehensive set of evaluators for different testing scenarios:

| Type            | Description                                                                                                                                                                         |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `equals`        | Checks if the output exactly equals the provided value                                                                                                                              |
| `contains`      | Checks if the output contains the expected output                                                                                                                                   |
| `is_instance`   | Checks if the output is an instance of a type with the given name                                                                                                                   |
| `max_duration`  | Checks if the execution time is under the specified maximum                                                                                                                         |
| `is_json`       | Checks if the output is a valid JSON string (optional json schema validation)                                                                                                       |
| `contains_json` | Checks if the output contains a valid json (optional json schema validation)                                                                                                        |
| `python`        | [Promptfoo compatible](https://www.promptfoo.dev/docs/configuration/expected-outputs/python/#external-py) Allows you to provide a custom Python function to validate the LLM output |


## Configuration

Promptdev uses YAML configuration files compatible with [Promptfoo](https://www.promptfoo.dev/docs/configuration/reference/) format, but only a subset is available for now:

### Promptfoo Compatibility

Promptdev maintains compatibility with promptfoo configurations to ease migration:

> To migrate if you are using ids with format `provider:chat|completion:model`, just remove the middle part `provider:model`, promptdev only supports chat.
>
> Some provider name can change for example `togetherai` is now `togeher`. Refer to [pydantic_ai models](https://ai.pydantic.dev/models/overview/) for the full list.

- **YAML configs** - Most promptfoo YAML configs work with minimal changes
- **JSONL datasets** - Existing test datasets are fully supported
- **Python assertions** - Custom `get_assert` functions work without modification
- **JSON schemas** - Schema validation uses the same format

> [!WARNING]
> Promptdev can run custom Python assertions. While powerful, 
> running arbitrary Python code always comes with [security issues](https://github.com/pydantic/pydantic-ai/pull/2808).
> Use this feature only with code you trust.

Example of a Python assertion:

```python
# tests/data/python_assert.py
from typing import Any


def get_assert(output:str, context:dict) -> bool | float | dict[str, Any]:
        """Test assertion that checks if output contains 'success'."""
        return "success" in str(output).lower()
```

## Development

```bash
# Setup development environment
uv sync

# Run tests
uv run pytest

# Format and lint code
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run ty check
```

## Roadmap

- [x] Core evaluation engine with PydanticAI integration
- [x] Multi-provider support for major AI platforms
- [x] YAML configuration loading with promptfoo compatibility
- [x] Comprehensive assertion types (JSON schema, Python, LLM-based)
- [x] File-based caching system with TTL support
- [x] Rich console reporting with failure analysis
- [x] Simple file disk cache
- [x] Better integration with PydanticAI, do not reinvent the wheel
- [x] Concurrent execution using PydanticAI natively, for faster large-scale evaluations
- [ ] Code cleanup
- [ ] Testing
- [ ] Testing promptfoo files
- [ ] Native support for PydanticAI agents
- [ ] Add support to run multiple config files with one command
- [ ] CI/CD integration helpers with change detection
- [ ] SQLite persistence for evaluation history and analytics
- [ ] Performance benchmarking and regression detection

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `uv sync`
4. Make your changes and add tests
5. Run tests: `uv run pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request


### Code Style

We use `ruff` for code formatting and linting, `ty` for type checking, and `pytest` for testing. Please ensure your code follows these standards:

```bash
uv run ruff check .       # Lint code
uv run ruff format .      # Format code
uv run ty check           # Type checking
uv run pytest             # Run tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [PydanticAI](https://ai.pydantic.dev/) for type-safe AI agent development
- Inspired by [promptfoo](https://github.com/promptfoo/promptfoo) for evaluation concepts
- Uses [Rich](https://github.com/Textualize/rich) for beautiful console output