"""Pydantic models for PromptDev configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# These models are a subset of the promptfoo schema
# https://promptfoo.dev/config-schema.json


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    id: str | None = Field(None, description="Unique identifier for the provider")
    label: str | None = Field(None, description="Human-readable label for the provider")
    model: str | None = Field(None, description="Model identifier (e.g., 'ollama:llama3.2')")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )
    prompts: list[str] | None = Field(None, description="Provider-specific prompts")
    transform: str | None = Field(None, description="Transform function for provider")
    delay: float | None = Field(None, description="Delay between requests in milliseconds")
    env: dict[str, str] | None = Field(None, description="Environment variables for provider")


class AssertionConfig(BaseModel):
    """Configuration for an assertion/evaluator."""

    type: str | None = Field(
        None, description="Type of assertion (e.g., 'json_schema', 'python', 'llm_judge')"
    )
    value: Any | None = Field(None, description="Assertion value or configuration")
    threshold: float | None = Field(None, description="Threshold for scoring assertions")
    weight: float | None = Field(None, description="Weight for this assertion in scoring")
    provider: str | None = Field(None, description="Provider to use for this assertion")
    rubric: str | None = Field(None, description="Rubric for LLM-based evaluation")
    model: str | None = Field(None, description="Model to use for LLM-based evaluation")

    # Support for promptfoo assertion formats
    metric: str | None = Field(None, description="Metric name for this assertion")
    transform: str | None = Field(None, description="Transform function for assertion result")


class TestConfig(BaseModel):
    """Configuration for individual tests."""

    description: str | None = Field(None, description="Description of the test")
    vars: dict[str, Any] | None = Field(None, description="Test variables")
    assert_: list[AssertionConfig | str] = Field(default_factory=list, alias="assert")
    provider: str | None = Field(None, description="Override provider for this test")
    options: dict[str, Any] | None = Field(None, description="Test-specific options")
    metadata: dict[str, Any] | None = Field(None, description="Test metadata")


class DatasetConfig(BaseModel):
    """Configuration for test datasets."""

    file: str | Path | None = Field(None, description="Path to dataset file (JSONL)")
    vars: dict[str, Any] | None = Field(None, description="Inline variables")
    tests: list[TestConfig] | None = Field(None, description="Inline test cases")

    # Support for promptfoo dataset formats
    transform: str | None = Field(None, description="Transform function for dataset")
    filter: str | None = Field(None, description="Filter function for dataset")


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(True, description="Enable caching")
    ttl: int | None = Field(None, description="Time to live for cache entries in seconds")
    cache_dir: str | None = Field(None, description="Directory to store cache files")


# Support for flexible provider definitions (string, object, or array)
ProviderType = str | ProviderConfig | list[str | ProviderConfig]


class PromptDevConfig(BaseModel):
    """Main configuration model for PromptDev evaluations."""

    description: str | None = Field(None, description="Description of this evaluation")
    prompts: list[str | Path] = Field(..., description="List of prompt files or templates")
    providers: ProviderType = Field(..., description="LLM providers to test")
    tests: list[TestConfig | DatasetConfig] | None = Field(None, description="Test configurations")

    # Support for promptfoo naming conventions
    default_test: TestConfig | None = Field(
        None, description="Default test configuration", alias="defaultTest"
    )
    assertion_templates: dict[str, AssertionConfig] | None = Field(
        None, description="Reusable assertion templates", alias="assertionTemplates"
    )

    # Additional promptfoo fields
    schemas: dict[str, dict[str, Any]] | None = Field(
        None, description="JSON schemas for validation"
    )
    cache: CacheConfig | None = Field(None, description="Cache configuration")
    tags: dict[str, str] | None = Field(None, description="Tags for categorization")

    # Evaluation options
    max_concurrency: int | None = Field(
        None, description="Maximum concurrent evaluations", alias="maxConcurrency"
    )
    delay: int | None = Field(None, description="Delay between evaluations in milliseconds")
    repeat: int | None = Field(None, description="Number of times to repeat each test")

    # Output options
    output: list[str] | None = Field(None, description="Output formats")
    write_latest_results: bool | None = Field(
        None, description="Write latest results", alias="writeLatestResults"
    )

    # Allow extra fields for promptfoo-style direct schema definitions
    model_config = {"populate_by_name": True, "extra": "allow"}
