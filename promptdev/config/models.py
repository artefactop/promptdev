"""Pydantic models for PromptDev configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    id: str = Field(..., description="Unique identifier for the provider")
    model: str = Field(..., description="Model identifier (e.g., 'openai:gpt-4')")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )


class AssertionConfig(BaseModel):
    """Configuration for an assertion/evaluator."""

    type: str | None = Field(
        None, description="Type of assertion (e.g., 'json_schema', 'python', 'llm_judge')"
    )
    value: Any | None = Field(None, description="Assertion value or configuration")
    threshold: float | None = Field(None, description="Threshold for scoring assertions")
    ref: str | None = Field(None, description="Reference to assertion template")
    rubric: str | None = Field(None, description="Rubric for LLM-based evaluation")
    model: str | None = Field(None, description="Model to use for LLM-based evaluation")

    def model_post_init(self, __context):
        """Validate that either type or ref is provided."""
        if not self.type and not self.ref:
            raise ValueError("Either 'type' or 'ref' must be provided for assertion configuration")


class DatasetConfig(BaseModel):
    """Configuration for test datasets."""

    file: str | Path | None = Field(None, description="Path to dataset file (JSONL)")
    vars: dict[str, Any] | None = Field(None, description="Inline variables")
    inline: list[dict[str, Any]] | None = Field(None, description="Inline test cases")


class TestConfig(BaseModel):
    """Configuration for individual tests."""

    vars: dict[str, Any] | None = Field(None, description="Test variables")
    assert_: list[AssertionConfig] = Field(default_factory=list, alias="assert")
    provider: str | None = Field(None, description="Override provider for this test")


class DisplayConfig(BaseModel):
    """Configuration for output display."""

    include: list[str] | None = Field(None, description="Fields to include in output")
    verbose: bool = Field(False, description="Enable verbose output")


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(True, description="Enable caching")
    ttl: int | None = Field(None, description="Time to live for cache entries in seconds")
    cache_dir: str | None = Field(None, description="Directory to store cache files")


class PromptDevConfig(BaseModel):
    """Main configuration model for PromptDev evaluations."""

    description: str | None = Field(None, description="Description of this evaluation")
    prompts: list[str | Path] = Field(..., description="List of prompt files or templates")
    providers: list[ProviderConfig] = Field(..., description="LLM providers to test")
    tests: list[TestConfig | DatasetConfig] = Field(..., description="Test configurations")
    default_test: TestConfig | None = Field(
        None, description="Default test configuration", alias="defaultTest"
    )
    assertion_templates: dict[str, AssertionConfig] | None = Field(
        None, description="Reusable assertion templates", alias="assertionTemplates"
    )
    schemas: dict[str, dict[str, Any]] | None = Field(
        None, description="JSON schemas for validation"
    )
    display: DisplayConfig | None = Field(None, description="Display configuration")
    cache: CacheConfig | None = Field(None, description="Cache configuration")

    # Allow extra fields for promptfoo-style direct schema definitions
    model_config = {"populate_by_name": True, "extra": "allow"}
