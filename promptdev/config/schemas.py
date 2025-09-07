from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# These models are a subset of the promptfoo schema
# https://promptfoo.dev/config-schema.json


class ProviderConfig(BaseModel):
    id: str | None = Field(None, description="Unique identifier for the provider")
    model: str
    config: dict[str, Any] = Field(default_factory=dict)


class AssertionConfig(BaseModel):
    type: str
    value: Any | None = None
    threshold: float | None = None
    # provider for llm-rubric, etc


class TestConfig(BaseModel):
    description: str | None = None
    vars: dict[str, Any] | None = None
    assert_: list[AssertionConfig | str] = Field(default_factory=list, alias="assert")
    metadata: dict[str, Any] | None = None
    ## threshold: float|None = None
    # provider for llm-rubric, etc


class DatasetConfig(BaseModel):
    file: str | Path | None = None
    vars: dict[str, Any] | None = None
    tests: list[TestConfig] | None = None


class PromptDevConfigOptions(BaseModel):
    cache_enabled: bool = Field(
        default=False, alias="cache", description="Whether or not to cache the results"
    )


class PromptDevConfig(BaseModel):
    description: str | None = None
    prompts: list[str | Path]
    providers: list[ProviderConfig]
    tests: list[TestConfig | DatasetConfig] | None = None
    default_test: TestConfig | None = Field(None, alias="defaultTest")
    options: PromptDevConfigOptions = Field(PromptDevConfigOptions, alias="evaluateOptions")
