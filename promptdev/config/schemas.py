from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field, field_validator


def must_exist_file(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path.resolve()


ExistingFilePath = Annotated[Path, AfterValidator(must_exist_file)]


# These models are a subset of the promptfoo schema
# https://promptfoo.dev/config-schema.json


class ProviderConfig(BaseModel):
    id: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def convert_promptfoo_provider_format(cls, v: str) -> str:
        """
        Change provider id format to keep compatibility with promptfoo:
        - Remove the middle part of the id `provider:chat|completion:model`
        - Replace provider name
        """
        if "togetherai" in v:
            v = v.replace("togetherai", "together")
        if ":chat:" in v:
            return v.replace("chat:", "")
        if ":completion:" in v:
            raise ValueError("promptdev does not support 'completions' only chat")
        return v


class AssertionConfig(BaseModel):
    type: str
    value: Any | None = None
    threshold: float | None = None
    # provider for llm-rubric, etc


class TestConfig(BaseModel):
    description: str | None = None
    vars: dict[str, Any] | None = None  # TODO allow Path
    assert_: list[AssertionConfig | ExistingFilePath] = Field(  # TODO rename to assertion
        default_factory=list, alias="assert", description="AssertConfig or link to it"
    )
    metadata: dict[str, Any] | None = None
    ## threshold: float|None = None
    # provider for llm-rubric, etc


class PromptDevConfigOptions(BaseModel):
    cache_enabled: bool = Field(
        default=False, alias="cache", description="Whether or not to cache the results"
    )


class PromptDevConfig(BaseModel):
    description: str | None = None
    prompts: list[str | ExistingFilePath] = Field()
    providers: list[ProviderConfig]
    tests: list[TestConfig | ExistingFilePath] | ExistingFilePath | None = None
    default_test: TestConfig | None = Field(None, alias="defaultTest")
    options: PromptDevConfigOptions = Field(
        default_factory=PromptDevConfigOptions, alias="evaluateOptions"
    )

    @field_validator("providers", mode="before")
    @classmethod
    def convert_provider_shorthand(cls, v: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Converts string provider IDs (e.g., "ollama_gemma3_shorthand")
        into ProviderConfig dictionary representations before Pydantic parsing.
        """
        processed_providers = []
        for item in v:
            if isinstance(item, str):
                processed_providers.append({"id": item})
            else:
                processed_providers.append(item)
        return processed_providers
