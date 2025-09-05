"""
PromptDev - Python-native prompt evaluation tool using PydanticAI

A modern replacement for promptfoo that leverages PydanticAI's type-safe
agent framework and Pydantic's data validation capabilities.
"""

__version__ = "0.0.1"

from .agents.pydantic_agent import PromptDevAgent
from .config.models import PromptDevConfig
from .evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "PromptDevAgent",
    "PromptDevConfig",
]
