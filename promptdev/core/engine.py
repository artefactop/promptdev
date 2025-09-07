import asyncio
from typing import Any

from pydantic_ai import Agent

from promptdev.core.cache import CacheManager
from promptdev.core.models import EvaluationContext, ProviderFactory
from promptdev.core.reporting import EvaluationReport, EvaluationReports


class EvaluationEngine:
    def __init__(self, context: EvaluationContext):
        self.context = context
        self.cache = CacheManager()

    async def run_evaluation(self) -> EvaluationReports:
        """Run evaluation with pre-built context"""
        # Execute all providers concurrently
        reports = await asyncio.gather(
            *[self._evaluate_provider(provider) for provider in self.context.built_providers]
        )

        return EvaluationReports(evaluation_reports=reports)

    async def _evaluate_provider(self, built_provider: ProviderFactory) -> EvaluationReport:
        """Evaluate single provider with built components"""

        async def agent_task(inputs: Any) -> str:
            # Everything is pre-built - just format and run
            prompt_template = self.context.prompt_templates[0]
            # TODO we only support one prompt for now, using the first one

            system_prompt = prompt_template.system_prompt.format(**inputs)
            user_prompt = prompt_template.user_template.format(**inputs)

            # Create agent with pre-built components
            agent = Agent(
                model=built_provider.model,  # Pre-built
                system_prompt=system_prompt,  # Just formatted
                model_settings=built_provider.model_settings,  # Pre-built
                output_type=str,
            )
            # TODO: add cache
            # return await self._run_with_cache(agent, user_prompt, built_provider, inputs)
            agent_run_result = await agent.run(user_prompt)
            return agent_run_result.output

        # Use pre-built dataset
        return await self.context.dataset.evaluate(agent_task, name=built_provider.config.id)
