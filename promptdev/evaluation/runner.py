"""Modern evaluation runner fully leveraging pydantic_evals."""

import asyncio
import time
from typing import Any, Final

from pydantic_evals import Dataset
from rich.console import Console
from rich.progress import Progress
from rich.theme import Theme
from ..agents.providers import get_provider_config
from ..agents.pydantic_agent import PromptDevAgent
from ..cache import get_cache
from ..config.models import PromptDevConfig, ProviderConfig
from ..utils.file_utils import resolve_file_path
from .dataset import PromptDevDataset as DatasetLoader
from .results import EvaluationResults, ProviderResult, TestResult

console = Console()


# Define a custom theme for log levels
custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red"})

MAX_CONCURRENCY:Final[int] = 5


class ModernEvaluationRunner:
    """Modern evaluation runner fully leveraging pydantic_evals."""

    def __init__(
        self,
        config: PromptDevConfig,
        verbose: bool = False,
        use_progress_bar: bool = False,
        max_concurrent: int = MAX_CONCURRENCY,
    ):
        """Initialize evaluation runner.

        Args:
            config: PromptDev configuration
            verbose: Enable verbose logging
            use_progress_bar: Use progress bar (ignored, pydantic_evals has built-in progress)
            max_concurrent: Maximum concurrent test executions per provider
        """
        self.config = config
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar  # For compatibility
        self.max_concurrent = max_concurrent

        # Initialize cache
        self.cache = get_cache()

        # Configure cache based on config
        if config.cache:
            self.cache.enabled = config.cache.enabled
        else:
            # If no cache config, keep default behavior (enabled by default)
            pass

    def _get_prompt_content(self, prompt_path: str) -> str:
        """Get prompt content for cache key generation."""
        try:
            if prompt_path.startswith("file://"):
                file_path = resolve_file_path(prompt_path)
                return file_path.read_text(encoding="utf-8")
            return prompt_path
        except Exception:
            return prompt_path

    async def run_evaluation(
        self, provider_override: str | None = None, parallel: bool = False
    ) -> EvaluationResults:
        """Run evaluation using pydantic_evals directly.

        Args:
            provider_override: Override to use specific provider only
            parallel: Run tests in parallel (ignored, pydantic_evals handles concurrency)

        Returns:
            Complete evaluation results
        """
        start_time = time.time()

        # Determine which providers to test
        providers_to_test = self.config.providers
        if provider_override:
            provider_config = get_provider_config(provider_override, self.config.providers)
            providers_to_test = [provider_config]

        if not self.verbose:
            total_cases = sum(
                len(DatasetLoader.from_config(test_config).cases)
                for test_config in self.config.tests
            )
            console.print(
                f"Evaluating {len(providers_to_test)} provider(s) with {total_cases} test(s) each..."
            )

        # Run evaluation for each provider - show all progress bars concurrently
        if not self.verbose and self.use_progress_bar:
            provider_results = await self._run_all_providers_with_concurrent_progress(
                providers_to_test
            )
        else:
            # Sequential execution for verbose mode or no progress bar
            provider_results = []
            for i, provider in enumerate(providers_to_test, 1):
                result = await self._evaluate_provider_with_pydantic_evals(provider)
                provider_results.append(result)

        if not self.verbose:
            console.print()  # New line after all providers

        total_time = (time.time() - start_time) * 1000

        return EvaluationResults(
            provider_results=provider_results,
            config_description=self.config.description,
            total_execution_time_ms=total_time,
            errors=[],
        )

    async def _evaluate_provider_with_pydantic_evals(
        self, provider_config: ProviderConfig
    ) -> ProviderResult:
        """Evaluate provider using pydantic_evals Dataset.evaluate() directly."""

        # Create agent
        # Use the first prompt (PromptDev supports multiple prompts but agent takes one)
        prompt_path_str = self.config.prompts[0] if self.config.prompts else "file://prompt.yaml"
        prompt_path = resolve_file_path(prompt_path_str)
        agent = PromptDevAgent(
            prompt_path=prompt_path,
            provider_config=provider_config,
        )

        # Load all test cases and convert to pydantic_evals format
        all_test_cases = []
        for test_config in self.config.tests:
            dataset_loader = DatasetLoader.from_config(test_config)
            # Convert Case objects back to dictionaries for evaluator processing
            for case in dataset_loader.cases:
                test_case_dict = {
                    "name": case.name,
                    "vars": case.inputs,
                    "expected": case.expected_output,
                    "metadata": case.metadata or {},
                    "assertions": [],
                }

                # Add default assertions from config if available
                if self.config.default_test and self.config.default_test.assert_:
                    test_case_dict["assertions"] = self.config.default_test.assert_

                all_test_cases.append(test_case_dict)

        # Create dataset with evaluators using the proper method
        dataset_wrapper = DatasetLoader.from_test_cases_with_evaluators(
            all_test_cases, self.config, verbose=self.verbose
        )
        dataset = dataset_wrapper.dataset

        if self.verbose:
            console.log(
                f"Debug: Created dataset with {len(dataset.cases)} cases and {len(dataset.evaluators)} evaluators",
                style="warning",
            )
            console.log(f"Debug: Test cases: {len(all_test_cases)}", style="warning")
            for i, test_case in enumerate(all_test_cases[:2]):  # Show first 2 test cases
                console.log(
                    f"Debug: Test case {i}: {test_case.get('name', 'unnamed')}",
                    style="warning",
                )
                console.log(
                    f"Debug: Assertions: {len(test_case.get('assertions', []))}",
                    style="warning",
                )
                for j, assertion in enumerate(test_case.get("assertions", [])):
                    console.log(f"Debug: Assertion {j}: {assertion}", style="warning")

        # Start timing
        import time

        start_time = time.time()

        # Define task function that captures raw output even on validation failures
        async def agent_task(inputs: Any) -> str:
            """Execute the agent for evaluation."""
            try:
                # Use the PromptDevAgent's run_test method with the full inputs dict
                result = await agent.run_test(inputs)
                return result
            except Exception as e:
                # If validation fails, try to get the raw output by creating a raw string agent
                try:
                    # Format the prompts manually using the same logic as PromptDevAgent
                    formatted_system = agent.system_prompt.format(**inputs)
                    formatted_prompt = agent.user_template.format(**inputs)

                    # Create a raw string agent to get unvalidated output
                    import pydantic_ai

                    model = agent._create_model(agent.provider_config)

                    raw_agent = pydantic_ai.Agent(
                        model=model,
                        system_prompt=formatted_system,
                        output_type=str,  # Get raw string output without validation
                    )

                    # Get model settings
                    model_settings = agent._get_run_settings()
                    if model_settings:
                        raw_result = await raw_agent.run(
                            formatted_prompt, model_settings=model_settings
                        )
                    else:
                        raw_result = await raw_agent.run(formatted_prompt)

                    return raw_result.output
                except Exception:
                    # If we can't get raw output, return the error message
                    return f"Error: {str(e)}"

        # Run evaluation with real-time progress if needed
        if not self.verbose and self.use_progress_bar:
            evaluation_result = await self._run_evaluation_with_live_progress(
                dataset_wrapper, agent_task, provider_config
            )
        else:
            evaluation_result = await dataset_wrapper.dataset.evaluate(
                agent_task,
                max_concurrency=self.config.max_concurrency or 5,
                progress=self.verbose,
            )

        elapsed_time = time.time() - start_time

        # Debug: check the actual type
        if self.verbose:
            console.log(
                f"Debug: evaluation_result type: {type(evaluation_result)}", style="warning"
            )
            console.log(
                f"Debug: has failures: {hasattr(evaluation_result, 'failures')}", style="warning"
            )
            console.log(
                f"Debug: Number of successful cases: {len(evaluation_result.cases)}",
                style="warning",
            )
            console.log(
                f"Debug: Number of failures: {len(evaluation_result.failures)}", style="warning"
            )
            if evaluation_result.cases:
                first_case = evaluation_result.cases[0]
                console.log(f"Debug: First case name: {first_case.name}", style="warning")
                console.log(f"Debug: First case scores: {first_case.scores}", style="warning")
                if first_case.scores:
                    for eval_name, eval_result in first_case.scores.items():
                        console.log(
                            f"Debug: Evaluator '{eval_name}': score={getattr(eval_result, 'value', 'N/A')}, reason={getattr(eval_result, 'reason', 'N/A')}",
                            style="warning",
                        )

        # Convert results to our format - process both successful and failed cases
        test_results = []

        # Process successful cases
        for case in evaluation_result.cases:
            # Calculate overall score from metrics/scores
            # Follow original runner logic: if any evaluator fails (score < 1.0), entire test fails
            if case.scores:
                # Extract numeric scores from EvaluationResult objects
                numeric_scores = []
                has_failures = False

                for score_value in case.scores.values():
                    if hasattr(score_value, "value") and score_value.value is not None:
                        score = score_value.value
                        numeric_scores.append(score)
                        if score < 1.0:
                            has_failures = True
                    elif hasattr(score_value, "score") and score_value.score is not None:
                        score = score_value.score
                        numeric_scores.append(score)
                        if score < 1.0:
                            has_failures = True
                    elif isinstance(score_value, (int | float)):
                        score = score_value
                        numeric_scores.append(score)
                        if score < 1.0:
                            has_failures = True

                # If any evaluator failed, return 0.0 to fail the test (original runner behavior)
                if has_failures:
                    avg_score = 0.0
                else:
                    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0
            else:
                avg_score = 1.0  # Default if no scores

            # Extract individual assertion results from case.scores
            assertions = []
            if case.scores:
                for evaluator_name, eval_result in case.scores.items():
                    score = eval_result.value if hasattr(eval_result, "value") else 0.0
                    reason = eval_result.reason if hasattr(eval_result, "reason") else None

                    # Get more detailed failure reason if available
                    details = reason
                    detailed_results_list = None
                    assertion_type = "unknown"
                    if hasattr(eval_result, "source") and hasattr(eval_result.source, "arguments"):
                        args = eval_result.source.arguments
                        if isinstance(args, dict):
                            details = args.get("last_failure_reason", reason)
                            detailed_results_list = args.get("last_detailed_results")
                            assertion_type = args.get("assertion_type", "unknown")
                        elif hasattr(args, "last_failure_reason"):
                            details = args.last_failure_reason
                            if hasattr(args, "last_detailed_results"):
                                detailed_results_list = args.last_detailed_results
                            if hasattr(args, "assertion_type"):
                                assertion_type = args.assertion_type

                    assertions.append(
                        {
                            "name": evaluator_name,
                            "type": assertion_type,
                            "score": score,
                            "passed": score == 1.0,  # Fail if score is not perfect
                            "details": details,
                            "detailed_results": detailed_results_list,
                        }
                    )

            test_result = TestResult(
                test_name=case.name or "unnamed_test",
                provider_id=provider_config.id,
                score=avg_score,
                passed=avg_score == 1.0,  # Fail if score is not perfect
                output=case.output,
                expected=case.expected_output,
                variables=case.inputs,
                execution_time_ms=case.task_duration * 1000 if case.task_duration else 0,
                error=None,  # Errors would be in evaluator_failures
                assertions=assertions,
            )
            test_results.append(test_result)

        # Process failed cases to ensure all test cases are reported
        for failure in evaluation_result.failures:
            test_result = TestResult(
                test_name=failure.name or "unnamed_test",
                provider_id=provider_config.id,
                score=0.0,  # Failed cases get 0 score
                passed=False,
                output=None,  # No output for failed cases
                expected=failure.expected_output,
                variables=failure.inputs,
                execution_time_ms=0,
                error=failure.error_message,
                assertions=[
                    {
                        "name": "evaluator",
                        "score": 0.0,
                        "passed": False,
                        "details": failure.error_message,
                    }
                ],
            )
            test_results.append(test_result)

        # Summary stats
        passed_tests = sum(1 for tr in test_results if tr.passed)
        avg_score = (
            sum(tr.score for tr in test_results) / len(test_results) if test_results else 0.0
        )

        # No need to show final status line since the progress bar already shows it

        return ProviderResult(
            provider_id=provider_config.id,
            test_results=test_results,
            model=provider_config.model,
            config=provider_config.config if hasattr(provider_config, "config") else None,
        )

    async def _run_all_providers_with_concurrent_progress(self, providers_to_test):
        """Run all providers concurrently with live progress bars for each."""
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            MofNCompleteColumn,
            TimeElapsedColumn,
        )
        import asyncio

        # Load datasets to get total test count
        total_tests_per_provider = sum(
            len(DatasetLoader.from_config(test_config).cases) for test_config in self.config.tests
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Create progress tasks for all providers
            provider_tasks = {}
            for provider in providers_to_test:
                task_id = progress.add_task(
                    f"[cyan]{provider.id}[/cyan]", total=total_tests_per_provider
                )
                provider_tasks[provider.id] = task_id

            # Run all providers concurrently
            async def run_provider_with_progress(provider):
                return await self._evaluate_provider_with_live_progress_updates(
                    provider, progress, provider_tasks[provider.id]
                )

            # Execute all providers concurrently
            provider_results = await asyncio.gather(
                *[run_provider_with_progress(provider) for provider in providers_to_test]
            )

        return provider_results

    async def _evaluate_provider_with_live_progress_updates(
        self, provider_config: ProviderConfig, progress: Progress, task_id: str
    ):
        """Evaluate provider with real-time progress updates during execution."""
        # Create agent
        prompt_path_str = self.config.prompts[0] if self.config.prompts else "file://prompt.yaml"
        prompt_path = resolve_file_path(prompt_path_str)
        agent = PromptDevAgent(
            prompt_path=prompt_path,
            provider_config=provider_config,
        )

        # Load all test cases
        all_test_cases = []
        for test_config in self.config.tests:
            dataset_loader = DatasetLoader.from_config(test_config)
            for case in dataset_loader.cases:
                test_case_dict = {
                    "name": case.name,
                    "vars": case.inputs,
                    "expected": case.expected_output,
                    "metadata": case.metadata or {},
                    "assertions": [],
                }
                if self.config.default_test and self.config.default_test.assert_:
                    test_case_dict["assertions"] = self.config.default_test.assert_
                all_test_cases.append(test_case_dict)

        # Create dataset with evaluators
        dataset_wrapper = DatasetLoader.from_test_cases_with_evaluators(
            all_test_cases, self.config, verbose=self.verbose
        )

        # Define agent task with caching
        async def agent_task(inputs):
            # Check cache first
            cache_key = None
            cached_output = None

            if self.cache.enabled:
                # Generate cache key
                prompt_content = self._get_prompt_content(agent.prompt_path)
                cache_key = self.cache.generate_cache_key(
                    model=provider_config.model,
                    prompt_content=prompt_content,
                    variables=inputs,
                    provider_config=provider_config.config,
                )

                # Try to get from cache
                cached_output = self.cache.get(cache_key)

                if cached_output is not None:
                    if self.verbose:
                        console.log("Debug: Cache hit! Using cached result", style="warning")
                    return cached_output
                else:
                    if self.verbose:
                        console.log("Debug: Cache miss, running agent...", style="warning")

            # Run agent if not cached
            try:
                result = await agent.run_test(inputs)

                # Store in cache
                if self.cache.enabled and cache_key:
                    # Get TTL from config if available
                    ttl = None
                    if (
                        self.config.cache
                        and hasattr(self.config.cache, "ttl")
                        and self.config.cache.ttl
                    ):
                        ttl = self.config.cache.ttl

                    self.cache.set(cache_key, result, ttl=ttl)
                    if self.verbose:
                        console.log("Debug: Result cached", style="warning")

                return result
            except Exception as e:
                try:
                    # Try to get raw output
                    formatted_system = agent.system_prompt.format(**inputs)
                    formatted_prompt = agent.user_template.format(**inputs)
                    import pydantic_ai

                    model = agent._create_model(agent.provider_config)
                    raw_agent = pydantic_ai.Agent(
                        model=model,
                        system_prompt=formatted_system,
                        output_type=str,
                    )
                    model_settings = agent._get_run_settings()
                    if model_settings:
                        raw_result = await raw_agent.run(
                            formatted_prompt, model_settings=model_settings
                        )
                    else:
                        raw_result = await raw_agent.run(formatted_prompt)

                    result = raw_result.output

                    # Store in cache even for fallback results
                    if self.cache.enabled and cache_key:
                        ttl = None
                        if (
                            self.config.cache
                            and hasattr(self.config.cache, "ttl")
                            and self.config.cache.ttl
                        ):
                            ttl = self.config.cache.ttl
                        self.cache.set(cache_key, result, ttl=ttl)
                        if self.verbose:
                            console.log("Debug: Fallback result cached", style="warning")

                    return result
                except Exception:
                    return f"Error: {str(e)}"

        # Create a wrapper for the agent task that updates progress
        completed_tests = 0
        total_tests = len(dataset_wrapper.dataset.cases)

        async def agent_task_with_progress(inputs):
            nonlocal completed_tests
            try:
                result = await agent_task(inputs)
                # Update progress after each test completes
                completed_tests += 1
                progress.update(task_id, completed=completed_tests)
                return result
            except Exception as e:
                # Still update progress even on failure
                completed_tests += 1
                progress.update(task_id, completed=completed_tests)
                raise e

        # Run the evaluation with progress updates
        evaluation_result = await dataset_wrapper.dataset.evaluate(
            agent_task_with_progress,
            max_concurrency=self.config.max_concurrency or MAX_CONCURRENCY,
            progress=False,  # We handle progress ourselves
        )

        # Determine overall status: ✓ if ALL pass, ✗ if ANY fail
        overall_passed = True

        # Check if any case failed
        for case in evaluation_result.cases:
            if hasattr(case, "scores") and case.scores:
                for eval_result in case.scores.values():
                    score = getattr(eval_result, "value", 0.0)
                    if score < 1.0:
                        overall_passed = False
                        break
            if not overall_passed:
                break

        # Check if there are any failures
        if evaluation_result.failures:
            overall_passed = False

        # Update progress bar with final status and color (like original)
        final_status = "✓" if overall_passed else "✗"
        status_color = "green" if overall_passed else "red"

        progress.update(
            task_id,
            description=f"[cyan]{provider_config.id}[/cyan] [{status_color}]{final_status}[/{status_color}]",
        )

        # Convert results to our format
        test_results = []
        for case in evaluation_result.cases:
            if case.scores:
                numeric_scores = []
                has_failures = False
                for score_value in case.scores.values():
                    if hasattr(score_value, "value") and score_value.value is not None:
                        score = score_value.value
                        numeric_scores.append(score)
                        if score < 1.0:
                            has_failures = True
                if has_failures:
                    avg_score = 0.0
                else:
                    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0
            else:
                avg_score = 1.0

            # Extract individual assertion results
            assertions = []
            if case.scores:
                for evaluator_name, eval_result in case.scores.items():
                    score = eval_result.value if hasattr(eval_result, "value") else 0.0
                    reason = eval_result.reason if hasattr(eval_result, "reason") else None
                    details = reason
                    detailed_results_list = None
                    assertion_type = "unknown"
                    if hasattr(eval_result, "source") and hasattr(eval_result.source, "arguments"):
                        args = eval_result.source.arguments
                        if isinstance(args, dict):
                            details = args.get("last_failure_reason", reason)
                            detailed_results_list = args.get("last_detailed_results")
                            assertion_type = args.get("assertion_type", "unknown")
                        elif hasattr(args, "last_failure_reason"):
                            details = args.last_failure_reason
                            if hasattr(args, "last_detailed_results"):
                                detailed_results_list = args.last_detailed_results
                            if hasattr(args, "assertion_type"):
                                assertion_type = args.assertion_type

                    assertions.append(
                        {
                            "name": evaluator_name,
                            "type": assertion_type,
                            "score": score,
                            "passed": score == 1.0,
                            "details": details,
                            "detailed_results": detailed_results_list,
                        }
                    )

            test_result = TestResult(
                test_name=case.name or "unnamed_test",
                provider_id=provider_config.id,
                score=avg_score,
                passed=avg_score == 1.0,
                output=case.output,
                expected=case.expected_output,
                variables=case.inputs,
                execution_time_ms=case.task_duration * 1000 if case.task_duration else 0,
                error=None,
                assertions=assertions,
            )
            test_results.append(test_result)

        # Process failed cases
        for failure in evaluation_result.failures:
            test_result = TestResult(
                test_name=failure.name or "unnamed_test",
                provider_id=provider_config.id,
                score=0.0,
                passed=False,
                output=None,
                expected=failure.expected_output,
                variables=failure.inputs,
                execution_time_ms=0,
                error=failure.error_message,
                assertions=[
                    {
                        "name": "evaluator",
                        "score": 0.0,
                        "passed": False,
                        "details": failure.error_message,
                    }
                ],
            )
            test_results.append(test_result)

        return ProviderResult(
            provider_id=provider_config.id,
            test_results=test_results,
            model=provider_config.model,
            config=provider_config.config if hasattr(provider_config, "config") else None,
        )


# Use the modern implementation as the main class
EvaluationRunner = ModernEvaluationRunner
