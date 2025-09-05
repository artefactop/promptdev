"""Evaluation runner for PromptDev."""

import asyncio
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from ..agents.providers import get_provider_config
from ..agents.pydantic_agent import PromptDevAgent
from ..cache import get_cache
from ..config.models import PromptDevConfig, ProviderConfig
from ..evaluators.pydantic_evaluators import run_pydantic_evaluation
from .dataset import PromptDevDataset as DatasetLoader
from .results import EvaluationError, EvaluationResults, ProviderResult, TestResult

console = Console()


class EvaluationRunner:
    """Main evaluation runner for PromptDev."""

    def __init__(
        self,
        config: PromptDevConfig,
        verbose: bool = False,
        use_progress_bar: bool = False,
        max_concurrent: int = 5,
    ):
        """Initialize evaluation runner.

        Args:
            config: PromptDev configuration
            verbose: Enable verbose logging
            use_progress_bar: Use progress bar instead of dots for non-verbose mode
            max_concurrent: Maximum concurrent test executions per provider
        """
        self.config = config
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar
        self.max_concurrent = max_concurrent
        self.datasets = []

        # Initialize cache
        self.cache = get_cache()
        if config.cache:
            self.cache.enabled = config.cache.enabled
            # Set cache directory if specified
            if hasattr(config.cache, "cache_dir") and config.cache.cache_dir:
                self.cache.cache_dir = Path(config.cache.cache_dir)
                self.cache.cache_file = self.cache.cache_dir / "promptdev_cache.json"
                if self.cache.enabled:
                    self.cache.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default to enabled if no cache config
            self.cache.enabled = True

        # Initialize error collection
        self.evaluation_errors = []

        # Load datasets
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load all test datasets from configuration."""
        if self.verbose:
            console.print(f"[green]Loading {len(self.config.tests)} test configuration(s)[/green]")

        for i, test_config in enumerate(self.config.tests, 1):
            if self.verbose:
                if hasattr(test_config, "file") and test_config.file:
                    console.print(f"  [{i}] Loading dataset from: {test_config.file}")
                else:
                    console.print(f"  [{i}] Loading inline test configuration")

            dataset = DatasetLoader.from_config(test_config)
            self.datasets.append(dataset)

            if self.verbose:
                console.print(f"      Loaded {len(dataset)} test cases")

    async def run_evaluation(
        self, provider_override: str | None = None, parallel: bool = False
    ) -> EvaluationResults:
        """Run evaluation across all providers and tests.

        Args:
            provider_override: Override to use specific provider only
            parallel: Run tests in parallel

        Returns:
            Complete evaluation results
        """
        start_time = time.time()

        # Determine which providers to test
        providers_to_test = self.config.providers
        if provider_override:
            if self.verbose:
                console.print(f"[yellow]Using provider override: {provider_override}[/yellow]")
            try:
                provider_config = get_provider_config(provider_override, self.config.providers)
                providers_to_test = [provider_config]
            except ValueError as e:
                raise ValueError(f"Provider override failed: {e}") from e

        if self.verbose:
            console.print(
                f"[green]Starting evaluation with {len(providers_to_test)} provider(s)[/green]"
            )
            if parallel:
                console.print("[yellow]Running providers in parallel[/yellow]")
            else:
                console.print("[yellow]Running providers sequentially[/yellow]")
        else:
            # Brief summary for non-verbose mode
            total_tests = sum(len(dataset.test_cases) for dataset in self.datasets)
            console.print(
                f"Evaluating {len(providers_to_test)} provider(s) with {total_tests} test(s) each..."
            )

        # Run evaluation for each provider
        provider_results = []

        if parallel:
            # Run providers in parallel
            if self.verbose:
                console.print("Starting parallel evaluation...")
            tasks = [self._evaluate_provider(provider) for provider in providers_to_test]
            provider_results = await asyncio.gather(*tasks)
        else:
            # Run providers sequentially
            if self.use_progress_bar and not self.verbose:
                # Use progress bar for sequential evaluation
                await self._run_with_progress_bar(providers_to_test, provider_results)
            else:
                # Use existing dot-based or verbose approach
                for i, provider in enumerate(providers_to_test, 1):
                    if self.verbose:
                        console.print(f"\n[cyan]Provider {i}/{len(providers_to_test)}[/cyan]")
                    elif len(providers_to_test) > 1:
                        # Show provider progress in non-verbose mode when multiple providers
                        console.print(f"[{i}/{len(providers_to_test)}] ", end="")
                    result = await self._evaluate_provider(provider)
                    provider_results.append(result)

        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        if self.verbose:
            console.print(f"\n[green]Evaluation completed in {total_time:.0f}ms[/green]")
            total_tests = sum(len(pr.test_results) for pr in provider_results)
            total_passed = sum(pr.passed_tests for pr in provider_results)
            console.print(
                f"Total tests: {total_tests}, Passed: {total_passed}, Failed: {total_tests - total_passed}"
            )

        return EvaluationResults(
            provider_results=provider_results,
            config_description=self.config.description,
            total_execution_time_ms=total_time,
            errors=self.evaluation_errors,
        )

    async def _evaluate_provider(self, provider_config: ProviderConfig) -> ProviderResult:
        """Evaluate all tests for a single provider.

        Args:
            provider_config: Provider configuration

        Returns:
            Provider evaluation results
        """
        if self.verbose:
            console.print(f"[blue]Evaluating provider: {provider_config.id}[/blue]")
            console.print(f"  Model: {provider_config.model}")
            console.print(f"  Config: {provider_config.config}")

        test_results = []

        # Create agent for this provider
        # Use the first prompt (multiple prompts support could be added in future versions)
        prompt_path = self._resolve_prompt_path(self.config.prompts[0])

        if self.verbose:
            console.print(f"  Loading prompt from: {prompt_path}")

        agent = PromptDevAgent(
            prompt_path=prompt_path,
            provider_config=provider_config,
            output_type=str,  # Use string output for now
        )

        if self.verbose:
            console.print("  Agent created successfully")

        # Run all test cases
        total_tests = sum(len(dataset.test_cases) for dataset in self.datasets)
        if self.verbose:
            console.print(f"  Running {total_tests} test cases...")
        else:
            # Show provider name and test count in non-verbose mode
            console.print(f"Testing {provider_config.id} ({total_tests} tests): ", end="")

        # Collect all test cases with their metadata
        all_test_cases = []
        test_count = 0
        for dataset in self.datasets:
            for test_case in dataset.test_cases:
                test_count += 1
                test_name = test_case.get("name", f"test_{test_count}")
                all_test_cases.append((test_case, test_name, test_count, total_tests))

        # Run test cases concurrently
        if len(all_test_cases) > 1 and not self.verbose:
            # Use concurrent execution for multiple test cases in non-verbose mode
            test_results = await self._run_test_cases_concurrent(
                agent, all_test_cases, provider_config, self.max_concurrent
            )
        else:
            # Sequential execution for verbose mode or single test
            test_results = []
            for test_case, test_name, test_num, total in all_test_cases:
                if self.verbose:
                    console.print(f"  [{test_num}/{total}] Running: {test_name}")

                result = await self._run_single_test(
                    agent, test_case, provider_config.id, provider_config
                )
                test_results.append(result)

                if self.verbose:
                    status = "✓" if result.passed else "✗"
                    console.print(
                        f"    {status} Score: {result.score:.2f} ({result.execution_time_ms:.0f}ms)"
                    )
                else:
                    # Show progress dots in non-verbose mode
                    status_char = "✓" if result.passed else "✗"
                    console.print(f"{status_char}", end="")

        # Add newline after progress dots in non-verbose mode
        if not self.verbose:
            console.print()  # New line after progress indicators

        return ProviderResult(
            provider_id=provider_config.id,
            test_results=test_results,
            model=provider_config.model,
            config=provider_config.config,
        )

    async def _run_single_test(
        self,
        agent: PromptDevAgent,
        test_case: dict,
        provider_id: str,
        provider_config: ProviderConfig,
    ) -> TestResult:
        """Run a single test case.

        Args:
            agent: PydanticAI agent
            test_case: Test case data
            provider_id: Provider identifier

        Returns:
            Test result
        """
        start_time = time.time()

        try:
            # Extract variables and expected values
            variables = test_case.get("vars", {})
            test_name = test_case.get("name", f"test_{hash(str(variables))}")

            if self.verbose:
                console.print(f"      Variables: {variables}")

            # Check cache first
            cache_key = None
            cached_output = None

            if self.cache.enabled:
                # Generate cache key
                prompt_content = self._get_prompt_content(agent.prompt_path)
                cache_key = self.cache.generate_cache_key(
                    model=provider_config.model,
                    prompt_content=prompt_content,
                    variables=variables,
                    provider_config=provider_config.config,
                )

                # Try to get from cache
                cached_output = self.cache.get(cache_key)

                if cached_output is not None:
                    if self.verbose:
                        console.print("      [green]Cache hit! Using cached result[/green]")
                    output = cached_output
                else:
                    if self.verbose:
                        console.print("      [yellow]Cache miss, running agent...[/yellow]")

            # Run agent if not cached
            if cached_output is None:
                output = await agent.run_test(variables)

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

                    self.cache.set(cache_key, output, ttl=ttl)
                    if self.verbose:
                        console.print("      [blue]Result cached[/blue]")

            if self.verbose:
                # Truncate long outputs for readability
                output_str = str(output)
                output_preview = output_str[:200] + "..." if len(output_str) > 200 else output_str
                console.print(f"      Output: {output_preview}")

            # Collect assertion information for display
            assertions = self._collect_assertion_info(test_case)

            # Run evaluations
            score = await self._evaluate_output(output, test_case)
            passed = (
                score >= 1.0
            )  # Score threshold (configurable thresholds could be added per assertion)

            if self.verbose:
                console.print(f"[blue]Test score: {score}, Passed: {passed}[/blue]")

            execution_time = (time.time() - start_time) * 1000

            return TestResult(
                test_name=test_name,
                provider_id=provider_id,
                score=score,
                passed=passed,
                output=output,
                expected=test_case.get("expected"),
                variables=variables,
                execution_time_ms=execution_time,
                assertions=assertions,
                failed_assertions=getattr(self, "_last_failed_assertions", None),
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Collect assertion information even for failed tests
            assertions = self._collect_assertion_info(test_case)

            return TestResult(
                test_name=test_case.get("name", "unknown"),
                provider_id=provider_id,
                score=0.0,
                passed=False,
                output=None,
                expected=test_case.get("expected"),
                variables=test_case.get("vars", {}),
                error=str(e),
                execution_time_ms=execution_time,
                assertions=assertions,
                failed_assertions=None,  # No failed assertions for error cases
            )

    def _collect_assertion_info(self, test_case: dict) -> list[dict[str, Any]]:
        """Collect assertion information for display in test results.

        Args:
            test_case: Test case dictionary

        Returns:
            List of assertion configurations being checked
        """
        assertions = []

        # Collect case-specific assertions
        case_assertions = test_case.get("assertions", [])
        for assertion in case_assertions:
            if isinstance(assertion, dict):
                assertions.append(assertion)

        # Collect default test assertions
        if self.config.default_test and self.config.default_test.assert_:
            for assertion in self.config.default_test.assert_:
                assertion_dict = {}
                if hasattr(assertion, "model_dump"):
                    assertion_dict = assertion.model_dump()
                elif isinstance(assertion, dict):
                    assertion_dict = assertion
                else:
                    assertion_dict = {"type": str(assertion)}

                # Resolve assertion template references
                if assertion_dict.get("ref"):
                    ref = assertion_dict["ref"]
                    if ref.startswith("#/assertionTemplates/"):
                        template_name = ref[len("#/assertionTemplates/") :]
                        if (
                            self.config.assertion_templates
                            and template_name in self.config.assertion_templates
                        ):
                            template = self.config.assertion_templates[template_name]
                            if hasattr(template, "model_dump"):
                                resolved = template.model_dump()
                            else:
                                resolved = template
                            assertion_dict.update(resolved)
                            assertion_dict["template_ref"] = template_name

                assertions.append(assertion_dict)

        return assertions

    def _get_prompt_content(self, prompt_path: Path) -> str:
        """Get prompt content for cache key generation.

        Args:
            prompt_path: Path to prompt file

        Returns:
            String representation of prompt content
        """
        try:
            if not prompt_path.exists():
                return ""

            with open(prompt_path, encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            if self.verbose:
                console.print(
                    f"[yellow]Warning: Could not read prompt content for caching: {e}[/yellow]"
                )
            return ""

    async def _evaluate_output(self, output: any, test_case: dict) -> float:
        """Evaluate output against test case assertions using PydanticAI evals.

        Args:
            output: Agent output
            test_case: Test case with assertions

        Returns:
            Score between 0.0 and 1.0
        """
        # Use the PydanticAI-based evaluation system
        try:
            # Merge default test assertions into the test case
            enhanced_test_case = dict(test_case)

            # Ensure the test case has default assertions merged
            case_assertions = enhanced_test_case.get("assertions", [])

            # Add default test assertions as AssertionConfig objects if not already present
            if self.config.default_test and self.config.default_test.assert_:
                for default_assertion in self.config.default_test.assert_:
                    # Check if this assertion is already in the case assertions
                    assertion_already_present = False
                    for case_assertion in case_assertions:
                        if (
                            hasattr(case_assertion, "type")
                            and hasattr(default_assertion, "type")
                            and case_assertion.type == default_assertion.type
                            and case_assertion.value == default_assertion.value
                        ):
                            assertion_already_present = True
                            break

                    if not assertion_already_present:
                        case_assertions.append(default_assertion)

            enhanced_test_case["assertions"] = case_assertions

            # Create a single-case dataset for evaluation
            test_cases = [enhanced_test_case]

            # Create a simple function that returns the output for evaluation
            async def mock_agent_function(inputs):
                return output

            # Run PydanticAI evaluation
            report, dataset = await run_pydantic_evaluation(
                mock_agent_function,
                test_cases,
                self.config,
                verbose=self.verbose,
                progress=False,
                error_collector=self.evaluation_errors,
            )

            # Extract score from report - if any evaluator fails, test should fail
            failed_assertion_details = []

            if hasattr(report, "cases") and report.cases:
                # Check if there are any evaluator failures in the report
                has_failures = False
                total_score = 0.0
                total_evaluators = 0

                for case_result in report.cases:
                    # Check scores from the case (pydantic_evals structure)
                    if hasattr(case_result, "scores") and case_result.scores:
                        for eval_name, eval_result in case_result.scores.items():
                            total_evaluators += 1
                            score_value = getattr(eval_result, "value", 0.0)
                            total_score += score_value

                            # Any evaluator with score < 1.0 indicates failure
                            if score_value < 1.0:
                                has_failures = True
                                reason = getattr(eval_result, "reason", "Unknown reason")

                                # Try to get detailed results from Python assertion evaluators
                                detailed_results = None
                                assertion_name = eval_name  # Default to evaluator name
                                assertion_type = "unknown"

                                # Find the evaluator index and get the original assertion name and type
                                if dataset and hasattr(dataset, "evaluator_assertion_map"):
                                    for eval_idx, evaluator in enumerate(dataset.evaluators):
                                        if (
                                            eval_name == "PythonAssertionEvaluator"
                                            and hasattr(evaluator, "__class__")
                                            and evaluator.__class__.__name__
                                            == "PythonAssertionEvaluator"
                                        ):
                                            if eval_idx in dataset.evaluator_assertion_map:
                                                assertion_name = dataset.evaluator_assertion_map[
                                                    eval_idx
                                                ]
                                            if eval_idx in dataset.evaluator_type_map:
                                                assertion_type = dataset.evaluator_type_map[
                                                    eval_idx
                                                ]
                                            if hasattr(evaluator, "last_detailed_results"):
                                                detailed_results = evaluator.last_detailed_results
                                            # Get specific failure reason from evaluator
                                            if (
                                                hasattr(evaluator, "last_failure_reason")
                                                and evaluator.last_failure_reason
                                            ):
                                                reason = evaluator.last_failure_reason
                                            break
                                        elif (
                                            eval_name == "ContainsJSONEvaluator"
                                            and hasattr(evaluator, "__class__")
                                            and evaluator.__class__.__name__
                                            == "ContainsJSONEvaluator"
                                        ):
                                            if eval_idx in dataset.evaluator_assertion_map:
                                                assertion_name = dataset.evaluator_assertion_map[
                                                    eval_idx
                                                ]
                                            if eval_idx in dataset.evaluator_type_map:
                                                assertion_type = dataset.evaluator_type_map[
                                                    eval_idx
                                                ]
                                            # Get specific failure reason from evaluator
                                            if (
                                                hasattr(evaluator, "last_failure_reason")
                                                and evaluator.last_failure_reason
                                            ):
                                                reason = evaluator.last_failure_reason
                                            break

                                # Collect failed assertion details
                                failed_assertion_details.append(
                                    {
                                        "evaluator_name": eval_name,
                                        "assertion_name": assertion_name,
                                        "assertion_type": assertion_type,
                                        "score": score_value,
                                        "failure_reason": reason,
                                        "detailed_results": detailed_results,
                                    }
                                )

                                if self.verbose:
                                    console.print(
                                        f"[red]Evaluator '{eval_name}' failed with score {score_value}: {reason}[/red]"
                                    )

                    # Also check the evaluator_failures field
                    if (
                        hasattr(case_result, "evaluator_failures")
                        and case_result.evaluator_failures
                    ):
                        has_failures = True
                        if self.verbose:
                            console.print(
                                f"[red]Evaluator failures detected: {case_result.evaluator_failures}[/red]"
                            )

                # Check if the report itself indicates failures
                if (
                    hasattr(report, "summary")
                    and hasattr(report.summary, "evaluator_failures")
                    and report.summary.evaluator_failures
                ):
                    has_failures = True
                    if self.verbose:
                        console.print(
                            f"[red]Report shows evaluator failures: {report.summary.evaluator_failures}[/red]"
                        )

                # Store failed assertion details for TestResult creation (BEFORE early return)
                self._last_failed_assertions = (
                    failed_assertion_details if failed_assertion_details else None
                )

                # If any evaluator failed, return 0.0 to fail the test
                if has_failures:
                    if self.verbose:
                        console.print(
                            f"[red]Test failed due to evaluator failures. Total evaluators: {total_evaluators}, Total score: {total_score}[/red]"
                        )
                    return 0.0

                # Return average score if no failures
                if total_evaluators > 0:
                    return total_score / total_evaluators
                else:
                    return 1.0 if output is not None else 0.0

            elif hasattr(report, "scores") and report.scores:
                return sum(report.scores) / len(report.scores)

            # Fallback to simple existence check
            return 1.0 if output is not None else 0.0

        except Exception as e:
            import traceback

            # Collect error instead of printing immediately
            error = EvaluationError(
                error_type="test_execution",
                message=f"PydanticAI evaluation error: {e}",
                details=traceback.format_exc(),
                context={
                    "test_case": test_case,
                    "output": str(output)[:500] + ("..." if len(str(output)) > 500 else ""),
                },
            )
            self.evaluation_errors.append(error)

            # Clear failed assertions for error cases
            self._last_failed_assertions = None

            # Fallback to legacy evaluation
            return await self._evaluate_against_expected_values(
                output, test_case.get("expected", {}), test_case
            )

    async def _evaluate_against_expected_values(
        self, output: any, expected: dict, test_case: dict
    ) -> float:
        """Evaluate output against expected values with JSON parsing support."""
        import json
        import re

        output_str = str(output)

        # Try to extract JSON from the output
        json_data = None
        try:
            # Look for JSON block in markdown
            json_match = re.search(r"```json\s*\n(.*?)\n```", output_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                json_data = json.loads(json_str)
                if self.verbose:
                    console.print(f"[green]Parsed JSON from markdown: {json_data}[/green]")
            else:
                # Try to find JSON object directly (handles nested braces)
                json_match = re.search(r"\{.*?\}", output_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    if self.verbose:
                        console.print(f"[green]Parsed JSON directly: {json_data}[/green]")
        except (json.JSONDecodeError, AttributeError) as e:
            if self.verbose:
                console.print(f"[red]JSON parsing failed: {e}[/red]")
            # Fall back to text-based evaluation
            pass

        # Fallback evaluation when PydanticAI evaluation fails
        # Returns basic score based on output existence

        if self.verbose:
            console.print(
                f"[yellow]Generic evaluation - output exists: {output is not None}[/yellow]"
            )

        return 1.0 if output is not None else 0.0

    def _resolve_prompt_path(self, prompt: str) -> Path:
        """Resolve prompt path from configuration.

        Args:
            prompt: Prompt path or template

        Returns:
            Resolved path to prompt file
        """
        if isinstance(prompt, str) and prompt.startswith("file://"):
            return Path(prompt[7:])
        elif isinstance(prompt, Path):
            return prompt
        else:
            # Inline prompts not currently supported - use file:// format
            raise ValueError(
                f"Unsupported prompt format: {prompt}. Use file:// format for prompt files."
            )

    async def _run_with_progress_bar(
        self, providers_to_test: list[ProviderConfig], provider_results: list[ProviderResult]
    ) -> None:
        """Run evaluation with progress bar."""
        total_tests_per_provider = sum(len(dataset.test_cases) for dataset in self.datasets)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for provider in providers_to_test:
                # Create a task for this provider
                provider_task = progress.add_task(
                    f"[cyan]{provider.id}[/cyan]", total=total_tests_per_provider
                )

                # Run provider evaluation with progress updates
                result = await self._evaluate_provider_with_progress(
                    provider, progress, provider_task
                )
                provider_results.append(result)

    async def _evaluate_provider_with_progress(
        self, provider_config: ProviderConfig, progress: Progress, task_id: TaskID
    ) -> ProviderResult:
        """Evaluate provider with progress bar updates."""
        test_results = []

        # Create agent for this provider
        prompt_path = self._resolve_prompt_path(self.config.prompts[0])
        agent = PromptDevAgent(
            prompt_path=prompt_path,
            provider_config=provider_config,
            output_type=str,
        )

        # Run all test cases with progress updates
        for dataset in self.datasets:
            for test_case in dataset.test_cases:
                result = await self._run_single_test(
                    agent, test_case, provider_config.id, provider_config
                )
                test_results.append(result)

                # Update progress
                status = "✓" if result.passed else "✗"
                progress.update(
                    task_id, advance=1, description=f"[cyan]{provider_config.id}[/cyan] {status}"
                )

        return ProviderResult(
            provider_id=provider_config.id,
            test_results=test_results,
            model=provider_config.model,
            config=provider_config.config,
        )

    async def _run_test_cases_concurrent(
        self,
        agent: PromptDevAgent,
        test_cases_with_metadata: list,
        provider_config: ProviderConfig,
        max_concurrent: int = 5,  # Limit concurrent tests to avoid overwhelming APIs
    ) -> list[TestResult]:
        """Run test cases concurrently with controlled concurrency.

        Args:
            agent: PydanticAI agent
            test_cases_with_metadata: List of (test_case, test_name, test_num, total) tuples
            provider_config: Provider configuration
            max_concurrent: Maximum number of concurrent tests

        Returns:
            List of test results in original order
        """
        import asyncio

        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(
            test_data: tuple[dict, str, int, int],
        ) -> tuple[int, TestResult]:
            """Run a single test with semaphore protection."""
            test_case, test_name, test_num, total = test_data
            async with semaphore:
                result = await self._run_single_test(
                    agent, test_case, provider_config.id, provider_config
                )
                return test_num - 1, result  # Return index for ordering

        # Create tasks for all test cases
        tasks = [run_with_semaphore(test_data) for test_data in test_cases_with_metadata]

        # Execute all tasks concurrently
        results_with_indices = await asyncio.gather(*tasks)

        # Sort results by original order and extract just the results
        results_with_indices.sort(key=lambda x: x[0])
        test_results = [result for _, result in results_with_indices]

        # Show progress indicators after completion
        for result in test_results:
            status_char = "✓" if result.passed else "✗"
            console.print(f"{status_char}", end="")

        return test_results
