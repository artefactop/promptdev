"""Command-line interface for PromptDev."""

import asyncio
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from promptdev.evaluation.results import ProviderResult

from .cache import clear_cache, get_cache
from .config.loader import load_config
from .evaluation.runner import EvaluationRunner

console = Console()


def _create_wrapped_panel(
    content: str, title: str = "", border_style: str = "dim", max_width: int = None
) -> Panel:
    """Create a dynamically wrapped panel that adjusts to terminal width.

    Uses Rich's Panel.fit() for intelligent content-aware sizing and automatic wrapping.

    Args:
        content: Text content to display in the panel
        title: Optional title for the panel
        border_style: Style for the panel border
        max_width: Maximum width for the panel (defaults to terminal width - indentation)

    Returns:
        Panel: A Rich Panel with dynamic wrapping
    """
    # Get terminal width and calculate available space (accounting for indentation)
    terminal_width = console.size.width if hasattr(console, "size") else 80
    available_width = max_width or max(50, terminal_width - 10)  # Leave margin for indentation

    # Create the panel with Rich's intelligent fitting
    # Panel.fit() automatically sizes to content but respects width constraints
    panel = Panel(
        content,
        title=title,
        border_style=border_style,
        width=min(available_width, len(content.split("\n")[0]) + 4)
        if len(content) < 100
        else available_width,
        expand=False,  # Don't expand to full terminal width
        padding=(0, 1),  # Add some internal padding
    )

    return panel


def _create_failed_tests_tree(provider_results: dict[str, "ProviderResult"]) -> Tree:
    """Create a hierarchical tree view of failed tests using Rich.Tree."""

    # Count total failures (handle None values)
    total_failures = sum(
        len([t for t in result.test_results if not t.passed])
        for result in provider_results.values()
        if result and result.test_results
    )

    if total_failures == 0:
        return Tree("🎉 All tests passed!")

    # Create root tree with balanced styling
    root = Tree(f"[bold white]failed tests[/bold white] [dim]({total_failures} failures)[/dim]")

    for provider_name, result in provider_results.items():
        # Skip None results or results without test_results
        if not result or not result.test_results:
            continue

        failed_tests = [t for t in result.test_results if not t.passed]

        if not failed_tests:
            continue

        # Add provider branch with subtle color
        provider_branch = root.add(
            f"[cyan]{provider_name}[/cyan] [dim]({len(failed_tests)} failures)[/dim]"
        )

        for test in failed_tests:
            # Add test case branch with color for visibility
            test_branch = provider_branch.add(
                f"[red]{test.test_name}[/red] [dim](score: {test.score:.2f})[/dim]"
            )

            # Add failed assertions as sub-branches
            if hasattr(test, "failed_assertions") and test.failed_assertions:
                for assertion in test.failed_assertions:
                    assertion_name = assertion.get("assertion_name", "Unknown")
                    assertion.get("assertion_type", "unknown")
                    assertion.get("score", 0.0)
                    reason = assertion.get("failure_reason", "No reason provided")

                    # Handle None reasons and truncate long reasons
                    if reason is None:
                        reason = "No reason provided"
                    elif len(reason) > 60:
                        reason = reason[:57] + "..."

                    test_branch.add(f"[red]{assertion_name}[/red] [dim]{reason}[/dim]")
            else:
                # Fallback for tests without detailed assertion info
                test_branch.add("⚠️  Assertion details not available")

    return root


def _create_enhanced_progress() -> Progress:
    """Create an enhanced progress bar with multiple columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} tests"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )


@click.group()
@click.version_option()
def cli():
    """PromptDev - Python-native prompt evaluation tool using PydanticAI."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json", "html"]),
    default="console",
    help="Output format",
)
@click.option("--provider", "-p", help="Override provider for evaluation")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--parallel", is_flag=True, help="Run tests in parallel")
@click.option("--progress-bar", is_flag=True, help="Force progress bar (auto-detected by default)")
@click.option("--no-progress-bar", is_flag=True, help="Force dots instead of progress bar")
@click.option("--no-cache", is_flag=True, help="Disable caching for this evaluation")
@click.option(
    "--max-concurrent",
    type=int,
    default=5,
    help="Maximum concurrent test executions per provider (default: 5)",
)
def eval(
    config_file: Path,
    output: str,
    provider: str | None,
    verbose: bool,
    parallel: bool,
    progress_bar: bool,
    no_progress_bar: bool,
    no_cache: bool,
    max_concurrent: int,
):
    """Run evaluation using configuration file.

    Examples:
        promptdev eval calendar_event_summary.yaml
        promptdev eval calendar_event_summary.yaml --provider pydantic-ai:openai
        promptdev eval calendar_event_summary.yaml --output json --verbose
        promptdev eval calendar_event_summary.yaml --no-cache
    """
    try:
        # Load configuration
        config = load_config(config_file)

        if verbose:
            console.print(f"[green]Loaded configuration from {config_file}[/green]")
            console.print(f"Description: {config.description or 'N/A'}")
            console.print(f"Providers: {len(config.providers)}")
            console.print(f"Tests: {len(config.tests)}")

        # Handle progress bar selection
        if no_progress_bar:
            # Explicitly disabled
            progress_bar = False
        elif not progress_bar:
            # Auto-detect: use progress bar if terminal supports it and not verbose mode
            from rich.console import Console

            auto_console = Console()
            progress_bar = auto_console.is_terminal and not verbose

        # Handle cache disable flag
        if no_cache:
            # Disable caching by modifying config
            if config.cache:
                config.cache.enabled = False
            else:
                from .config.models import CacheConfig

                config.cache = CacheConfig(enabled=False)

        # Run evaluation
        runner = EvaluationRunner(
            config, verbose=verbose, use_progress_bar=progress_bar, max_concurrent=max_concurrent
        )
        results = asyncio.run(runner.run_evaluation(provider_override=provider, parallel=parallel))

        # Output results
        if output == "console":
            _print_results_console(results, verbose)
        elif output == "json":
            results.export_json(config_file.parent / f"{config_file.stem}_results.json")
            console.print(f"[green]Results exported to {config_file.stem}_results.json[/green]")
        elif output == "html":
            results.export_html(config_file.parent / f"{config_file.stem}_results.html")
            console.print(f"[green]Results exported to {config_file.stem}_results.html[/green]")

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        console.print(f"[yellow]Configuration file: {config_file}[/yellow]")
        if provider:
            console.print(f"[yellow]Provider override: {provider}[/yellow]")
        if verbose:
            console.print("[red]Full traceback:[/red]")
            console.print_exception()
        else:
            console.print("[dim]Use --verbose for full error details[/dim]")
        raise click.Abort() from e


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory for reports"
)
def redteam(config_file: Path, output_dir: Path | None):
    """Run red team security evaluation.

    Examples:
        promptdev redteam calendar_event_summary.yaml
        promptdev redteam calendar_event_summary.yaml --output-dir ./reports
    """
    # Red teaming functionality planned for future release
    console.print("[yellow]Red team evaluation coming soon![/yellow]")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def validate(config_file: Path):
    """Validate configuration file without running evaluation.

    Examples:
        promptdev validate calendar_event_summary.yaml
    """
    try:
        config = load_config(config_file)
        console.print("[green]✓ Configuration file is valid[/green]")
        console.print(f"Description: {config.description or 'N/A'}")
        console.print(f"Providers: {len(config.providers)}")
        console.print(f"Tests: {len(config.tests)}")

        # Validate prompts exist
        for prompt in config.prompts:
            if isinstance(prompt, str) and prompt.startswith("file://"):
                prompt_path = Path(prompt[7:])
                if not prompt_path.exists():
                    console.print(f"[yellow]Warning: Prompt file not found: {prompt_path}[/yellow]")
                else:
                    console.print(f"✓ Prompt file exists: {prompt_path}")

    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed: {e}[/red]")
        raise click.Abort() from e


@cli.command()
def init():
    """Initialize a new PromptDev project.

    Creates a sample configuration file and directory structure.
    """
    # Project initialization feature planned for future release
    console.print("[yellow]Project initialization coming soon![/yellow]")


@cli.group()
def cache():
    """Cache management commands."""
    pass


@cache.command("clear")
def cache_clear():
    """Clear the evaluation cache."""
    try:
        clear_cache()
        console.print("[green]✓ Cache cleared successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing cache: {e}[/red]")


@cache.command("stats")
def cache_stats():
    """Show cache statistics."""
    try:
        cache_instance = get_cache()
        stats = cache_instance.stats()

        console.print("\n[bold]Cache Statistics[/bold]")
        console.print(f"Enabled: [{'green' if stats['enabled'] else 'red'}]{stats['enabled']}[/]")
        console.print(f"Cached items: [cyan]{stats['size']}[/cyan]")

        if stats.get("cache_file"):
            console.print(f"Cache file: [dim]{stats['cache_file']}[/dim]")

        if stats.get("cache_file_exists"):
            file_size = stats.get("cache_file_size_bytes", 0)
            size_str = f"{file_size / 1024:.1f} KB" if file_size > 1024 else f"{file_size} bytes"
            console.print(f"Cache file size: [cyan]{size_str}[/cyan]")

        if stats["size"] > 0:
            console.print(f"\nFirst {min(5, len(stats['keys']))} cache keys:")
            for i, key in enumerate(stats["keys"][:5], 1):
                console.print(f"  {i}. {key[:64]}{'...' if len(key) > 64 else ''}")

    except Exception as e:
        console.print(f"[red]Error getting cache stats: {e}[/red]")


def _print_results_console(results, verbose: bool = False):
    """Print evaluation results to console with provider comparison."""

    # If multiple providers, show comparison view first
    if len(results.provider_results) > 1:
        _print_provider_comparison(results)
        console.print("\n" + "=" * 80 + "\n")

    # Show individual provider details
    for provider_result in results.provider_results:
        provider_title = provider_result.provider_id
        if provider_result.model:
            provider_title += f" ({provider_result.model})"

        # Calculate pass/fail stats for this provider
        total_tests = len(provider_result.test_results)
        passed_tests = sum(1 for test in provider_result.test_results if test.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        console.print(f"\n[bold]Evaluation Summary: {provider_title}[/bold]")
        console.print(
            f"[green]{passed_tests} passed[/green], [red]{failed_tests} failed[/red] ([dim]{pass_rate:.1f}% pass rate[/dim])"
        )

        # Create detailed table with case-by-case breakdown
        table = Table()
        table.add_column("Case ID", style="cyan")
        table.add_column("Inputs", style="dim")
        table.add_column("Outputs", style="dim")
        table.add_column("Scores", justify="center")
        table.add_column("Assertions", justify="center")

        total_score = 0.0
        total_evaluators = 0
        total_passed = 0

        for test_result in provider_result.test_results:
            # Format inputs (variables)
            inputs_str = ""
            if test_result.variables:
                key_vars = []
                for k, v in test_result.variables.items():
                    if k not in [
                        "expected_name",
                        "expected_out_of_office",
                        "expected_event_type",
                    ]:  # Skip expected vars
                        v_str = str(v)
                        if len(v_str) > 30:
                            v_str = v_str[:30] + "..."
                        key_vars.append(f"{k}: {v_str}")
                inputs_str = ", ".join(key_vars[:2])  # Show max 2 variables
                if len(key_vars) > 2:
                    inputs_str += "..."

            # Format outputs (truncate if too long)
            output_str = str(test_result.output) if test_result.output else ""
            if len(output_str) > 50:
                output_str = output_str[:50] + "..."

            # Format scores
            score_str = f"Score: {test_result.score:.2f}"
            assertion_status = "✔" if test_result.passed else "✗"

            table.add_row(
                test_result.test_name, inputs_str, output_str, score_str, assertion_status
            )

            total_score += test_result.score
            total_evaluators += 1
            if test_result.passed:
                total_passed += 1

        # Add averages row
        avg_score = total_score / total_evaluators if total_evaluators > 0 else 0.0
        pass_rate = (total_passed / total_evaluators * 100) if total_evaluators > 0 else 0.0

        table.add_row(
            "[bold]Averages[/bold]",
            "",
            "",
            f"[bold]Score: {avg_score:.2f}[/bold]",
            f"[bold]{pass_rate:.1f}% ✔[/bold]",
        )

        console.print(table)

    # Show failed tests with enhanced Rich features
    if any(
        any(not test.passed for test in result.test_results) for result in results.provider_results
    ):
        # Clean separator
        console.print()
        console.print(Rule("Failed Tests Analysis", style="red"))

        # Create hierarchical tree view of failures
        provider_results_dict = {result.provider_id: result for result in results.provider_results}
        failed_tree = _create_failed_tests_tree(provider_results_dict)
        console.print()
        console.print(failed_tree)
        console.print()

        # Detailed report separator
        console.print(Rule("Detailed Failed Tests Report", style="yellow"))
        _print_failed_tests_by_provider(results, verbose)

    # Show error summary if there were any errors during evaluation
    if results.errors:
        console.print()
        console.print(Rule("⚠️  Evaluation Errors Summary", style="yellow"))
        console.print(f"[red]{len(results.errors)} error(s) occurred during evaluation:[/red]")

        for i, error in enumerate(results.errors, 1):
            console.print(f"\n[red bold]🔥 Error {i}: {error.error_type}[/red bold]")
            console.print(f"[red]Message: {error.message}[/red]")

            if verbose and error.details:
                console.print("[yellow]Details:[/yellow]")
                details_lines = error.details.split("\n")
                if len(details_lines) > 10:
                    details_lines = details_lines[:5] + ["... (truncated) ..."] + details_lines[-5:]
                console.print("\n".join(details_lines))

            if error.context:
                console.print("[yellow]Context:[/yellow]")
                for key, value in error.context.items():
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    console.print(f"  {key}: {value}")

        if not verbose:
            console.print("[dim]Use --verbose for full error details and tracebacks[/dim]")

        # Final separator
        console.print("\n" + "=" * 80)
    else:
        # Add final separator even when no errors
        console.print("\n" + "=" * 80)
        console.print("\n[green bold]✅ Evaluation completed successfully![/green bold]")
        console.print("=" * 80)


def _print_provider_comparison(results):
    """Print a comparison table showing all providers side by side."""
    console.print("\n[bold]🔍 Provider Comparison[/bold]")

    # Create comparison table
    table = Table()
    table.add_column("Test Case", style="cyan")
    table.add_column("Input", style="dim")

    # Add a column for each provider
    for provider_result in results.provider_results:
        provider_name = provider_result.provider_id
        if provider_result.model:
            # Show just the model name, not the full path
            model_name = (
                provider_result.model.split(":")[-1]
                if ":" in provider_result.model
                else provider_result.model
            )
            provider_name += f"\n({model_name})"
        table.add_column(provider_name, justify="center")

    # Get all test cases (assuming all providers run the same tests)
    if results.provider_results:
        test_cases = results.provider_results[0].test_results

        for i, test_case in enumerate(test_cases):
            test_name = test_case.test_name

            # Format input
            input_str = ""
            if test_case.variables:
                key_vars = []
                for k, v in test_case.variables.items():
                    if k not in ["expected_name", "expected_out_of_office", "expected_event_type"]:
                        v_str = str(v)
                        if len(v_str) > 40:
                            v_str = v_str[:40] + "..."
                        key_vars.append(f"{k}: {v_str}")
                input_str = "\n".join(key_vars[:1])  # Show main input

            # Collect scores for this test across all providers
            row_data = [test_name, input_str]

            best_score = -1
            best_providers = []

            # Get scores from each provider for this test
            for provider_result in results.provider_results:
                if i < len(provider_result.test_results):
                    test_result = provider_result.test_results[i]
                    score = test_result.score
                    status = "✔" if test_result.passed else "✗"

                    # Track best score
                    if score > best_score:
                        best_score = score
                        best_providers = [provider_result.provider_id]
                    elif score == best_score:
                        best_providers.append(provider_result.provider_id)

                    score_text = f"{status} {score:.2f}"
                    row_data.append(score_text)
                else:
                    row_data.append("N/A")

            # Highlight best performer(s)
            for j, provider_result in enumerate(results.provider_results):
                # Make the best score bold and green
                if (
                    provider_result.provider_id in best_providers
                    and best_score > 0
                    and i < len(provider_result.test_results)
                ):
                    test_result = provider_result.test_results[i]
                    status = "✔" if test_result.passed else "✗"
                    row_data[j + 2] = f"[bold green]{status} {test_result.score:.2f}[/bold green]"

            table.add_row(*row_data)

        # Add summary row
        summary_row = ["[bold]Overall[/bold]", ""]
        for provider_result in results.provider_results:
            avg_score = provider_result.average_score
            pass_rate = (
                (provider_result.passed_tests / provider_result.total_tests * 100)
                if provider_result.total_tests > 0
                else 0.0
            )
            summary_row.append(f"[bold]{avg_score:.2f} ({pass_rate:.1f}%)[/bold]")

        table.add_row(*summary_row)

    console.print(table)


def _print_failed_tests_by_provider(results, verbose: bool = False):
    """Print failed tests grouped by provider using hierarchical tree format."""
    # Collect all failed tests
    all_failed = {}
    total_failures = 0
    for provider_result in results.provider_results:
        failed_tests = [tr for tr in provider_result.test_results if not tr.passed]
        if failed_tests:
            all_failed[provider_result.provider_id] = failed_tests
            total_failures += len(failed_tests)

    if not all_failed:
        console.print("\n[green]🎉 All tests passed across all providers![/green]")
        return

    # Detailed report section
    console.print("[bold white]failures:[/bold white]")
    console.print()

    for provider_id, failed_tests in all_failed.items():
        # Find the provider result to get model info
        provider_result = next(
            (p for p in results.provider_results if p.provider_id == provider_id), None
        )
        model_info = (
            provider_result.model if provider_result and provider_result.model else "unknown model"
        )

        # Always show provider header with model info
        console.print(
            f"\n[red bold]┌─ Provider: {provider_id} ({model_info}) - {len(failed_tests)} failures[/red bold]"
        )
        console.print(f"[red bold]└{'─' * 60}[/red bold]")

        for i, test_result in enumerate(failed_tests, 1):
            # Test separator with provider context
            console.print(f"\n  [red bold]Test {i}: {test_result.test_name}[/red bold]")
            console.print("     " + "─" * 50)

            # Colorful inputs display
            if test_result.variables:
                console.print("     [yellow bold]📥 Inputs:[/yellow bold]")
                for key, value in test_result.variables.items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    console.print(f"       • [cyan]{key}:[/cyan] {value_str}")

            # Show expected vs actual in clearly marked sections
            if test_result.expected:
                console.print("\n     [green bold]✓ Expected:[/green bold]")
                expected_str = str(test_result.expected)
                if not verbose and len(expected_str) > 200:
                    expected_str = expected_str[:200] + "... [use --verbose for full output]"

                # Use our dynamic wrapping with green styling
                expected_panel = _create_wrapped_panel(
                    expected_str, title="", border_style="green dim"
                )
                padded_panel = Padding(expected_panel, (0, 0, 0, 7))  # Left padding of 7 spaces
                console.print(padded_panel)

            if test_result.output:
                console.print("\n     [red bold]✗ Actual Output:[/red bold]")
                output_str = str(test_result.output)
                if not verbose and len(output_str) > 500:
                    output_str = output_str[:500] + "... [use --verbose for full output]"

                # Use our dynamic wrapping with red styling
                output_panel = _create_wrapped_panel(output_str, title="", border_style="red dim")
                padded_panel = Padding(output_panel, (0, 0, 0, 7))  # Left padding of 7 spaces
                console.print(padded_panel)

            # Show failed assertions with detailed results (restored)
            if hasattr(test_result, "failed_assertions") and test_result.failed_assertions:
                console.print("\n     [red bold]Failed Assertion(s):[/red bold]")
                assertion_counter = 1

                for failed_assertion in test_result.failed_assertions:
                    evaluator_name = failed_assertion.get("evaluator_name", "Unknown")
                    assertion_name = failed_assertion.get("assertion_name", evaluator_name)
                    assertion_type = failed_assertion.get("assertion_type", "unknown")
                    score = failed_assertion.get("score", 0.0)
                    detailed_results = failed_assertion.get("detailed_results")

                    if evaluator_name == "PythonAssertionEvaluator" and detailed_results:
                        # Show assertion name with type, then detailed field-by-field results
                        console.print(
                            f"       {assertion_counter}. [red]{assertion_name}[/red] (type: {assertion_type}, score: {score:.2f}):"
                        )
                        for detail in detailed_results:
                            field_name = detail.get("field", "Unknown Field")
                            actual = detail.get("actual")
                            expected = detail.get("expected")
                            console.print(f"           {field_name}: {actual} != {expected}")
                        assertion_counter += 1
                    else:
                        # Show assertion name for other evaluators
                        failure_reason = failed_assertion.get("failure_reason", "Unknown reason")
                        console.print(
                            f"       {assertion_counter}. [red]{assertion_name}[/red] (type: {assertion_type}, score: {score:.2f}): {failure_reason}"
                        )
                        assertion_counter += 1

            elif test_result.assertions:
                console.print("\n     [yellow bold]⚠️  All Assertions Failed:[/yellow bold]")
                for j, assertion in enumerate(test_result.assertions, 1):
                    assertion_desc = _format_assertion_description(assertion)
                    console.print(f"       {j}. {assertion_desc}")

            if test_result.error:
                console.print("\n     [red bold]💥 Error:[/red bold]")
                console.print(f"       {test_result.error}")

            # End of test separator with proper spacing
            console.print(f"\n     [dim]{'─' * 50}[/dim]")


def _format_assertion_description(assertion: dict[str, Any]) -> str:
    """Format assertion information for display."""
    assertion_type = assertion.get("type", "unknown")

    if assertion_type == "python":
        # Python assertion
        if assertion.get("value"):
            file_path = assertion["value"]
            if file_path.startswith("file://"):
                file_path = file_path[7:]
            return f"Python assertion from {file_path}"
        return "Python assertion"

    elif assertion_type == "contains-json":
        # JSON schema validation
        if assertion.get("template_ref"):
            return f"JSON schema validation (template: {assertion['template_ref']})"
        return "JSON schema validation"

    elif assertion_type == "llm-rubric":
        # LLM rubric evaluation
        rubric = assertion.get("rubric", assertion.get("value", ""))
        if rubric:
            rubric_preview = rubric[:50] + "..." if len(rubric) > 50 else rubric
            return f"LLM rubric: {rubric_preview}"
        return "LLM rubric evaluation"

    elif assertion_type == "g-eval":
        # G-Eval evaluation
        criteria = assertion.get("value", "")
        if criteria:
            criteria_preview = criteria[:50] + "..." if len(criteria) > 50 else criteria
            return f"G-Eval: {criteria_preview}"
        return "G-Eval evaluation"

    elif assertion_type == "contains":
        # Contains string check
        value = assertion.get("value", "")
        return f"Contains: '{value}'"

    elif assertion_type == "equals":
        # Exact match check
        value = assertion.get("value", "")
        return f"Equals: '{value}'"

    elif assertion.get("ref"):
        # Template reference
        ref = assertion["ref"]
        if ref.startswith("#/assertionTemplates/"):
            template_name = ref[len("#/assertionTemplates/") :]
            return f"Template: {template_name}"
        return f"Reference: {ref}"

    else:
        # Generic assertion
        if assertion.get("value"):
            value_str = str(assertion["value"])[:50]
            if len(str(assertion["value"])) > 50:
                value_str += "..."
            return f"{assertion_type}: {value_str}"
        return f"{assertion_type} assertion"


if __name__ == "__main__":
    cli()
