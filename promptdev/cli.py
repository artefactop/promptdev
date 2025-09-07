"""Command-line interface for PromptDev."""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from promptdev.evaluation.results import TestResult

from .cache import clear_cache, get_cache
from .config.loader import load_config
from .evaluation.runner import EvaluationRunner
from .utils.file_utils import resolve_file_path

console = Console()


def group_by_provider(tests: list[TestResult]) -> dict[str, list[TestResult]]:
    """Group test results by provider."""
    grouped = {}
    for test in tests:
        if test.provider_id not in grouped:
            grouped[test.provider_id] = []
        grouped[test.provider_id].append(test)
    return grouped


def _create_wrapped_panel(
    content: str, title: str = "", border_style: str = "dim", max_width: int | None = None
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
    # For output content, use a reasonable minimum width to prevent character-by-character wrapping
    return Panel(
        content,
        title=title,
        border_style=border_style,
        width=available_width,  # Always use full available width for proper text wrapping
        expand=False,  # Don't expand to full terminal width
        padding=(0, 1),  # Add some internal padding
    )


@click.group()
@click.version_option()
def cli():
    """PromptDev - Python-native prompt evaluation tool using PydanticAI."""


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
        elif not verbose:
            # Auto-detect: use progress bar if terminal supports it and not verbose mode
            from rich.console import Console

            auto_console = Console()
            progress_bar = auto_console.is_terminal

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
                prompt_path = resolve_file_path(prompt)
                if not prompt_path.exists():
                    console.print(f"[yellow]Warning: Prompt file not found: {prompt_path}[/yellow]")
                else:
                    console.print(f"✓ Prompt file exists: {prompt_path}")
                    #TODO: validate prompt schema

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
        console.print()
        console.print(Rule("Failed Tests Analysis", style="red"))

        failed_tests = [
            test
            for result in results.provider_results
            for test in result.test_results
            if not test.passed
        ]
        tree = Tree(f"failed tests ({len(failed_tests)} failures)")

        for provider_id, tests in group_by_provider(failed_tests).items():
            provider_node = tree.add(f"{provider_id} ({len(tests)} failures)")
            for test in tests:
                test_node = provider_node.add(f"{test.test_name} (score: {test.score:.2f})")
                if test.assertions:
                    for assertion in test.assertions:
                        if not assertion.get("passed", True):
                            assertion_type = assertion.get("type", "unknown")
                            details = assertion.get("details", "No details")
                            test_node.add(
                                f"{assertion_type} {details or 'Assertion details not available'}"
                            )
                elif test.error:
                    test_node.add(f"Error: {test.error}")
        console.print(tree)
        console.print()  # Add a blank line for spacing

        # Detailed report separator
        console.print(Rule("Detailed Failed Tests Report", style="yellow"))

        failed_tests = [
            test
            for result in results.provider_results
            for test in result.test_results
            if not test.passed
        ]

        for provider_id, tests in group_by_provider(failed_tests).items():
            provider_model = results.get_provider_model(provider_id)
            console.print()
            console.print(
                f"[red bold]┌─ Provider: {provider_id} ({provider_model}) - {len(tests)} failures[/red bold]"
            )
            console.print(f"[red bold]└{'─' * 60}[/red bold]")

            for i, test in enumerate(tests, 1):
                # Test separator with provider context
                console.print(f"\n  [red bold]Test {i}: {test.test_name}[/red bold]")
                console.print("     " + "─" * 50)

                # Colorful inputs display
                if test.variables:
                    console.print("\n     [yellow bold]📥 Inputs:[/yellow bold]")
                    for key, value in test.variables.items():
                        value_str = str(value)
                        if not verbose and len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        console.print(f"       • [cyan]{key}:[/cyan] {value_str}")

                # Show expected vs actual in clearly marked sections
                if test.expected:
                    console.print("\n     [green bold]✓ Expected:[/green bold]")
                    expected_str = str(test.expected)
                    if not verbose and len(expected_str) > 200:
                        expected_str = expected_str[:200] + "... [use --verbose for full output]"

                    expected_panel = _create_wrapped_panel(
                        expected_str, title="", border_style="green dim"
                    )
                    padded_panel = Padding(expected_panel, (0, 0, 0, 7))  # Left padding of 7 spaces
                    console.print(padded_panel)

                if test.output:
                    console.print("\n     [red bold]✗ Actual Output:[/red bold]")
                    output_str = str(test.output)
                    if not verbose and len(output_str) > 500:
                        output_str = output_str[:500] + "... [use --verbose for full output]"

                    output_panel = _create_wrapped_panel(
                        output_str, title="", border_style="red dim"
                    )
                    padded_panel = Padding(output_panel, (0, 0, 0, 7))  # Left padding of 7 spaces
                    console.print(padded_panel)

                # Display failed assertions
                console.print()
                console.print("     [bold red]Failed Assertion(s):[/bold red]")

                if test.assertions:
                    failed_assertions = [a for a in test.assertions if not a.get("passed", True)]
                    if failed_assertions:
                        for j, assertion in enumerate(failed_assertions, 1):
                            assertion_type = assertion.get("type", "python")
                            assertion_score = assertion.get("score", 0.0)
                            assertion_details = assertion.get("details", "No details")
                            detailed_results = assertion.get("detailed_results")

                            console.print(
                                f"       {j}. {assertion_type} (type: {assertion_type}, score: {assertion_score:.2f}):"
                            )

                            # Format detailed results for python evaluator
                            if detailed_results and isinstance(detailed_results, list):
                                for res in detailed_results:
                                    console.print(
                                        f"           {res.get('field')}: {res.get('actual')} != {res.get('expected')}"
                                    )
                            else:
                                console.print(f"           {assertion_details}")
                    else:
                        console.print(
                            "       ⚠️  No failed assertions found, but test failed (score < 1.0)."
                        )
                elif test.error:
                    console.print(f"       [bold red]Execution error:[/bold red] {test.error}")
                else:
                    console.print("       ⚠️  No assertion results available for this failed test.")

                console.print()
                console.print("     " + "─" * 50)
                console.print()

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
                    details_lines = [*details_lines[:5], "... (truncated) ...", *details_lines[-5:]]
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


if __name__ == "__main__":
    cli()
