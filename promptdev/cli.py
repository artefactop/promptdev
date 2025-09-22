import asyncio
import time
from pathlib import Path

import click
from rich.console import Console

from promptdev.config.loader import load_config
from promptdev.core.cache import CacheManager
from promptdev.core.engine import EvaluationEngine
from promptdev.core.models import EvaluationContext
from promptdev.utils.format import render_duration

console = Console()


@click.group()
@click.version_option()
def cli():
    """Promptdev - Python-native prompt evaluation tool using PydanticAI."""


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console"]),
    default="console",
    help="Output format. JSON and HTML output coming soon.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--no-cache", is_flag=True, help="Disable caching for this evaluation")
def eval(
    config_file: Path,
    output: str,
    verbose: bool,
    no_cache: bool,
):
    """Run evaluation using configuration file.

    Examples:
        promptdev eval calendar_event_summary.yaml
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
            for provider in config.providers:
                console.print(f"  {provider.id}")

        # Handle cache disable flag
        if no_cache:
            # Disable caching by modifying config
            config.options.cache_enabled = False

        # Build evaluation context
        context = EvaluationContext.from_config(config)

        # Run evaluation
        runner = EvaluationEngine(context)
        start_time = time.time()
        reports = asyncio.run(runner.run_evaluation())
        end_time = time.time()
        total_duration = end_time - start_time

        # Output results
        if output == "console":
            reports.print(verbose=verbose)
        elif output == "json":
            reports.export_json(config_file.parent / f"{config_file.stem}_results.json")
            console.print(f"[green]Results exported to {config_file.stem}_results.json[/green]")
        elif output == "html":
            reports.export_html(config_file.parent / f"{config_file.stem}_results.html")
            console.print(f"[green]Results exported to {config_file.stem}_results.html[/green]")

        if runner.cache is not None:
            # TODO: show cache hits and misses
            console.print("[blue]Cache was enabled[/blue]")

        console.print(f"[green]Evaluation completed in {render_duration(total_duration)}[/green]")

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        console.print(f"[yellow]Configuration file: {config_file}[/yellow]")
        if verbose:
            console.print("[red]Full traceback:[/red]")
            console.print_exception()
        else:
            console.print("[dim]Use --verbose for full error details[/dim]")
        raise click.Abort() from e


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def validate(config_file: Path, verbose: bool):
    """Validate configuration file without running evaluation.

    Examples:
        promptdev validate calendar_event_summary.yaml
    """
    try:
        config = load_config(config_file)
        # Build context to check if everything is correct
        _ = EvaluationContext.from_config(config)

        console.print("[green]✓ Configuration file is valid[/green]")
        console.print(f"Description: {config.description or 'N/A'}")
        console.print(f"Providers: {len(config.providers)}")

    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed: {e}[/red]")
        if verbose:
            console.print("[red]Full traceback:[/red]")
            console.print_exception()
        else:
            console.print("[dim]Use --verbose for full error details[/dim]")
        raise click.Abort() from e


@cli.group()
def cache():
    """Cache management commands."""


@cache.command("clear")
def cache_clear():
    """Clear the evaluation cache."""
    try:
        CacheManager().cache.clear()
        console.print("[green]✓ Cache cleared successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing cache: {e}[/red]")
        raise click.Abort() from e


@cache.command("stats")
def cache_stats():
    """Show cache statistics."""
    try:
        cache_instance = CacheManager().cache
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
        raise click.Abort() from e


if __name__ == "__main__":
    cli()
