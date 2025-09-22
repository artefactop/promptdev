from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals.evaluators.evaluator import EvaluationResult
from pydantic_evals.reporting import EvaluationReport, RenderValueConfig, ReportCase
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from promptdev.utils.format import render_duration

# Constants for configuration
DEFAULT_PASS_THRESHOLD = 0.5
FULL_PASS_THRESHOLD = 1.0
PARTIAL_PASS_THRESHOLD = 0.5
LOW_SCORE_THRESHOLD = 0.5
MAX_OUTPUT_LENGTH = 500
MAX_INPUT_LENGTH = 500
REASON_PREVIEW_LENGTH = 35


@dataclass
class CaseSummary:
    """Summary information for a test case."""

    pass_rate: float
    avg_score: float
    passed: int
    total: int
    duration: float


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length with ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


@dataclass
class EvaluationReports:
    """Complete evaluation report."""

    evaluation_reports: list[EvaluationReport]

    def _analyze_case_results(self, case: ReportCase) -> tuple[bool, float, int, int]:
        """Analyze case results and return (has_failure, avg_score, passed_count, total_count)."""
        has_failure = False
        score_values: list[float] = []
        scores_passed = 0
        # Scores (numeric)
        for result in case.scores.values():
            try:
                v = float(result.value)
            except (TypeError, ValueError):
                continue
            score_values.append(v)
            if v >= DEFAULT_PASS_THRESHOLD:
                scores_passed += 1
            else:
                has_failure = True
        # Assertions (boolean)
        assertion_total = len(case.assertions)
        assertion_passed = sum(1 for r in case.assertions.values() if bool(r.value))
        if assertion_passed < assertion_total:
            has_failure = True
        # Metrics
        avg_score = (sum(score_values) / len(score_values)) if score_values else 0.0
        passed_count = scores_passed + assertion_passed
        total_count = len(score_values) + assertion_total
        return has_failure, avg_score, passed_count, total_count

    def _get_color_for_score(self, score: float) -> str:
        """Get color style for a score/pass rate."""
        if score >= 0.8:
            return "green"
        if score >= PARTIAL_PASS_THRESHOLD:
            return "yellow"
        return "red"

    def _get_status_text(self, pass_rate: float) -> Text:
        """Get status text with appropriate color for a pass rate."""
        if pass_rate == FULL_PASS_THRESHOLD:
            return Text("PASS", style="green")
        if pass_rate >= PARTIAL_PASS_THRESHOLD:
            return Text("PARTIAL", style="yellow")
        return Text("FAIL", style="red")

    def _extract_short_reason(self, reason: str) -> str:
        """Extract a short, readable reason from an evaluation result reason."""
        return _truncate_text(reason.split("\n")[0], REASON_PREVIEW_LENGTH)

    def _print_evaluation_reason(self, console: Console, result: EvaluationResult) -> None:
        """Print the reason for an evaluation result failure."""
        if not result.reason or not (reason := str(result.reason).strip()):
            console.print("     [dim]No reason provided[/dim]")
            return

        for line in reason.split("\n"):
            if line.strip():
                console.print(f"     [dim]{line.strip()}[/dim]")

    def _get_case_summary(self, case: ReportCase) -> CaseSummary:
        """Extract summary info from a case using pydantic_evals ReportCase structure."""
        _has_failure, avg_score, passed_count, total_count = self._analyze_case_results(case)

        pass_rate = passed_count / total_count if total_count > 0 else 0.0

        return CaseSummary(
            pass_rate=pass_rate,
            avg_score=avg_score,
            passed=passed_count,
            total=total_count,
            duration=case.task_duration,
        )

    def _print_summary_statistics(self, console: Console):
        """Print overall evaluation statistics."""
        if not self.evaluation_reports:
            return

        console.print(Rule("📊 Evaluation Summary"))
        console.print()

        # Create summary table
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Model", style="dim")
        summary_table.add_column("Pass Rate", justify="center")
        summary_table.add_column("Avg Score", justify="center")
        summary_table.add_column("Checks", justify="center")
        summary_table.add_column("Avg Duration", justify="center")

        for report in self.evaluation_reports:
            total_assertions_passed = 0
            total_assertions = 0
            all_scores = []
            total_duration = 0
            for case in report.cases:
                summary = self._get_case_summary(case)
                total_assertions_passed += summary.passed
                total_assertions += summary.total
                all_scores.append(summary.avg_score)
                total_duration += summary.duration

            pass_rate = (
                (total_assertions_passed / total_assertions) if total_assertions > 0 else 0.0
            )
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            avg_duration = total_duration / len(report.cases) if report.cases else 0

            # Color code pass rate and average score
            pass_rate_text = Text(f"{pass_rate:.1%}", style=self._get_color_for_score(pass_rate))
            score_text = Text(f"{avg_score:.2f}", style=self._get_color_for_score(avg_score))

            summary_table.add_row(
                report.name,
                pass_rate_text,
                score_text,
                f"{total_assertions_passed}/{total_assertions}",
                render_duration(avg_duration),
            )

        console.print(summary_table)
        console.print()

    def _print_concise_results(self, console: Console):
        """Print a concise, readable results table."""
        console.print(Rule("📋 Detailed Results"))
        console.print()

        for report in self.evaluation_reports:
            console.print(f"[bold blue]{report.name}[/bold blue]")

            # Create results table
            results_table = Table(show_header=True, header_style="bold")
            results_table.add_column("Test Case", style="dim", width=25)
            results_table.add_column("Status", justify="center", width=12)
            results_table.add_column("Score", justify="center", width=10)
            results_table.add_column("Checks", justify="center", width=12)

            results_table.add_column("Duration", justify="right", width=10)

            for case in report.cases:
                summary = self._get_case_summary(case)

                # Determine status
                status = self._get_status_text(summary.pass_rate)

                # Color code score and assertions
                score_text = Text(
                    f"{summary.avg_score:.2f}", style=self._get_color_for_score(summary.avg_score)
                )

                passed_total_text = f"{summary.passed}/{summary.total}"
                pass_total_display = Text(
                    passed_total_text, style=self._get_color_for_score(summary.pass_rate)
                )

                # Format duration
                duration_text = render_duration(summary.duration)

                results_table.add_row(
                    case.name,
                    status,
                    score_text,
                    pass_total_display,
                    duration_text,
                )

            console.print(results_table)
            console.print()

    def _get_failed_cases(self) -> list[tuple[str, list[ReportCase]]]:
        """Get all failed test cases organized by provider using proper ReportCase structure."""
        failed_by_provider: list[tuple[str, list[ReportCase]]] = []

        for report in self.evaluation_reports:
            failed_cases: list[ReportCase] = []
            for case in report.cases:
                has_failure, _, _, _ = self._analyze_case_results(case)
                if has_failure:
                    failed_cases.append(case)

            if failed_cases:
                failed_by_provider.append((report.name, failed_cases))

        return failed_by_provider

    def _print_failed_tests_analysis(self, console: Console):
        """Print a tree structure analysis of failed tests."""
        failed_by_provider = self._get_failed_cases()

        if not failed_by_provider:
            return

        total_failures = sum(len(cases) for _, cases in failed_by_provider)

        console.print(Rule("Failed Tests Analysis", style="red"))
        console.print()

        # Create a tree structure using Rich's Tree component with improved colors
        tree = Tree(f"[bold red]failed tests ({total_failures} failures)[/bold red]")

        for provider_name, failed_cases in failed_by_provider:
            provider_branch = tree.add(
                f"[bold yellow]{provider_name}[/bold yellow] [dim]({len(failed_cases)} failures)[/dim]"
            )

            for case in failed_cases:
                summary = self._get_case_summary(case)
                case_branch = provider_branch.add(
                    f"[bold cyan]{case.name}[/bold cyan] [dim](score: {summary.avg_score:.2f})[/dim]"
                )

                # Add details for failing scores with score values and categorization
                for result in case.scores.values():
                    if result.value < DEFAULT_PASS_THRESHOLD and result.reason:
                        short_reason = self._extract_short_reason(str(result.reason))
                        # Show partial vs complete failures differently
                        if result.value > 0:
                            score_display = f"[yellow]score: {result.value:.2f}[/yellow]"
                            icon = "⚠️"
                        else:
                            score_display = f"[red]score: {result.value:.2f}[/red]"
                            icon = "❌"
                        case_branch.add(
                            f"{icon} [bright_black]{result.name}[/bright_black] ({score_display}) [dim]{short_reason}[/dim]"
                        )

                # Add details for failing assertions
                for result in case.assertions.values():
                    if not result.value and result.reason:
                        short_reason = self._extract_short_reason(str(result.reason))
                        case_branch.add(
                            f"❌ [bright_black]{result.name}[/bright_black] [dim red]{short_reason}[/dim red]"
                        )

        console.print(tree)
        console.print()

    def _print_detailed_failed_tests_report(self, console: Console):
        """Print detailed failed tests report with inputs, outputs, and assertion details."""
        failed_by_provider = self._get_failed_cases()

        if not failed_by_provider:
            return

        console.print(Rule("Detailed Failed Tests Report", style="red"))
        console.print()

        for provider_name, failed_cases in failed_by_provider:
            # Create provider panel
            provider_panel = Panel(
                f"[red]Provider: {provider_name} - {len(failed_cases)} failures[/red]",
                expand=False,
                border_style="red",
                padding=(0, 1),
            )
            console.print(provider_panel)

            for case in failed_cases:
                # Print case header
                console.print(f"\n[bold]{case.name}[/bold]")

                # Inputs section
                console.print("📥 [bold blue]Inputs:[/bold blue]")
                if case.inputs and isinstance(case.inputs, dict):
                    for key, value in case.inputs.items():
                        # For debugging purposes, show full values for important fields
                        # Only truncate very long values for readability
                        if isinstance(value, str):
                            value = _truncate_text(value, MAX_INPUT_LENGTH)
                        console.print(f"  • {key}: {value}")
                console.print()

                # Actual output section
                console.print("✗ [bold red]Actual Output:[/bold red]")
                if case.output:
                    output_text = str(case.output)
                    # Truncate very long outputs for readability
                    output_text = _truncate_text(output_text, MAX_OUTPUT_LENGTH)

                    actual_panel = Panel(
                        output_text, expand=False, border_style="red", padding=(0, 1)
                    )
                    console.print(actual_panel)
                console.print()

                # Failed evaluations section - distinguish between scores and assertions
                has_failed_scores = any(
                    result.value < DEFAULT_PASS_THRESHOLD for result in case.scores.values()
                )
                has_failed_assertions = any(not result.value for result in case.assertions.values())

                if has_failed_scores:
                    console.print("[bold red]Failed Score(s):[/bold red]")
                    score_count = 1
                    for result in case.scores.values():
                        if result.value < DEFAULT_PASS_THRESHOLD:
                            result_name = result.name
                            # Show partial scores more prominently
                            if result.value > 0:
                                score_style = "yellow"
                                score_label = f"[{score_style}]partial[/{score_style}]"
                            else:
                                score_style = "red"
                                score_label = f"[{score_style}]failed[/{score_style}]"

                            console.print(
                                f"  {score_count}. [bold]{result_name}[/bold] ({score_label}, score: [bold]{result.value:.2f}[/bold]):"
                            )
                            self._print_evaluation_reason(console, result)
                            score_count += 1
                    console.print()

                if has_failed_assertions:
                    console.print("[bold red]Failed Assertion(s):[/bold red]")
                    assertion_count = 1
                    for result in case.assertions.values():
                        if not result.value:
                            result_name = result.name
                            console.print(
                                f"  {assertion_count}. [bold]{result_name}[/bold] ([red]failed[/red]):"
                            )
                            self._print_evaluation_reason(console, result)
                            assertion_count += 1

                console.print(Rule(style="dim"))  # Separator between test cases

        console.print()

    def _print_reports(self, console: Console, verbose: bool = False):
        if verbose:
            # Use the detailed original format
            console.print(Rule("Detailed Evaluation Reports"))
            console.print()
            for report in self.evaluation_reports:

                def render_metadata(metadata: dict[str, Any]) -> str:
                    return f"{metadata}"

                metadata_render = RenderValueConfig(
                    value_formatter=render_metadata,
                )

                report.print(
                    metadata_config=metadata_render,
                    include_input=True,
                    include_output=True,
                    include_errors=True,
                    include_reasons=True,
                )
        else:
            # Use the new concise format in the requested order
            self._print_detailed_failed_tests_report(console)
            self._print_failed_tests_analysis(console)
            self._print_concise_results(console)
            self._print_summary_statistics(console)

    def print(self, width: int | None = None, verbose: bool = False):
        """Print the evaluation report."""
        console = Console(width=width)
        self._print_reports(console, verbose=verbose)

    def export_json(self, output_path: Path) -> None:
        """Export results to JSON file."""
        raise NotImplementedError()

    def export_html(self, output_path: Path) -> None:
        """Export results to HTML file."""
        raise NotImplementedError()
