from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals.reporting import EvaluationReport, RenderValueConfig
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table


def _create_wrapped_panel(
    console: Console,
    content: str,
    title: str = "",
    border_style: str = "dim",
    max_width: int | None = None,
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


@dataclass
class EvaluationReports:
    """Complete evaluation report."""

    evaluation_reports: list[EvaluationReport]

    def _print_report_comparison(self, console: Console) -> None:
        console.print(Rule("Provider Comparison"))
        console.print()

        # Create comparison table
        table = Table()
        table.add_column("Case ID", style="bold")
        # Add a column for each provider
        for provider_report in self.evaluation_reports:
            table.add_column(provider_report.name, justify="center")

        grouped_cases = {}
        for report in self.evaluation_reports:
            for case in report.cases:
                if case.name in grouped_cases:
                    grouped_cases[case.name].update({report.name: case})
                else:
                    grouped_cases[case.name] = {report.name: case}

        for case_name, case_by_report in grouped_cases.items():
            row_data = [case_name]
            for report_name, case in case_by_report.items():
                # TODO build a aggregated by case
                console.log(case)
                row_data.append(f"TBD:{report_name}")

            table.add_row(*row_data)
        console.print(table)

    def _print_reports(self, console: Console, baseline: EvaluationReport | None = None):
        console.print(Rule("Evaluation Summaries"))
        console.print()
        for report in self.evaluation_reports:

            def render_metadata(metadata: dict[str, Any]) -> str:
                return f"{metadata}"

            metadata_render = RenderValueConfig(
                value_formatter=render_metadata,
            )

            # todo color the scores according the threshold
            report.print(
                baseline=baseline,
                metadata_config=metadata_render,
                include_input=True,
                include_output=True,
                include_errors=True,
                include_reasons=True,
            )
            # console.print(report.failures_table())

        if len(self.evaluation_reports) > 0:
            # self._print_report_comparison(console)
            console.print(Rule())

    def _print_detailed_failed_report_cases(self, console: Console) -> None:
        """Print the evaluation report."""
        for report in self.evaluation_reports:
            failed_cases = []
            for case in report.cases:
                not_failed = True
                evaluation_results = list(case.scores.values())
                evaluation_results.extend(list(case.assertions.values()))
                for evaluation_result in evaluation_results:
                    if isinstance(evaluation_result.value, bool):
                        not_failed &= evaluation_result.value
                    elif isinstance(evaluation_result.value, float):
                        not_failed &= evaluation_result.value >= 0.5  # TODO configured threshold

                if not not_failed:
                    failed_cases.append(case)
                    # TODO bonito del to
            if failed_cases:
                console.print(Rule(f"Error cases for {report.name}"), style="red")
            for failed_case in failed_cases:
                console.print(failed_case)

    def print(self, width: int | None = None):
        """Print the evaluation report."""
        console = Console(width=width)
        # self._print_detailed_failed_report_cases(console)
        self._print_reports(console)

    def export_json(self, output_path: Path) -> None:
        """Export results to JSON file."""
        raise NotImplementedError()

    def export_html(self, output_path: Path) -> None:
        """Export results to HTML file."""
        raise NotImplementedError()
