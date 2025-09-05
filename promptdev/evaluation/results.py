"""Results management for PromptDev evaluations."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    provider_id: str
    score: float
    passed: bool
    output: Any
    expected: Any
    variables: dict[str, Any]
    error: str | None = None
    execution_time_ms: float | None = None
    assertions: list[dict[str, Any]] | None = None  # List of assertion configurations being checked
    failed_assertions: list[dict[str, Any]] | None = (
        None  # List of assertions that failed with failure reasons
    )


@dataclass
class ProviderResult:
    """Results for all tests run with a single provider."""

    provider_id: str
    test_results: list[TestResult]
    model: str | None = None
    config: dict[str, Any] | None = None

    @property
    def total_tests(self) -> int:
        """Total number of tests."""
        return len(self.test_results)

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        return sum(1 for r in self.test_results if r.passed)

    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        return self.total_tests - self.passed_tests

    @property
    def average_score(self) -> float:
        """Average score across all tests."""
        if not self.test_results:
            return 0.0
        return sum(r.score for r in self.test_results) / len(self.test_results)


@dataclass
class EvaluationError:
    """Information about an error that occurred during evaluation."""

    error_type: str  # "evaluator_creation", "test_execution", "provider_setup", etc.
    message: str
    details: str | None = None  # Full traceback or additional context
    context: dict[str, Any] | None = None  # Test case, assertion config, etc.


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    provider_results: list[ProviderResult]
    config_description: str | None = None
    total_execution_time_ms: float | None = None
    errors: list[EvaluationError] = None

    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []

    @property
    def total_tests(self) -> int:
        """Total number of tests across all providers."""
        return sum(pr.total_tests for pr in self.provider_results)

    @property
    def total_passed(self) -> int:
        """Total number of passed tests across all providers."""
        return sum(pr.passed_tests for pr in self.provider_results)

    @property
    def total_failed(self) -> int:
        """Total number of failed tests across all providers."""
        return sum(pr.failed_tests for pr in self.provider_results)

    @property
    def overall_score(self) -> float:
        """Overall score across all providers and tests."""
        if not self.provider_results:
            return 0.0

        all_scores = []
        for pr in self.provider_results:
            all_scores.extend(r.score for r in pr.test_results)

        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def export_json(self, output_path: Path) -> None:
        """Export results to JSON file."""
        data = {
            "summary": {
                "description": self.config_description,
                "total_tests": self.total_tests,
                "total_passed": self.total_passed,
                "total_failed": self.total_failed,
                "overall_score": self.overall_score,
                "execution_time_ms": self.total_execution_time_ms,
            },
            "providers": [],
        }

        for pr in self.provider_results:
            provider_data = {
                "provider_id": pr.provider_id,
                "summary": {
                    "total_tests": pr.total_tests,
                    "passed_tests": pr.passed_tests,
                    "failed_tests": pr.failed_tests,
                    "average_score": pr.average_score,
                },
                "tests": [],
            }

            for tr in pr.test_results:
                test_data = {
                    "test_name": tr.test_name,
                    "score": tr.score,
                    "passed": tr.passed,
                    "output": tr.output,
                    "expected": tr.expected,
                    "variables": tr.variables,
                    "error": tr.error,
                    "execution_time_ms": tr.execution_time_ms,
                }
                provider_data["tests"].append(test_data)

            data["providers"].append(provider_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_html(self, output_path: Path) -> None:
        """Export results to HTML file."""
        # HTML export feature planned for future release
        # For now, create a simple HTML representation
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PromptDev Evaluation Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .provider {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
                .test {{ margin: 10px 0; padding: 10px; background: #f9f9f9; }}
                .passed {{ border-left: 5px solid #4CAF50; }}
                .failed {{ border-left: 5px solid #f44336; }}
            </style>
        </head>
        <body>
            <h1>PromptDev Evaluation Results</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Description:</strong> {self.config_description or "N/A"}</p>
                <p><strong>Total Tests:</strong> {self.total_tests}</p>
                <p><strong>Passed:</strong> {self.total_passed}</p>
                <p><strong>Failed:</strong> {self.total_failed}</p>
                <p><strong>Overall Score:</strong> {self.overall_score:.2f}</p>
            </div>
        """

        for pr in self.provider_results:
            html_content += f"""
            <div class="provider">
                <h3>Provider: {pr.provider_id}</h3>
                <p>Tests: {pr.total_tests} | Passed: {pr.passed_tests} | Failed: {pr.failed_tests} | Score: {pr.average_score:.2f}</p>
            """

            for tr in pr.test_results:
                status_class = "passed" if tr.passed else "failed"
                html_content += f"""
                <div class="test {status_class}">
                    <strong>{tr.test_name}</strong> - Score: {tr.score:.2f}
                    {f"<br><em>Error: {tr.error}</em>" if tr.error else ""}
                </div>
                """

            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
