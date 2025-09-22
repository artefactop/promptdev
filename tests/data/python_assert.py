from typing import Any


def get_assert(output: str, context: dict) -> bool | float | dict[str, Any]:
    """Test assertion that checks if output contains 'success'."""
    return "success" in str(output).lower()
