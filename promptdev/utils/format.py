def render_duration(seconds: float) -> str:
    """Format a duration given in seconds to a string.

    If the duration is less than 1 millisecond, show microseconds.
    If it's less than one second, show milliseconds.
    If it's less than one minute, show seconds.
    Otherwise, show minutes and seconds.
    """
    if seconds == 0:
        return "0s"
    precision = 1
    if (abs_seconds := abs(seconds)) < 1e-3:
        value = seconds * 1_000_000
        unit = "µs"
        if abs(value) >= 1:
            precision = 0
    elif abs_seconds < 1:
        value = seconds * 1_000
        unit = "ms"
    elif abs_seconds < 60:
        value = seconds
        unit = "s"
    else:
        minutes = int(seconds / 60)
        sec = int(seconds % 60)
        return f"{minutes}m{sec}s"

    return f"{value:,.{precision}f}{unit}"
