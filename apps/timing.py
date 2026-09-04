"""Small wall-clock progress timer shared by the MNIST applications."""

import time


def format_duration(seconds):
    """Format a duration as HH:MM:SS, retaining tenths below one minute."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Stopwatch:
    """Simple injectable wall-clock timer for epoch and experiment timing."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.started_at = clock()

    def elapsed(self):
        return self._clock() - self.started_at
