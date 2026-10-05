"""Small phase counters; wall durations include host dispatch, not CUDA kernel attribution."""
from contextlib import contextmanager
from time import perf_counter


class BenchMetrics:
    def __init__(self): self.phases = {}

    @contextmanager
    def measure(self, name):
        start = perf_counter()
        try: yield
        finally:
            row = self.phases.setdefault(name, {"calls": 0, "wall_seconds": 0.0})
            row["calls"] += 1; row["wall_seconds"] += perf_counter()-start

    def inspect(self):
        return {"phases": {k: dict(v) for k, v in self.phases.items()},
            "timing_scope": "since session open; nested host wall durations, CUDA dispatch may be asynchronous; weights excluded"}
