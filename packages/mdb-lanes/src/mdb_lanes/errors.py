"""Dependency-free errors shared by execution and offline verification."""


class BenchError(RuntimeError):
    """Unsupported capability, incompatible state, or incomplete transaction."""
