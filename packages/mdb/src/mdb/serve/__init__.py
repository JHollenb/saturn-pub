"""``mdb serve``: a read-only HTTP view over a recorded MDB project.

The server has no runtime of its own. Views are pure functions over project reads; tensors leave
the server only as bounded reductions (``reduce``). Importing this package loads neither a model
framework nor FastAPI; ``create_app`` imports FastAPI when called.
"""
from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
