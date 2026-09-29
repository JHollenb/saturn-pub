"""Reversible PyTorch update intervals, adapted from Saturn's verified controller."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..values import clone
from ._checkpoint import (
    DeterministicCheckpointController,
    EvaluationMutationError,
    EvaluationScores,
    ImmutableDirectoryReceiptStore,
    PromotionObjective,
    PromotionPolicy,
    TrainingIntervalError,
    TrainingStateBoundary,
)


class TrainingTransaction(DeterministicCheckpointController):
    """Register caller-owned caches/hooks alongside the sampler/training cursor.

    Callbacks must expose restorable data, not arbitrary executable objects.
    Baseline establishment and interval execution use the original controller API.
    """

    def __init__(
        self,
        *,
        mutable: Mapping[str, tuple[Callable, Callable]] | None = None,
        cursor_capture: Callable,
        cursor_restore: Callable,
        **kwargs: Any,
    ):
        registered = dict(mutable or {})

        def capture() -> dict[str, Any]:
            return {
                "cursor": clone(cursor_capture()),
                "mutable": {name: clone(callbacks[0]()) for name, callbacks in registered.items()},
            }

        def restore(state: dict[str, Any]) -> None:
            cursor_restore(clone(state["cursor"]))
            for name, callbacks in registered.items():
                callbacks[1](clone(state["mutable"][name]))

        super().__init__(cursor_capture=capture, cursor_restore=restore, **kwargs)


__all__ = [
    "TrainingTransaction",
    "EvaluationScores",
    "PromotionObjective",
    "PromotionPolicy",
    "ImmutableDirectoryReceiptStore",
    "TrainingStateBoundary",
    "TrainingIntervalError",
    "EvaluationMutationError",
]
