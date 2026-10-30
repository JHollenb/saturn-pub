"""Explicit session-exclusive frozen-weight ownership for resident J readers.

The caller must route every weight/module/config/tokenizer mutation through
``bench.weight_mutation()`` (including .data writes and custom kernels). The
guards below diagnose ordinary accidental changes; they do not certify arbitrary
external memory immutability. Uncontrolled mutable consumers use strict readers.
"""
from __future__ import annotations

from .errors import BenchError


def _signature(model):
    rows = []
    for name, value in model.named_parameters():
        try:
            version = value._version
        except RuntimeError as exc:
            raise BenchError("frozen reader binding requires version-tracked resident weights") from exc
        rows.append((name, id(value), value.data_ptr(), version, str(value.device),
                     value.dtype, tuple(value.shape), tuple(value.stride()), value.requires_grad))
    return (id(model.model.norm), id(model.lm_head), model.training, tuple(rows))


class FrozenResidentBindings:
    def __init__(self, bench):
        if bench.model.training or any(p.requires_grad for p in bench.model.parameters()):
            raise BenchError("managed frozen readers require eval mode and requires_grad=False")
        self.bench = bench
        self.signature = _signature(bench.model)
        self.epoch = bench._weight_epoch
        self.valid = True
        self.readers = {}
        self.stats = {"reader_builds": 0, "cache_hits": 0, "invalidations": 0}

    def guard(self):
        if (not self.valid or self.epoch != self.bench._weight_epoch
                or self.signature != _signature(self.bench.model)):
            self.valid = False
            self.readers.clear()
            raise BenchError("frozen weight binding invalidated; rebind after verifying the new model/lens")

    def invalidate(self):
        self.valid = False
        self.readers.clear()
        self.stats["invalidations"] += 1


def bind(bench):
    current = getattr(bench, "_frozen_readers", None)
    if current is not None and current.valid:
        current.guard()
        return current
    current = FrozenResidentBindings(bench)
    bench._frozen_readers = current
    return current
