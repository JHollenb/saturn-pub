"""Registered timing protocol for predictor-versus-native speed claims.

``time_arms`` times named zero-argument callables under one protocol: warmup
calls, then ``reps`` repetitions, each bracketed by a CUDA synchronize when the
device is CUDA. It reports median, p10 and p90 milliseconds and the speedup of
every arm against a named baseline. A speed claim is refused, and no speedup is
reported, when the supplied parity gate did not pass: a predictor whose resumed
or injected computation does not reproduce native cannot be timed as a stand-in
for it. Torch is imported only when CUDA synchronization is requested.
"""
from __future__ import annotations

import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from typing import Any

TIMING_SCHEMA = "mdb-timing-v1"
PROTOCOL = {"name": "mdb-registered-timing-v1", "warmup": 5, "reps": 30, "batches": [1, 256],
            "synchronize": "torch.cuda.synchronize after warmup and around every repetition on CUDA",
            "statistic": "median with p10 and p90 of per-repetition wall milliseconds",
            "baseline": "native prefix-cached forward of the same suffix rows through the same layer loop"}


class TimingError(ValueError):
    pass


def _quantile(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def time_fn(fn: Callable[[], Any], *, reps: int = PROTOCOL["reps"], warmup: int = PROTOCOL["warmup"],
            cuda: bool = False) -> dict[str, Any]:
    if type(reps) is not int or reps < 1 or type(warmup) is not int or warmup < 0:
        raise TimingError("reps must be a positive integer and warmup a non-negative integer")
    sync = (lambda: None)
    if cuda:
        import torch
        sync = torch.cuda.synchronize
    for _ in range(warmup):
        fn()
    sync()
    ms = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ms.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": statistics.median(ms), "p10_ms": _quantile(ms, 0.1), "p90_ms": _quantile(ms, 0.9),
            "reps": reps, "warmup": warmup}


def gpu_processes() -> list[str] | None:
    """Best-effort ``nvidia-smi`` compute-process snapshot (pid, used memory), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else None


def time_arms(arms: Mapping[str, Callable[[], Any]], *, baseline: str, gate: Mapping[str, Any] | None,
              reps: int = PROTOCOL["reps"], warmup: int = PROTOCOL["warmup"], cuda: bool = False,
              batch: int | None = None, device: str | None = None) -> dict[str, Any]:
    """Time every arm and the baseline under the registered protocol. ``gate`` is a
    parity record with ``gate["passed"]``; without a passing gate the record keeps
    the timings but sets ``claim`` to refused and reports no speedups."""
    if baseline not in arms:
        raise TimingError(f"baseline {baseline!r} is not an arm")
    passed = bool(gate and gate.get("passed"))
    before = gpu_processes() if cuda else None
    results = {name: time_fn(fn, reps=reps, warmup=warmup, cuda=cuda) for name, fn in arms.items()}
    after = gpu_processes() if cuda else None
    base = results[baseline]["median_ms"]
    for name, r in results.items():
        r["speedup_vs_baseline"] = base / r["median_ms"] if passed and r["median_ms"] > 0 else None
    return {"schema": TIMING_SCHEMA, "protocol": {**PROTOCOL, "reps": reps, "warmup": warmup},
            "batch": batch, "device": device, "cuda_synchronized": bool(cuda), "baseline": baseline,
            "claim": "speedups reported" if passed else "refused: parity gate missing or failed",
            "gate": dict(gate) if gate else None, "arms": results,
            "gpu_processes": {"before": before, "after": after}}


__all__ = ["PROTOCOL", "TIMING_SCHEMA", "TimingError", "gpu_processes", "time_arms", "time_fn"]
