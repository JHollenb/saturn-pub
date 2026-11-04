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


def time_arms_interleaved(arms: Mapping[str, Callable[[], Any]], *, baseline: str, gate: Mapping[str, Any] | None,
                          blocks: int = 5, reps: int = PROTOCOL["reps"], warmup: int = PROTOCOL["warmup"],
                          cuda: bool = False, pairs: list[tuple[str, str]] | None = None,
                          status: Mapping[str, str] | None = None, batch: int | None = None,
                          device: str | None = None) -> dict[str, Any]:
    """Repeat-noise record: ``blocks`` independent timing blocks per arm, interleaved round-robin across the arms (round
    b times every arm once, in the arms' order, so slow drift cancels across arms), each block the registered
    protocol's median of ``reps`` repetitions. Reports per arm the block medians, their mean and between-block sd (the
    sd of the median), the per-round time ratio against ``baseline``, and for each named pair (a, b) the per-round
    paired difference median_a - median_b with its mean and sd. ``status`` maps an arm to its qualification status
    (for example 'sealed', 'amended', 'unqualified'), copied into the record next to its times. Speedups and ratios are
    refused without a passing gate, as in ``time_arms``."""
    if baseline not in arms:
        raise TimingError(f"baseline {baseline!r} is not an arm")
    if type(blocks) is not int or blocks < 2:
        raise TimingError("blocks must be an integer >= 2")
    passed = bool(gate and gate.get("passed"))
    meds: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(blocks):
        for name, fn in arms.items():
            meds[name].append(time_fn(fn, reps=reps, warmup=warmup, cuda=cuda)["median_ms"])
    out_arms = {}
    for name, ms in meds.items():
        ratio = [m / b for m, b in zip(ms, meds[baseline])] if passed else None
        out_arms[name] = {"block_median_ms": ms, "mean_ms": statistics.fmean(ms), "sd_ms": statistics.stdev(ms),
                          "median_ms": statistics.median(ms), "time_ratio_by_block": ratio,
                          "time_ratio_mean": statistics.fmean(ratio) if ratio else None,
                          "time_ratio_sd": statistics.stdev(ratio) if ratio else None,
                          "status": (status or {}).get(name)}
    pooled = statistics.fmean([a["sd_ms"] ** 2 for a in out_arms.values()]) ** 0.5
    prs = {}
    for a, b in pairs or []:
        dif = [x - y for x, y in zip(meds[a], meds[b])]
        prs[f"{a}-{b}"] = {"diff_ms_by_block": dif, "mean_ms": statistics.fmean(dif), "sd_ms": statistics.stdev(dif),
                           "status": [(status or {}).get(a), (status or {}).get(b)]}
    return {"schema": TIMING_SCHEMA + "-interleaved", "protocol": {**PROTOCOL, "reps": reps, "warmup": warmup,
                                                                   "blocks": blocks, "order": "round-robin by block"},
            "batch": batch, "device": device, "cuda_synchronized": bool(cuda), "baseline": baseline,
            "claim": "ratios reported" if passed else "refused: parity gate missing or failed", "gate": dict(gate) if gate else None,
            "arms": out_arms, "pooled_between_block_sd_ms": pooled, "pairs": prs}


__all__ = ["PROTOCOL", "TIMING_SCHEMA", "TimingError", "gpu_processes", "time_arms", "time_arms_interleaved", "time_fn"]
