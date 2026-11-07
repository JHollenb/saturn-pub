"""Framework-free toy anchor for the six-head read-weight transport equation.

This is an executable algebra demo, not a model run.  The six-head coefficients
below are deterministic toy scalars chosen to make the normalized score update
visible; they are not trained or authenticated Qwen measurements.  The
authenticated trained result is linked from READ-WEIGHT-TRANSPORT.md and the
qualification findings.

Run:

    python read_weight_transport_demo.py --dose 0.5
    python read_weight_transport_demo.py --dose 0.5 --json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

# Explicitly toy: these six rows stand in for six query heads reading one KV
# head.  Each head has five baseline score rows, a first-order score direction
# a_j, and the current-row cross term b_t.
TOY_HEADS: tuple[dict[str, Any], ...] = (
    {"name": "h0", "baseline_logits": (1.10, 0.15, -0.40, 0.35, 0.05),
     "a": (0.12, -0.18, 0.04, 0.07, 0.21), "b": 0.03},
    {"name": "h1", "baseline_logits": (0.70, -0.20, 0.10, 0.25, -0.30),
     "a": (-0.08, 0.16, -0.05, 0.11, 0.09), "b": -0.02},
    {"name": "h2", "baseline_logits": (0.35, 0.80, -0.15, -0.25, 0.20),
     "a": (0.05, 0.09, 0.14, -0.12, 0.18), "b": 0.04},
    {"name": "h3", "baseline_logits": (-0.10, 0.45, 0.60, -0.30, 0.05),
     "a": (0.17, -0.04, 0.06, 0.13, -0.10), "b": 0.01},
    {"name": "h4", "baseline_logits": (0.20, -0.35, 0.55, 0.10, 0.40),
     "a": (-0.11, 0.07, 0.03, 0.15, 0.05), "b": -0.03},
    {"name": "h5", "baseline_logits": (0.90, 0.05, -0.25, 0.30, -0.10),
     "a": (0.02, -0.13, 0.08, 0.05, 0.19), "b": 0.02},
)


def _log_softmax(values: tuple[float, ...] | list[float]) -> list[float]:
    pivot = max(values)
    normalizer = pivot + math.log(sum(math.exp(value - pivot) for value in values))
    return [value - normalizer for value in values]


def _toy_prediction(head: dict[str, Any], *, dose: float, source_row: int,
                    current_row: int, stale: bool = False) -> dict[str, float]:
    baseline = tuple(float(value) for value in head["baseline_logits"])
    direction = tuple(float(value) for value in head["a"])
    if len(baseline) != len(direction):
        raise ValueError("toy baseline and direction row counts differ")
    logp0 = _log_softmax(baseline)
    if stale:
        predicted = logp0
    else:
        delta = [dose * value for value in direction]
        delta[current_row] += dose * dose * float(head["b"])
        predicted = _log_softmax([base + change for base, change in zip(logp0, delta)])
    return {
        "baseline_probability": math.exp(logp0[source_row]),
        "predicted_probability": math.exp(predicted[source_row]),
        "delta_log_probability": predicted[source_row] - logp0[source_row],
    }


def run_demo(*, dose: float, source_row: int, current_row: int) -> dict[str, Any]:
    if not math.isfinite(dose):
        raise ValueError("dose must be finite")
    row_count = len(TOY_HEADS[0]["baseline_logits"])
    if not 0 <= source_row < row_count:
        raise ValueError(f"source_row must be in [0, {row_count})")
    if not 0 <= current_row < row_count:
        raise ValueError(f"current_row must be in [0, {row_count})")
    rows = []
    for head in TOY_HEADS:
        transport = _toy_prediction(
            head, dose=dose, source_row=source_row, current_row=current_row
        )
        stale = _toy_prediction(
            head, dose=dose, source_row=source_row, current_row=current_row, stale=True
        )
        rows.append({"head": head["name"], "transport": transport, "stale": stale})
    return {
        "schema": "mdb-read-weight-transport-toy-demo-v1",
        "scope": "toy-anchored scalar algebra; no trained coefficients and no model execution",
        "heads": len(TOY_HEADS),
        "rows": row_count,
        "source_row": source_row,
        "current_row": current_row,
        "dose": dose,
        "results": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose", type=float, default=0.5,
                        help="scalar lambda in the toy score update (default: 0.5)")
    parser.add_argument("--source-row", type=int, default=1,
                        help="row whose probability is reported (default: 1)")
    parser.add_argument("--current-row", type=int, default=4,
                        help="current append row receiving the b_t term (default: 4)")
    parser.add_argument("--json", action="store_true",
                        help="emit the complete toy result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_demo(
        dose=args.dose,
        source_row=args.source_row,
        current_row=args.current_row,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print("scope: " + report["scope"])
    print(
        f"dose={report['dose']:.3g} source_row={report['source_row']} "
        f"current_row={report['current_row']} heads={report['heads']}"
    )
    print("head  baseline_pB  transport_pB  transport_delta_logpB  stale_pB")
    for row in report["results"]:
        transport = row["transport"]
        stale = row["stale"]
        print(
            f"{row['head']:>3}   {transport['baseline_probability']:.6f}"
            f"     {transport['predicted_probability']:.6f}"
            f"             {transport['delta_log_probability']:+.6f}"
            f"     {stale['predicted_probability']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
