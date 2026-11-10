"""Evaluate sealed midpoint coefficients without a model or tensor framework.

Dense compilation and native target measurements are already paid. The fixture
binds this replay to its source report, receiver, parent cut and forecast seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mdb.read_weight_centered import evaluate_centered_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--method", choices=("centered", "baseline-corrected", "stale"), default="centered")
    args = parser.parse_args()
    fixture = json.loads(args.program.read_text())
    program = fixture["program"]
    digest = hashlib.sha256(json.dumps(
        program, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if digest != fixture["program_sha256"]:
        raise ValueError("saved centered program digest changed")
    result = evaluate_centered_program(program, args.dose, method=args.method)
    print(json.dumps({"source": fixture["source"], "prediction": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
