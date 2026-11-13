"""Evaluate saved, measured six-head coefficients without loading a model.

    python read_weight_taylor_demo.py --program ../qualification/read_weight_second_order_20261107/DEMO-PROGRAM.json --dose 0.6

The fixture binds the original report and forecast seal. This command is an
offline coefficient evaluation; it does not reproduce the dense compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mdb_lanes.read_weight_taylor import evaluate_taylor_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=0.6)
    parser.add_argument("--method", choices=("order2", "order1", "stale"), default="order2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    saved = json.loads(args.program.read_text())
    program = saved["program"]
    digest = hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != saved["program_sha256"]:
        raise ValueError("saved program digest changed")
    prediction = evaluate_taylor_program(program, args.dose, method=args.method)
    if args.json:
        print(json.dumps({"source": saved["source"], "prediction": prediction}, indent=2, sort_keys=True))
        return
    print("Offline evaluation of measured Qwen2.5-1.5B coefficients; dense compilation already paid")
    print("job=" + saved["source"]["job_id"] + " receiver=" + saved["source"]["receiver"])
    print(f"dose={args.dose:g} method={args.method}; primary values=18 (72 bytes packed FP32)")
    print("head  baseline_logpB  predicted_delta_logpB  predicted_pB")
    for index, (base, delta, probability) in enumerate(zip(program["logp0"], prediction["delta_logpB"], prediction["pB"], strict=True)):
        print(f"{index:>4}  {base[0]:>14.7f}  {delta[0]:>21.7f}  {probability[0]:>12.7f}")
    print("Source report SHA256: " + saved["source"]["report_sha256"])


if __name__ == "__main__":
    main()
