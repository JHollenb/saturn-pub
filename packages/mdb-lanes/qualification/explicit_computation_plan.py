"""Outcome-free protocol for explicit suffix equations at an actual MDB frontier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
ARMS = ("native", "zero", "U_swap", "S_swap", "both", "sham",
        "U_swap_P2native", "S_swap_P2native")


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_plan():
    cases = []
    for name, family, code, rdi, scope in (
        ("sub_254_anchor", "sub_r64_32", "4881ef01000000", 254, "previously-observed-explanation"),
        ("xor_255_anchor", "xor_r64_r64", "4831f7", 255, "previously-observed-explanation"),
        ("sub_257_new", "sub_r64_32", "4881ef01000000", 257, "fresh-suffix-forecast"),
        ("sub_258_new", "sub_r64_32", "4881ef01000000", 258, "fresh-suffix-forecast"),
        ("xor_253_new", "xor_r64_r64", "4831f7", 253, "fresh-suffix-forecast"),
        ("xor_254_new", "xor_r64_r64", "4831f7", 254, "fresh-suffix-forecast"),
    ):
        cases.append({"case_id": name, "supplied_family": family,
            "instruction_hex": code, "program_hex": code * 2,
            "register_words": [0, 0, 0, 0, 0, 0, 3, rdi], "flags": [0, 0, 0, 0],
            "scope": scope,
            "explanatory_readers": [{"register": 6, "bit": 1}, {"register": 7, "bit": 0},
                                    {"register": 7, "bit": 12}, {"register": 7, "bit": 13}]})
    plan = {"schema": "mdb-explicit-computation-plan-v1", "model_identity": CHECKPOINT,
        "backend": "cpu-fp32", "cpu_threads": 1, "cases": cases, "arms": list(ARMS),
        "scientific_question": "Can a complete explicit equation account predict the final mixer, post MLP and all consumers from a real prefix frontier, including conditional edits and collateral?",
        "frontier": {"stop_before": "mixers.2", "observe": ["decoder"],
            "allowed": ["native and post.0-zero prefix state S", "prefix decoder z", "raw initial registers/flags",
                "complete frozen parameter dictionary", "declared byte programs and numerical execution contract"],
            "forbidden_before_forecast_seal": ["native final-mixer body execution", "native post.2 execution",
                "native final readout execution", "full-model recording", "suffix replay", "native suffix activations or heads"]},
        "account": {"implementation": "independent NumPy FP64 equations with sequential recurrence",
            "compression_required": False, "native_evaluation": "CPU FP32 native parallel scan",
            "predictions": ["complete suffix intermediate arrays", "all eleven continuous heads", "all decoded registers, flags, length and effect bits",
                "per-register-bit skip, projection, post-MLP, bias and copy-prior contributions"],
            "numerics": "retain every residual; no arbitrary numerical threshold discards scientific rows"},
        "recipes": {
            "native": "native prefix S, unmodified suffix",
            "zero": "post.0-zero prefix S, unmodified suffix",
            "U_swap": "post.0-zero prefix; replace U with frozen FP32 analytically predicted native U",
            "S_swap": "post.0-zero prefix; replace M with frozen FP32(native S + predicted edited U)",
            "both": "post.0-zero prefix; replace M with frozen FP32(native S + predicted native U)",
            "sham": "post.0-zero prefix; same-value clone of actual U",
            "U_swap_P2native": "U_swap, then replace P2 with frozen FP32 analytically predicted native P2",
            "S_swap_P2native": "S_swap, then replace P2 with same frozen analytically predicted native P2"},
        "release_order": ["capture all six actual prefixes with downstream execution forbidden",
            "branch source cuts and compute edited prefixes through MDB replay",
            "derive all suffix forecasts and exact FP32 intervention payloads from allowed inputs",
            "seal and persist complete forecast file/payload bytes and pre-release call counters",
            "release native full captures, MDB suffix branches and independent native audits",
            "optional successors consume literal predicted state; no full-next-forward numerical forecast claimed"],
        "scoring": {"primary": "fresh and anchor strata reported separately; full heads, stage residuals and all register/flag decisions",
            "causal": "compare hybrid responses with their natural versus common frozen post.2 write",
            "controls": "prefix exclusion proof, prefix/full trace joins, exact native replay/audit and same-value Act parity",
            "scope": "conditional on observed prefix state; not an end-to-end raw-input computation account",
            "terminal_status": "unestablished"},
        "supplied": ["architecture/source equations and full weights", "source removal and hybrid intervention geometry",
            "prefix boundary", "two retrospective anchors and four fixed new inputs", "consumer coordinates for explanation"],
        "not_claimed": ["full ISA understanding", "compression", "training-unseen inputs", "general semantic operation naming",
            "cross-instruction latent memory", "native bit-exactness of FP64 equations"]}
    plan["plan_sha256"] = seal(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "plan_sha256": plan["plan_sha256"], "cases": 6,
        "fresh_cases": 4, "model_calls": 0}))


if __name__ == "__main__":
    main()
