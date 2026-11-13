"""Outcome-free, evidence-bound forecasts for a small route-transfer experiment.

Instruction semantics below are supplied fixture knowledge, not learned symbols.
No model/framework import or candidate output is used to construct this plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
EVIDENCE = {
    "screen.json": "4bc6aa534d8ddb690970098238a686d18977f53a15db0165ab3c4e2993ea0783",
    "followups.json": "74be64e8542ef582fc52faabb371d7905401096161c2858307dd0958ae7fbde9",
    "report.json": "69ccaed5e4561b91fd5b7f38ef1f3561cc9a4f6365f61c29219a7022af7d46d0",
}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def words(rsi, rdi):
    return [0, 0, 0, 0, 0, 0, rsi, rdi]


def build_plan():
    # Three positive and three negative reader transitions; all six are fixed.
    specifications = (
        ("sub_256", "sub_r64_32", "4881ef01000000", 3, 256, 8, 255, 254, -1),
        ("sub_254", "sub_r64_32", "4881ef01000000", 3, 254, 0, 253, 252, 1),
        ("xor_252", "xor_r64_r64", "4831f7", 3, 252, 0, 255, 252, 1),
        ("xor_255", "xor_r64_r64", "4831f7", 3, 255, 0, 252, 255, -1),
        ("mov_up", "mov_r64_r64", "4889f7", 256, 255, 8, 256, 256, 1),
        ("mov_down", "mov_r64_r64", "4889f7", 255, 256, 8, 255, 255, -1),
    )
    cases = []
    predictions = []
    for case_id, op, code, rsi, rdi, bit, first, second, sign in specifications:
        case = {"case_id": case_id, "op_label": op, "instruction_hex": code,
                "instruction_length": len(bytes.fromhex(code)), "program_hex": code * 2,
                "register_words": words(rsi, rdi), "flags": [0, 0, 0, 0],
                "reader": {"head": "regs", "register": 7, "bit": bit},
                "expected_first_rdi": first, "expected_second_rdi": second, "sign": sign}
        case["window_hex"] = bytes.fromhex(case["program_hex"])[:15].ljust(15, b"\0").hex()
        case["input_envelope_sha256"] = identity({key: case[key] for key in (
            "window_hex", "register_words", "flags")})
        cases.append(case)
        arms = {}
        for arm in ("identity", "post0_zero", "post1_zero", "post2_zero",
                    "projection_native", "skip_native", "carrier_native", "projection_sham"):
            suppressed = arm in {"post0_zero", "projection_sham"}
            abstain = arm == "skip_native"
            arms[arm] = {
                "first_register_words": None if abstain else words(rsi, rdi if suppressed else first),
                "second_register_words": None if abstain else words(rsi, first if suppressed else second),
                "register_relation_to_native": "abstain" if abstain else (
                    "prior_state" if suppressed else "equal"),
                "length": len(bytes.fromhex(code)),
                "flags": "abstain", "exact_margins": "abstain",
            }
        predictions.append({"case_id": case_id, "arms": arms,
            "signed_effects": {
                "post0_zero_minus_identity": -sign,
                "projection_native_minus_post0_zero": sign,
            },
            "route_ranking": {
                "formula": "abs(Y(projection_native)-Y(identity)) < abs(Y(skip_native)-Y(identity))",
                "reader": case["reader"], "expected": True, "pattern_id": "route_0002",
            },
            "baselines": {
                "no_effect_at_any_post": {site: words(rsi, first) for site in (
                    "post0_zero", "post1_zero", "post2_zero")},
                "prior_state_at_every_post": {site: words(rsi, rdi) for site in (
                    "post0_zero", "post1_zero", "post2_zero")},
                "constant_positive_effect": {"post0_zero_minus_identity": 1,
                    "projection_native_minus_post0_zero": 1},
                "constant_negative_effect": {"post0_zero_minus_identity": -1,
                    "projection_native_minus_post0_zero": -1},
            },
            "abstentions": ["skip_native complete first/second state", "flag signs and decisions",
                "unchanged-reader effect signs", "continuous margin magnitudes",
                "other-head effect signs", "full-state equivalence of single-operand restores"],
        })
    empirical = {"job_id": "job-c5bfaffc8b0a", "model_identity": CHECKPOINT,
        "program_hex": "4881c701000000" * 2, "rsi": 3, "rdi": [252, 253, 254, 255],
        "flags": [0, 0, 0, 0], "restoration_contexts": [252, 255],
        "source_artifacts": EVIDENCE}
    plan = {"schema": "mdb-predictive-route-plan-v1", "model_identity": CHECKPOINT,
        "backend": "cpu-fp32", "cpu_threads": 1, "empirical_scope": empirical,
        "candidate_applicability": {"status": "untested-scope-expansion",
            "envelopes": [case["input_envelope_sha256"] for case in cases],
            "requires": ["exact checkpoint, installed implementation and ABI",
                "fresh structural resolution of post.0/post.1/post.2 and final mixer operands",
                "same-context native donors; no cross-input donor",
                "two instruction calls with literal predicted state and length; hidden reset each call"],
            "outside_scope": "abstain; six new operation fixtures do not certify their entire families"},
        "patterns": [
            {"id": "pattern_0001", "name": "early-write suppression",
             "claim": "Zero post.0 preserves the incoming register bank rather than its native update",
             "provenance": "screen.json: all four old ADD contexts at dose0 versus dose1",
             "scope": "old ADD observations; prospective exact-envelope expansion only"},
            {"id": "route_0002", "name": "projected-write restoration",
             "claim": "After post.0 removal, restoring final U recovers native register decisions and is closer than restoring S at the selected reader",
             "provenance": "followups.json: context_00/bit0 and context_03/bit8, source post.0",
             "old_reader_distances": [
                 {"rdi": 252, "bit": 0, "U": 5.901226043701172, "S": 21.030853271484375},
                 {"rdi": 255, "bit": 8, "U": 4.695281982421875, "S": 28.640533924102783}],
             "scope": "two old ADD receiving states; proposed six-envelope transfer"},
            {"id": "pattern_0003", "name": "later-write decision preservation",
             "claim": "Zero post.1 or post.2 preserves native register decisions",
             "provenance": "screen.json: both sites, four old ADD contexts, dose0 versus dose1",
             "scope": "old ADD observations; no universal necessity or semantic operation claim"},
        ],
        "cases": cases, "predictions": predictions,
        "static_reachability_baseline": {"affected_heads": ["regs", "flags", "imm", "store_addr", "store_value"],
            "unchanged_heads": ["dest", "src", "width", "length", "syscall", "store"],
            "authority": "architecture/static graph only; not evidence of active causal route or learned semantics"},
        "scoring": {"signed_effect": "sign of raw reader-logit difference; exact zero is a separate outcome",
            "route": "compare the two absolute reader distances from unchanged native; ties retained",
            "site": "compare complete eight-register vectors; also retain each wrong bit and native-relative match",
            "continuation": "score complete second register vectors, predicted fetch, and own-state local fixture result separately",
            "aggregation": "per-case and per-family counts, no scientific stopping threshold",
            "controls": "identity exact; no-op equals post0-zero; full carrier restore equals native across all heads"},
        "supplied": ["operation grammar and six raw programs/states", "reader coordinates",
            "semantic expected first/second registers and signed directions", "generic pattern hypotheses",
            "intervention sites/restoration geometry", "baseline and scoring definitions"],
        "learned_from_old_evidence": ["site-specific ablation response", "projected-versus-skip restoration ranking"],
        "not_claimed": ["training-unseen instructions", "automatic instruction semantics discovery",
            "universal route law", "persistent internal memory across calls", "terminal certification"],
    }
    plan["plan_sha256"] = identity(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in EVIDENCE.items():
        if hashlib.sha256((args.evidence_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("historical evidence binding differs: " + name)
    plan = build_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "cases": len(plan["cases"]),
        "output": str(args.output), "model_calls": 0}))


if __name__ == "__main__":
    main()
