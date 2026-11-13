"""Independent saved-array Qwen-family audit; zero model forwards or weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain_metadata(value):
    """Normalize the public immutable reader's mappings/tuples to JSON values."""
    if isinstance(value, Mapping):
        return {name: plain_metadata(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_metadata(item) for item in value]
    require(value is None or type(value) in (str, int, float, bool), "non-JSON leaf in sealed metadata index")
    return value


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def compare(predicted, actual):
    require(predicted.shape == actual.shape and predicted.dtype == actual.dtype, "numeric geometry differs")
    require(np.isfinite(predicted).all() and np.isfinite(actual).all(), "nonfinite saved evidence")
    difference = predicted.astype(np.float64) - actual.astype(np.float64)
    return dict(elements=int(predicted.size), max_abs=float(np.abs(difference).max(initial=0.)),
        rms=float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.,
        signed_mean=float(difference.mean()) if difference.size else 0.,
        bit_equal=predicted.tobytes() == actual.tobytes())


def audit(root, manifest_sha, output):
    from mdb.computation_protocol import evaluate_rule, verify_plan
    from mdb.equation_comparison import compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(root), Path(output)
    require(sha(root / "files.json") == manifest_sha, "published inventory differs")
    inventory = read(root / "files.json")["files"]
    for name, record in inventory.items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) and
                path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "immutable member differs: " + name)
    verification = verify_transaction(root / "transaction")
    config, request = read(root / "plan.json"), read(root / "request.json")
    plan = verify_plan(config["compiled_protocol"])
    summary, labels, inputs = read(root / "summary.json"), read(root / "answer-labels.json"), read(root / "inputs.json")
    predicted, native = read(root / "forecast-index.json"), read(root / "native-index.json")
    sealed = load_forecast_tree(root / "transaction", read(root / "transaction/forecasts/all_futures.json")["predictions"])
    require(plain_metadata(sealed) == predicted and set(predicted) == set(native), "sealed case census differs")
    timeline = [json.loads(row) for row in (root / "timeline.jsonl").read_text().splitlines()]
    seals = [index for index, row in enumerate(timeline) if row["kind"] == "global_forecast_sealed"]
    require(len(seals) == 1 and timeline[seals[0]]["counts"]["native_calls"] == 0
            and not any(row["kind"].startswith("native_") for row in timeline[:seals[0]])
            and summary["counts"]["denied_entries"] == 0, "native-exclusion timeline differs")
    require(summary["counts"]["native_calls"] == summary["counts"]["functional_calls"] == len(predicted) * config["tokens"], "physical forward census differs")

    def load(record):
        path = root / record["path"]
        require(sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "sealed array reference differs")
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    cases, first, comparisons, ports, cache_trajectory, equation_reports = {}, {}, [], {}, {}, {}
    token_matches = 0
    source_port_map = dict(q="q", k="k", v="v", original_q="original_q", original_k="original_k", original_v="original_v",
        gate="gate", up="up", original_gate="original_gate", original_up="original_up", mask="mask", residual="residual", normalized="normalized",
        attention_return="attention_return", post_attention="post_attention", mlp_return="mlp_return")
    for key in predicted:
        cid, action = key.split("/")
        require(len(predicted[key]["steps"]) == len(native[key]["steps"]) == config["tokens"], "token horizon differs")
        lane_rows, selected = {}, []
        for lane, index in (("forecast", predicted), ("native", native)):
            previous = None
            records = []
            for step, reference in enumerate(index[key]["steps"]):
                row = load(reference)
                expected_input = inputs[cid]["P" if action == "source" else "Q"] if step == 0 else [[previous]]
                require(row["input_ids"].tolist() == expected_input and int(row["token"][0]) == int(row["logits"][0, -1].argmax()), "own-token recurrence broken")
                length = len(expected_input[0]) if step == 0 else len(inputs[cid]["Q"][0]) + step
                for name, value in row.items():
                    if name.startswith(("k", "v")) and name[1:].isdigit():
                        require(value.shape[-2] == length, "KV clock differs")
                previous = int(row["token"][0])
                records.append(row)
            lane_rows[lane] = records
        for step, (forecast, actual) in enumerate(zip(lane_rows["forecast"], lane_rows["native"])):
            require(set(forecast) == set(actual), "retained numeric field census differs")
            reports = {name: compare(forecast[name], actual[name]) for name in forecast}
            comparisons.extend(dict(case=key, step=step, field=name, **report) for name, report in reports.items())
            agreement = int(forecast["token"][0]) == int(actual["token"][0])
            token_matches += int(agreement)
            selected.append(dict(step=step, forecast_token=int(forecast["token"][0]), native_token=int(actual["token"][0]),
                token_agreement=agreement, logits=reports["logits"], max_cache_error=max(report["max_abs"] for name, report in reports.items() if name.startswith(("k", "v")) and name[1:].isdigit())))
        first[key] = lane_rows["native"][0]
        p, n = load(predicted[key]["ports"]), load(native[key]["ports"])
        ports[key] = {name: compare(p[name], n[target]) for name, target in source_port_map.items() if name in p and target in n}
        ports[key]["endpoint"] = compare(lane_rows["forecast"][0]["endpoint"], lane_rows["native"][0]["endpoint"])
        cases[key] = dict(steps=selected, forecast_text=predicted[key]["text"], native_text=native[key]["text"],
            all_tokens_agree=all(row["token_agreement"] for row in selected), final_cache_length=int(lane_rows["native"][-1]["k00"].shape[-2]))
        equation_inputs = {lane: dict(endpoint=rows[0]["endpoint"],
            logits=np.concatenate([row["logits"] for row in rows], axis=1)) for lane, rows in lane_rows.items()}
        equation_reports[key] = compare_equations(equation_inputs["forecast"], equation_inputs["native"],
            decision_rules=dict(logits="argmax"), max_elements=2_000_000)

    measurements, observations, controls, interactions = [], {}, {}, {}
    for context in config["contexts"]:
        cid = context["id"]
        p_id, q_id = labels[cid]["P"][0], labels[cid]["Q"][0]
        rows = {}
        q_first = first[cid + "/recipient"]
        for action in plan["actions"]:
            aid = action["id"]
            row, base = first[cid + "/" + aid], q_first["logits"].astype(np.float64)
            logits = row["logits"].astype(np.float64)
            score = float(logits[0, -1, q_id] - logits[0, -1, p_id])
            shifted = logits[0, -1] - logits[0, -1].max()
            probability = np.exp(shifted) / np.exp(shifted).sum()
            rows[aid] = dict(score=score, label_mass=float(probability[p_id] + probability[q_id]),
                p_label_logit=float(logits[0, -1, p_id]), q_label_logit=float(logits[0, -1, q_id]),
                full_logit_rms_from_Q=float(np.sqrt(np.mean((logits-base)**2))),
                first_token=int(row["token"][0]), text=cases[cid + "/" + aid]["native_text"])
            measurements.append(dict(context=cid, action=aid, value=score))
            layers = []
            for index in range(len([name for name in row if name.startswith("k") and name[1:].isdigit()])):
                reports = [compare(q_first[f"{name}{index:02d}"], row[f"{name}{index:02d}"]) for name in ("k", "v")]
                layers.append(dict(layer=index, max_abs=max(item["max_abs"] for item in reports), rms=max(item["rms"] for item in reports)))
            cache_trajectory[cid + "/" + aid] = layers
        baseline = first[cid + "/baseline"]
        controls[cid] = {name: compare(q_first[name], baseline[name]) for name in q_first}
        candidates = plan["candidate_menu"]
        a, b = first[cid + "/" + candidates[0]], first[cid + "/" + candidates[1]]
        both = first[cid + "/both"]
        interactions[cid] = {}
        for field in ("endpoint", "logits"):
            mixed = both[field].astype(np.float64)-a[field].astype(np.float64)-b[field].astype(np.float64)+baseline[field].astype(np.float64)
            interactions[cid][field] = dict(rms=float(np.sqrt(np.mean(mixed*mixed))), max_abs=float(np.abs(mixed).max()), signed_mean=float(mixed.mean()))
        observations[cid] = dict(rows=rows, natural_headroom=rows["recipient"]["score"]-rows["source"]["score"],
            interpretation="Smaller supplied-label contrast is more P-directed; label mass and natural headroom are diagnostics, not semantic certification")
    packet = dict(plan_sha256=plan["plan_sha256"], model_identity=plan["specification"]["model_identity"],
        metric=plan["specification"]["metric"], contexts=config["contexts"], rows=measurements,
        provenance=dict(native_job=summary["job_id"], request_sha256=request["request_sha256"], native_manifest_sha256=manifest_sha))
    evaluation = evaluate_rule(plan, config["effect_rule"], packet) if config["phase"] == "fresh" else None
    report = dict(schema="mdb-qwen-family-saved-audit-v1", mechanics_status="verified", instrument_status="executed-on-all-declared-fields",
        trend_status="measurements-retained", terminal_status="unassessed", model_forwards=0,
        family=config["family"], native_job=summary["job_id"], request_sha256=request["request_sha256"],
        source_root=str(root), source_manifest_sha256=manifest_sha, transaction_verification=verification,
        cases=cases, comparisons=comparisons, selected_layer_port_comparisons=ports, equation_reports=equation_reports,
        observations=observations, measurement_packet=packet, frozen_effect_evaluation=evaluation,
        QQ_control_comparisons=controls, first_readout_mixed_terms=interactions, prefill_cache_write_trajectory=cache_trajectory,
        token_matches=token_matches, token_predictions=len(cases)*config["tokens"],
        compared_array_pairs=len(comparisons), compared_elements=sum(row["elements"] for row in comparisons),
        max_logit_error=max(row["max_abs"] for row in comparisons if row["field"] == "logits"),
        max_cache_error=max(row["max_abs"] for row in comparisons if row["field"].startswith(("k", "v")) and row["field"][1:].isdigit()),
        array_bit_misses=sum(not row["bit_equal"] for row in comparisons),
        limitations=["One checkpoint, CPU FP32 eager, supplied two-factor graphs and zero-based layer12", "Three fresh dependent prompt pairs; not training-data holdouts", "Numerical recurrence uses original parameters and full kernels; no compression or faster inference claim", "Label contrast does not certify a semantic assignment switch", "Six choices use prefill and five cached calls; final choice remains pending; EOS does not stop this fixed recurrence"])
    save(output / "audit.json", report)
    print("MDB_QWEN_AUDIT_READY " + json.dumps(dict(output_root=str(output), audit_sha256=sha(output / "audit.json")), sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.manifest_sha256, args.out)
