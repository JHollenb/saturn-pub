"""Independent saved-array audit: no model execution and no Torch import."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(root, descriptor):
    path = root / descriptor["path"]
    if sha(path) != descriptor["sha256"]:
        raise ValueError("array custody mismatch")
    raw = np.load(path, allow_pickle=False).tobytes()
    dtype = descriptor["dtype"]
    if dtype == "torch.bfloat16":
        words = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
        value = words.view("<f4")
    elif dtype == "torch.float32":
        value = np.frombuffer(raw, dtype="<f4")
    elif dtype == "torch.float16":
        value = np.frombuffer(raw, dtype="<f2").astype("<f4")
    else:
        raise ValueError("unsupported saved tensor dtype: " + dtype)
    value = value.reshape(descriptor["shape"])
    if not np.isfinite(value).all():
        raise ValueError("nonfinite saved tensor")
    return value


def scalar_sub(a, b, dtype):
    x = np.array([np.float32(a) - np.float32(b)], dtype="<f4")
    if dtype == "torch.bfloat16":
        words = x.view("<u4")
        return float(((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2").astype("<u4").__lshift__(16).view("<f4")[0])
    if dtype == "torch.float16":
        return float(x.astype("<f2")[0])
    return float(x[0])


def audit(root):
    manifest = read(root / "output-manifest.json")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("inventory mismatch: " + entry["path"])
    graph = read(root / "invocations.json")
    body = {k: v for k, v in graph.items() if k != "fingerprint"}
    if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != graph["fingerprint"]:
        raise ValueError("graph fingerprint mismatch")
    node_ids = [n["id"] for n in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)) or any(e["source"] not in node_ids or e["target"] not in node_ids for e in graph["edges"]):
        raise ValueError("graph identity/edge closure mismatch")
    rows = read(root / "native-rows.json")
    for row in rows:
        logits = array(root, row["logits"])[0]
        if int(logits.argmax()) != row["next_token_id"]:
            raise ValueError("native argmax differs from recorded token")
    groups = {}
    binding_count = 0
    for occurrence in graph["occurrences"]:
        for binding in occurrence["bindings"]:
            if binding["status"] != "resolved":
                continue
            binding_count += 1
            measurements = {"logical": [], "fixed_distance": [], "shuffled": [], "control_heads": []}
            for head in binding["heads"]:
                measurements["logical"].append(sum(a["attention_by_head"][str(head)] for a in binding["arguments"]))
                measurements["fixed_distance"].append(sum(a["fixed_distance_attention"][str(head)] or 0.0 for a in binding["arguments"]))
                measurements["shuffled"].append(sum(a["shuffled_attention"][str(head)] for a in binding["arguments"]))
            for head in (0, 3):
                measurements["control_heads"].append(sum(a["control_heads_attention"][str(head)] for a in binding["arguments"]))
            key = occurrence["case"].split("-")[0] + "/" + binding["role"]
            group = groups.setdefault(key, {"invocations": 0, "logical": [], "fixed_distance": [], "shuffled": [], "control_heads": []})
            group["invocations"] += 1
            for name, values in measurements.items():
                group[name].append(statistics.mean(values))
    summaries = {key: {"invocations": value["invocations"], **{
        name + "_median_mass": statistics.median(value[name])
        for name in ("logical", "fixed_distance", "shuffled", "control_heads")}} for key, value in groups.items()}
    interventions = read(root / "interventions.json")
    effects, sham_exact, margin_exact, feedback_links = [], 0, 0, 0
    for pair in interventions:
        recipient = array(root, pair["recipient_native"]["logits"])[0]
        target, alternative = pair["target_token_id"], pair["alternative_token_id"]
        dtype = pair["recipient_native"]["logits"]["dtype"]
        base = scalar_sub(recipient[target], recipient[alternative], dtype)
        for arm in pair["arms"]:
            actual = array(root, arm["observation"]["logits"])[0]
            effect = scalar_sub(actual[target], actual[alternative], dtype) - base
            if effect != arm["margin_effect"]:
                raise ValueError("native finite margin arithmetic differs from saved calculation")
            margin_exact += 1
            if arm["arm"] == "sham":
                sham_exact += int(np.array_equal(actual, recipient))
            trajectory = [arm["observation"], *arm["continuation"]]
            if [row["next_token_id"] for row in trajectory] != arm["own_token_ids"]:
                raise ValueError("own-output decision chain mismatch")
            for previous, current in zip(trajectory, trajectory[1:]):
                if current["consumed_token_id"] != previous["next_token_id"]:
                    raise ValueError("supplied value in generated continuation")
                if current["cache_rows"] != previous["cache_rows"] + 1:
                    raise ValueError("cache clock mismatch")
                feedback_links += 1
            delta = actual.astype(np.float64) - recipient.astype(np.float64)
            effects.append({"pair": pair["pair"]["id"], "arm": arm["arm"],
                           "margin_effect": effect, "next_token_id": arm["next_token_id"],
                           "own_text": arm["own_text"], "writer23_l2": arm["writer23_l2"],
                           "logit_l2_recomputed": float(np.linalg.norm(delta)),
                           "logit_l2_worker": arm["logit_l2"], "changed_kv_layers": arm["changed_kv_layers"]})
    report = read(root / "report.json")
    return {"schema": "mdb-live-scaffold-saved-audit-v1", "job_id": report["job_id"],
            "raw_report_sha256": sha(root / "report.json"), "files_verified": len(manifest["files"]),
            "native_argmax_rows_verified": len(rows), "bindings": binding_count,
            "graph_nodes": len(node_ids), "graph_edges": len(graph["edges"]),
            "schema_fingerprint": graph["scaffold"]["fingerprint"],
            "case_outcomes": report["cases"], "read_mass": summaries,
            "intervention_pairs": len(interventions), "sham_logits_exact": sham_exact,
            "finite_margin_effects_exact": margin_exact, "own_output_feedback_links_verified": feedback_links,
            "effects": effects, "object_followup": report["object_followup"],
            "mechanics_status": "valid-saved-array-and-graph-audit",
            "instrument_status": "full logits, port descriptors and bookkeeping checked; no independent native rerun or full-cache numerical replay",
            "trend_status": "per-role masses and per-arm signed effects retained",
            "terminal_status": "not-requested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("job_id", "files_verified", "native_argmax_rows_verified", "bindings", "sham_logits_exact", "own_output_feedback_links_verified")}))


if __name__ == "__main__":
    main()
