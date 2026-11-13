"""Saved-only custody, typed-payload, native decision and frozen-route audit."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(root, descriptor):
    path = root / descriptor["path"]
    if sha(path) != descriptor["sha256"]:
        raise ValueError("saved array digest mismatch")
    raw = np.load(path, allow_pickle=False).tobytes()
    if descriptor["dtype"] == "torch.bfloat16":
        value = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    elif descriptor["dtype"] == "torch.float32":
        value = np.frombuffer(raw, dtype="<f4")
    else:
        raise ValueError("unsupported native dtype")
    value = value.reshape(descriptor["shape"])
    if not np.isfinite(value).all():
        raise ValueError("nonfinite retained array")
    return value


def audit(root):
    manifest = read(root / "output-manifest.json")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("retained file inventory mismatch: " + entry["path"])
    report = read(root / "report.json")
    routes = dict(read(root / "request.json")["config"].get("frozen_routes", {}))
    results, native_rows, feedback_links, payloads, shams = [], 0, 0, 0, 0
    graph_nodes, graph_edges = [], []
    numeric_rows = []
    for short in report["cases"]:
        result = read(root / (short["id"] + "-result.json"))
        case = result["case"]
        frozen = result["frozen_route"]
        if case["split"] == "discovery":
            if frozen != read(root / (case["lane"] + "-frozen-route.json")):
                raise ValueError("discovery route file mismatch")
            routes[case["lane"]] = frozen
        elif frozen != routes[case["lane"]]:
            raise ValueError("route refitted on transfer")
        if result["restored_parent_cut"] != result["recipient_binding"]["parent_cut"]:
            raise ValueError("selected parent not restored")
        functions = [("native", result["recipient_native"]), ("donor-native", result["donor_native"]),
                     ("observed", result["observed_native"])]
        functions.extend((arm["arm"], arm["future"]) for arm in result["arms"])
        for name, future in functions:
            rows = future["trajectory"]
            if [r["next_token_id"] for r in rows] != future["own_token_ids"]:
                raise ValueError("native decision chain mismatch")
            for row in rows:
                logits = array(root, row["logits"])[0]
                if int(logits.argmax()) != row["next_token_id"]:
                    raise ValueError("saved full-vocabulary argmax mismatch")
                native_rows += 1
            for previous, current in zip(rows, rows[1:]):
                if current["consumed_token_id"] != previous["next_token_id"]:
                    raise ValueError("later step consumed a supplied future answer")
                if current["history_ids"] != previous["history_ids"] + [previous["next_token_id"]]:
                    raise ValueError("native token-history extension mismatch")
                feedback_links += 1
        native = array(root, result["recipient_native"]["trajectory"][0]["logits"])[0]
        observed = array(root, result["observed_native"]["trajectory"][0]["logits"])[0]
        if not np.array_equal(native, observed):
            raise ValueError("observer control changed native logits")
        observable = result["observable"]
        a, b = observable["recipient_token_id"], observable["donor_token_id"]
        base = float(np.float32(native[b] - native[a]))
        effects = []
        for arm in result["arms"]:
            actual = array(root, arm["future"]["trajectory"][0]["logits"])[0]
            effect = float(np.float32(actual[b] - actual[a])) - base
            if effect != arm["donor_margin_effect"]:
                raise ValueError("saved native semantic margin mismatch")
            for payload in arm["installed"]["payloads"]:
                before, after = array(root, payload["before"]), array(root, payload["after"])
                head = payload["kv_head"]
                if head is not None:
                    mask = np.arange(before.shape[1]) != head
                    if not np.array_equal(before[:, mask], after[:, mask]):
                        raise ValueError("typed edit escaped selected head")
                if arm["arm"] == "sham" and not np.array_equal(before, after):
                    raise ValueError("sham changed source payload")
                payloads += 1
            if arm["arm"] == "sham":
                if not np.array_equal(native, actual) or arm["changed_kv_layers"]:
                    raise ValueError("sham changed native result or KV")
                shams += 1
            item = {"arm": arm["arm"], "own_text": arm["future"]["own_text"],
                    "donor_margin_effect": effect, "logit_l2": arm["logit_l2"],
                    "writer_l2": arm["writer_l2"], "changed_kv_layers": arm["changed_kv_layers"],
                    "recipient_score": arm["recipient_score"], "donor_score": arm["donor_score"]}
            effects.append(item)
            numeric_rows.append({"case": case["id"], "split": case["split"], "arm": arm["arm"],
                                 "metrics": {"signed_donor_margin_effect": effect,
                                             "logit_l2": arm["logit_l2"], "writer_l2": arm["writer_l2"]},
                                 "source_report": str(root / (case["id"] + "-result.json"))})
        record = {"id": case["id"], "lane": case["lane"], "split": case["split"],
                  "route": frozen, "recipient_score": result["recipient_score"],
                  "donor_score": result["donor_score"], "effects": effects,
                  "observable": observable,
                  "observable_matches_native_pair": a == result["recipient_native"]["trajectory"][0]["next_token_id"]
                    and b == result["donor_native"]["trajectory"][0]["next_token_id"],
                  "native_first_tokens": {side: result[side + "_native"]["trajectory"][0]["next_token_text"]
                                          for side in ("recipient", "donor")}}
        results.append(record)
        key = case["id"]
        for side in ("recipient", "donor"):
            graph_nodes.append({"id": key + ":" + side + ":argument", "kind": "supplied-semantic-argument",
                                "lane": case["lane"], "text": case["source" if side == "recipient" else "donor_source"],
                                "source_rows": result[side + "_binding"]["source_rows"]})
        for kind in ("source-value", "reader", "writer", "output", "future"):
            graph_nodes.append({"id": key + ":" + kind, "kind": kind, "route": frozen,
                                "evidence": case["id"] + "-result.json"})
        for left, right, authority in (("recipient:argument", "source-value", "supplied span plus native address"),
                ("source-value", "reader", "source-constrained nomination; effects separate"),
                ("reader", "writer", "declared native execution, not isolated mediation"),
                ("writer", "output", "declared native execution, not isolated mediation"),
                ("output", "future", "verified own-output consumption")):
            graph_edges.append({"source": key + ":" + left, "target": key + ":" + right, "authority": authority})
        for effect in effects:
            node = key + ":act:" + effect["arm"]
            graph_nodes.append({"id": node, "kind": "typed-value-intervention", "effect": effect})
            graph_edges.append({"source": node, "target": key + ":output", "authority": "measured native intervention response"})
    summary = {"schema": "mdb-semantic-scaffold-saved-audit-v1", "job_id": report["job_id"],
               "files_verified": len(manifest["files"]), "native_argmax_rows_verified": native_rows,
               "own_output_feedback_links_verified": feedback_links, "typed_payloads_checked": payloads,
               "sham_logits_and_first_kv_exact": shams, "frozen_routes": routes, "cases": results,
               "mechanics_status": "valid-saved-array-route-and-payload-audit",
               "instrument_status": "saved logits, payload locality, source routes and history checked; no independent native rerun",
               "terminal_status": "not-assessed", "numeric_rows": numeric_rows,
               "graph": {"nodes": graph_nodes, "edges": graph_edges}}
    return summary


def render(summary, destination):
    rows = []
    for case in summary["cases"]:
        for effect in case["effects"]:
            values = (case["id"], str(case["route"]["layer"]) + "/KV" + str(case["route"]["kv_head"]),
                      case["recipient_score"]["output"], effect["arm"], effect["own_text"],
                      f"{effect['donor_margin_effect']:+.4f}", effect["donor_score"]["correct"])
            rows.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + "</tr>")
    data = json.dumps(summary["graph"], allow_nan=False).replace("<", "\\u003c")
    content = '<!doctype html><meta charset="utf-8"><title>Semantic scaffold assay</title><style>body{font:15px system-ui;margin:32px;background:#f5f7fb;color:#182033}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #dbe2ed;text-align:left;white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
    content += '<h1>Semantic arguments, native circuits and measured futures</h1><p>Source labels and tasks are supplied. Discovery nominates two source-restricted carriers; signed native effects choose one route. Transfer keeps that route frozen. Full dense/native execution remains in use.</p>'
    content += '<table><thead><tr><th>Case</th><th>Frozen route</th><th>Native</th><th>Act</th><th>Own output</th><th>Donor margin shift</th><th>Donor answer</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    content += '<details><summary>Complete occurrence graph</summary><pre id="raw"></pre></details><script type="application/json" id="graph">' + data + '</script><script>document.getElementById("raw").textContent=JSON.stringify(JSON.parse(document.getElementById("graph").textContent),null,2)</script>'
    destination.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.results.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "audit.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
    with (args.out / "observations.jsonl").open("x") as stream:
        for row in summary["numeric_rows"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    render(summary, args.out / "scaffold.html")
    print(json.dumps({k: summary[k] for k in ("job_id", "files_verified", "native_argmax_rows_verified",
                     "own_output_feedback_links_verified", "typed_payloads_checked", "sham_logits_and_first_kv_exact")}))


if __name__ == "__main__":
    main()
