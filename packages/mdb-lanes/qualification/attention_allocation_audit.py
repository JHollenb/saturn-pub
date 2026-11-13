"""Saved-only QK/allocation/value-flow audit, summaries and inspection view."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def array(root, desc):
    path = root / desc["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != desc["sha256"]:
        raise ValueError("array checksum mismatch")
    raw = np.load(path, allow_pickle=False).tobytes()
    if desc["dtype"] == "torch.bfloat16":
        data = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    elif desc["dtype"] == "torch.float32":
        data = np.frombuffer(raw, dtype="<f4")
    else:
        raise ValueError("unsupported captured dtype")
    result = data.reshape(desc["shape"])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite captured native value")
    return result


def bf16(value):
    bits = np.ascontiguousarray(value, dtype="<f4").view("<u4")
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
    return (rounded.astype("<u4") << 16).view("<f4")


def competition(logits, candidates):
    logits = logits.astype(np.float64)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {name: {"token": token, "logit": float(logits[token]), "probability": float(p[token]),
        "rank": int((logits > logits[token]).sum()) + 1} for name, token in candidates.items()}


def audit(root, output):
    output.mkdir(exist_ok=False)
    inventory = read(root / "output-manifest.json")["files"]
    for item in inventory:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("native inventory mismatch")
    all_rows, summaries, routes, math_rows = [], [], {}, []
    candidates = {"cold": 9255, "hot": 4017, "boiling": 49837}
    decisions, links, controls = 0, 0, 0
    for lane in ("bench", "public"):
        record = read(root / (lane + "-result.json"))
        source = record["source_row"]
        arms = {a["arm"].removeprefix(lane + "-"): a for a in record["results"]}
        baseline = arms["native"]
        for name in ("unobserved-native", "sham"):
            if not all(np.array_equal(array(root, a["logits"]), array(root, b["logits"]))
                for a, b in zip(arms[name]["trajectory"], baseline["trajectory"])):
                raise ValueError("native passive/sham parity mismatch")
            controls += 1
        lane_summary = {"lane": lane, "arms": []}
        native_maps = {}
        for item in baseline["traces"]:
            for layer in item["layers"]:
                native_maps[(item["step"], layer["layer"])] = layer
        for name, arm in arms.items():
            for step, row in enumerate(arm["trajectory"]):
                logits = array(root, row["logits"])[0]
                if int(logits.argmax()) != row["next_token_id"] or arm["own_ids"][step] != row["next_token_id"]:
                    raise ValueError("native full-vocabulary argmax mismatch")
                decisions += 1
                if row["consumed_token_id"] != row["input_ids"][-1]:
                    raise ValueError("native consumed-token mismatch")
                if step:
                    previous = arm["trajectory"][step - 1]
                    if row["input_ids"] != previous["input_ids"] + [previous["next_token_id"]]:
                        raise ValueError("future did not consume its own output")
                    links += 1
            summary = {"arm": name, "output": arm["own_text"],
                "words": competition(array(root, arm["trajectory"][0]["logits"])[0], candidates)}
            if name == "unobserved-native":
                lane_summary["arms"].append(summary)
                continue
            source_maps, attention_changes, head_changes = [], [], []
            for trace in arm["traces"]:
                step = trace["step"]
                step_source = []
                labels = record["token_text"]["donor" if name == "donor-native" else "recipient"] + [
                    arm["trajectory"][i]["next_token_text"] for i in range(step)]
                for item in trace["layers"]:
                    layer = item["layer"]
                    q, k, v, a, returns = [array(root, item[key]) for key in
                        ("query", "key", "value", "attention", "head_return")]
                    if a.shape != (1, 12, 1, len(labels)) or k.shape[1] != 2 or q.shape[1] != 12:
                        raise ValueError("attention capture geometry mismatch")
                    reference = array(root, native_maps[(step, layer)]["attention"])
                    reference_return = array(root, native_maps[(step, layer)]["head_return"])
                    if step == 0:
                        attention_changes.append(float(np.abs(a - reference).max()))
                        head_changes.append(float(np.linalg.norm(returns - reference_return)))
                    repeated_k = np.repeat(k, 6, axis=1)
                    scores = bf16(bf16(q @ repeated_k.swapaxes(-1, -2)) * np.float32(q.shape[-1] ** -.5))
                    ex = np.exp(scores - scores.max(-1, keepdims=True))
                    expected = bf16(ex / ex.sum(-1, keepdims=True))
                    repeated_v = np.repeat(v, 6, axis=1)
                    predicted_return = bf16(a @ repeated_v).transpose(0, 2, 1, 3)
                    math_rows.append({"lane": lane, "arm": name, "step": step, "layer": layer,
                        "attention_max_abs": float(np.abs(a - expected).max()),
                        "attention_exact_fraction": float((a == expected).mean()),
                        "return_max_abs": float(np.abs(returns - predicted_return).max()),
                        "return_exact_fraction": float((returns == predicted_return).mean()),
                        "row_sum_max_abs": float(np.abs(a.sum(-1) - 1).max())})
                    step_source.append(a[0, :, 0, source].tolist())
                    for head in range(12):
                        weights = a[0, head, 0]
                        contribution = weights[source] * v[0, head // 6, source]
                        head_return = returns[0, 0, head]
                        numerator = float(np.linalg.norm(contribution))
                        denominator = float(np.linalg.norm(head_return))
                        all_rows.append({"lane": lane, "arm": name, "step": step,
                            "layer": layer, "head": head, "kv_head": head // 6,
                            "source_mass": float(weights[source]),
                            "entropy_nats": float(-(weights * np.log(np.maximum(weights, 1e-30))).sum()),
                            "source_value_return_norm": numerator, "head_return_norm": denominator,
                            "source_to_total_norm_ratio": numerator / denominator if denominator else None,
                            "source_return_cosine": float(np.dot(contribution, head_return) / (numerator * denominator)) if numerator and denominator else None,
                            "allocation": weights.tolist(), "token_labels": labels,
                            "demonstration_category_mass": float(weights[[6, 14]].sum()),
                            "demonstration_description_mass": float(weights[[2, 10]].sum()),
                            "query_scaffold_mass": float(weights[[16, 17, 19, 20, 21]].sum())})
                source_maps.append(step_source)
            first_rows = [r for r in all_rows if r["lane"] == lane and r["arm"] == name and r["step"] == 0]
            late = [r for r in first_rows if r["layer"] >= 19]
            summary.update(top_late_source_heads=sorted(late, key=lambda r: -r["source_mass"])[:12],
                late_source_mass_sum=float(sum(r["source_mass"] for r in late)),
                late_category_mass_sum=float(sum(r["demonstration_category_mass"] for r in late)),
                first_changed_attention_layer=next((i for i, value in enumerate(attention_changes) if value), None),
                first_changed_head_return_layer=next((i for i, value in enumerate(head_changes) if value), None),
                first_attention_max_abs_by_layer=attention_changes)
            lane_summary["arms"].append(summary)
            routes[(lane, name)] = np.asarray(source_maps)
        summaries.append(lane_summary)
    paired = []
    for name in ("native", "donor-native", "sham", "donor-late", "donor-all"):
        x, y = routes[("bench", name)][0], routes[("public", name)][0]
        paired.append({"arm": name, "source_allocation_max_abs": float(np.abs(x - y).max()),
            "source_allocation_mae": float(np.abs(x - y).mean()),
            "source_allocation_correlation": float(np.corrcoef(x.flatten(), y.flatten())[0, 1])})
    result = {"mechanics_status": "valid", "instrument_status": "passive and sham exact across all three logits arrays in both lanes",
        "files_verified": len(inventory), "native_argmax_rows": decisions, "own_feedback_links": links,
        "exact_controls": controls, "attention_layer_observations": len(math_rows), "head_observations": len(all_rows),
        "source_row": 18, "first_receiver_row": 21, "zero_based_layers_and_heads": True,
        "lanes": summaries, "cross_lane": paired,
        "reconstruction": {key: max(row[key] for row in math_rows) for key in
            ("attention_max_abs", "return_max_abs", "row_sum_max_abs")},
        "terminal_status": "not-requested; exploratory",
        "scope": "Attention is observed routing; source-V coalition causality established separately; individual head necessity not measured"}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "head-observations.json").write_text(json.dumps(all_rows) + "\n")
    (output / "reconstruction.json").write_text(json.dumps(math_rows, indent=2) + "\n")
    np.savez_compressed(output / "source-maps.npz", **{lane + "__" + arm: a for (lane, arm), a in routes.items()})
    export_html(output, all_rows)
    print(json.dumps({key: result[key] for key in ("files_verified", "native_argmax_rows", "own_feedback_links",
        "exact_controls", "attention_layer_observations", "head_observations", "cross_lane", "reconstruction")}))


def export_html(output, rows):
    data = json.dumps(rows).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>Temperature circuit attention</title>
<style>body{font:15px system-ui;margin:28px;background:#0f172a;color:#e2e8f0}select{padding:7px;margin-right:15px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #334155;text-align:center;min-width:43px}th{max-width:95px;white-space:pre-wrap}td.source,th.source{outline:2px solid #fbbf24}small{color:#94a3b8}#wrap{overflow:auto}details{margin-top:20px}pre{white-space:pre-wrap}</style>
<h1>Temperature circuit: native attention allocation</h1><p>Rows are query heads; columns are actual input tokens. Gold marks the source description. Choose the native execution lane, intervention, generation step and layer. Layer/head indices start at zero.</p><p>Attention is routing evidence; individual head necessity is untested. The source-value coalition has a separate native mediation experiment.</p>
<label>Lane <select id="lane"><option>bench</option><option>public</option></select></label><label>Arm <select id="arm"><option>native</option><option>donor-native</option><option>sham</option><option>donor-late</option><option>donor-all</option></select></label><label>Step <select id="step"><option>0</option><option>1</option><option>2</option></select></label><label>Layer <select id="layer"></select></label><p id="caption"></p><div id="wrap"></div><details><summary>Head diagnostics</summary><pre id="details"></pre></details>
<script>const data=DATA;const esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const el=id=>document.getElementById(id);el('layer').innerHTML=Array.from({length:28},(_,i)=>'<option>'+i+'</option>').join('');el('layer').value=20;function draw(){const rows=data.filter(r=>r.lane===el('lane').value&&r.arm===el('arm').value&&r.step===+el('step').value&&r.layer===+el('layer').value);const labels=rows[0].token_labels;el('caption').textContent='Each attention row sums approximately to 100%. The two KV heads each serve six distinct query heads.';el('wrap').innerHTML='<table><thead><tr><th>Query / KV</th>'+labels.map((s,i)=>'<th class="'+(i===18?'source':'')+'">'+i+' '+esc(s)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr><th>H'+r.head+' / KV'+r.kv_head+'</th>'+r.allocation.map((p,i)=>'<td class="'+(i===18?'source':'')+'" style="background:rgba(14,165,233,'+(p*.94)+')">'+(p*100).toFixed(1)+'%</td>').join('')+'</tr>').join('')+'</tbody></table>';el('details').textContent=JSON.stringify(rows.map(({allocation,token_labels,...r})=>r),null,2)}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',draw));draw();</script>""".replace("DATA", data)
    (output / "attention.html").write_text(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.results.resolve(), args.output.resolve())
