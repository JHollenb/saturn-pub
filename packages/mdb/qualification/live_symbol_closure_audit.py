"""Saved-only audit of canonical symbol joins and native causal futures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from mdb.project import Project


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(root, descriptor):
    path = root / descriptor["path"]
    if sha(path) != descriptor["sha256"]:
        raise ValueError("saved array checksum mismatch")
    raw = np.load(path, allow_pickle=False).tobytes()
    if descriptor["dtype"] == "torch.bfloat16":
        value = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    elif descriptor["dtype"] == "torch.float32":
        value = np.frombuffer(raw, dtype="<f4")
    else:
        raise ValueError("unsupported saved dtype")
    value = value.reshape(descriptor["shape"])
    if not np.isfinite(value).all():
        raise ValueError("nonfinite saved native value")
    return value


def source_values(root, cut_id, row):
    """Read sealed BF16 page bytes with NumPy, without hydrating a Session."""
    store = root / "project/store"
    cut = read(store / "cuts" / (cut_id + ".json"))
    page = read(store / "pages" / (cut["slots"]["values"] + ".json"))
    path = store / "blobs" / (page["tensor_blob"] + ".safetensors")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != page["tensor_blob"]:
        raise ValueError("native value blob digest mismatch")
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_size])
    values = []
    for layer in range(19, 28):
        spec = header[page["encoded"]["items"][layer]["name"]]
        if spec["dtype"] != "BF16":
            raise ValueError("native value page dtype changed")
        lo, hi = spec["data_offsets"]
        bits = np.frombuffer(raw[8 + header_size + lo:8 + header_size + hi], dtype="<u2")
        values.append(bits.reshape(spec["shape"])[:, :, row].copy())
    return np.stack(values)


def audit(root, output):
    output.mkdir(exist_ok=False)
    inventory = read(root / "output-manifest.json")["files"]
    for entry in inventory:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("retained inventory mismatch: " + entry["path"])
    project = Project(root / "project", read_only=True)
    project.store.load = lambda *args, **kw: (_ for _ in ()).throw(ValueError("metadata join hydrated state"))
    rows, uses, versions, addresses = [], 0, 0, 0
    for session in project.sessions():
        sid = session["session_id"]
        selected = project.session(sid)["selected"]
        retained = project.symbol_lifecycles(sid)
        versions += len(retained)
        latest = {}
        for record in retained:
            body = record["lifecycle"]
            previous = latest.get(body["binding_fingerprint"])
            if previous is None or len(body["uses"]) > len(previous["lifecycle"]["uses"]):
                latest[body["binding_fingerprint"]] = record
            for item in record["joins"]["uses"]:
                if item["receipt_closure"] != "native-parent-result-matched" or item["consumer_support"] != "native-contract-matched":
                    raise ValueError("canonical use lost native closure")
        if len(latest) != len(project.symbols(sid)):
            raise ValueError("registered binding/lifecycle coverage differs")
        for key, record in latest.items():
            body = record["lifecycle"]
            occurrence = body["definitions"][0]
            binding = body["binding"]
            port = binding["locations"][0]
            reverse = project.symbols_at(sid, port["slot"], cut_id=occurrence["source_cut"], selector=port["selector"])
            if key not in {r["binding_fingerprint"] for r in reverse["matches"]}:
                raise ValueError("canonical reverse lookup lost source binding")
            uses += len(body["uses"])
            addresses += len(binding["locations"])
            rows.append({"session_id": sid, "symbol": body["symbol"], "binding_fingerprint": key,
                "lifecycle_id": record["lifecycle_id"], "source_cut": occurrence["source_cut"],
                "clock": occurrence["clock"]["reading"], "locations": binding["locations"],
                "metadata": occurrence["metadata"], "uses": body["uses"], "claim": binding["claim"],
                "registered": True})
        if project.session(sid)["selected"] != selected:
            raise ValueError("metadata navigation changed selected native cut")
    results, argmax_rows, feedback, payloads, shams, persistence = [], 0, 0, 0, 0, 0
    report = read(root / "report.json")
    for short in report["cases"]:
        result = read(root / (short["id"] + "-result.json"))
        a, b = result["observable_tokens"]["recipient"], result["observable_tokens"]["donor"]
        native = result["outcomes"][0]
        base = array(root, native["trajectory"][0]["logits"])[0]
        base_margin = float(np.float32(base[b] - base[a]))
        by_binding = {r["binding_fingerprint"]: r for r in rows}
        native_values = {side: source_values(root,
            by_binding[result["source_formation"][side]["formed_binding"]]["source_cut"],
            result["source_row"]) for side in ("recipient", "donor")}
        summary = []
        for arm in result["outcomes"]:
            side = "donor" if arm["arm"] == "native-donor" else "recipient"
            if result["source_formation"][side]["source_cut"] not in project.cut_ids(arm["session_id"]):
                raise ValueError("arm did not retain the exact shared pre-writer parent")
            local = [r for r in rows if r["session_id"] == arm["session_id"]]
            writer = next(r for r in local if r["symbol"] == "argument.writer-input")
            formed = source_values(root, writer["uses"][0]["result_cut"], result["source_row"])
            for name, reference in native_values.items():
                x = (formed.astype("<u4") << 16).view("<f4")
                y = (reference.astype("<u4") << 16).view("<f4")
                # Worker subtracted BF16 tensors before converting the delta
                # to FP32. Reproduce that rounding, rather than silently
                # replacing its historical observable with FP32 subtraction.
                bits = (x - y).view("<u4")
                rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
                distance = float(np.linalg.norm((rounded.astype("<u4") << 16).view("<f4")))
                if not np.isclose(distance, arm["formation"][name + "_distance"], rtol=1e-6, atol=1e-6):
                    raise ValueError("source-formation metric differs from native writer-result bytes")
            installed = (by_binding[arm["installed_source_binding"]] if "installed_source_binding" in arm
                else next(r for r in local if r["symbol"] == "argument.source-values"))
            original = source_values(root, installed["source_cut"], result["source_row"])
            trajectory = arm["trajectory"]
            for step, row in enumerate(trajectory):
                current = source_values(root, by_binding[row["source_binding"]]["source_cut"], result["source_row"])
                if not np.array_equal(original, current):
                    raise ValueError("original argument carrier changed before a later receiver")
                persistence += 1
                logits = array(root, row["logits"])[0]
                if int(logits.argmax()) != row["next_token_id"] or row["next_token_id"] != arm["own_ids"][step]:
                    raise ValueError("full-vocabulary argmax/committed token mismatch")
                argmax_rows += 1
                if step and row["input_ids"] != trajectory[step - 1]["input_ids"] + [trajectory[step - 1]["next_token_id"]]:
                    raise ValueError("future used a supplied answer")
                feedback += int(step > 0)
            logits = array(root, trajectory[0]["logits"])[0]
            margin = float(np.float32(logits[b] - logits[a]))
            if margin != arm["donor_minus_recipient_margin"]:
                raise ValueError("saved signed margin differs")
            for payload in arm["value_payloads"]:
                before, after = array(root, payload["before"]), array(root, payload["after"])
                mask = np.arange(before.shape[-2]) != result["source_row"]
                if not np.array_equal(before[:, :, mask], after[:, :, mask]):
                    raise ValueError("value intervention escaped source row")
                payloads += 1
            if arm["arm"] == "sham-writer-input":
                if not all(np.array_equal(array(root, x["logits"]), array(root, y["logits"]))
                    for x, y in zip(native["trajectory"], trajectory)):
                    raise ValueError("writer sham changed native future")
                shams += 1
            summary.append({"arm": arm["arm"], "output": arm["own_text"],
                "donor_margin_effect": margin - base_margin, "margin": margin,
                "formation": arm["formation"], "recipient_score": arm["recipient_score"],
                "donor_score": arm["donor_score"],
                "ordinary_argmax_matches": arm["ordinary_argmax_matches"],
                "ordinary_logit_max_abs": arm["ordinary_logit_max_abs"]})
        results.append({"id": short["id"], "source_row": result["source_row"], "arms": summary})
    # Query operations must leave all sealed raw bytes intact.
    for entry in inventory:
        if sha(root / entry["path"]) != entry["sha256"]:
            raise ValueError("saved-only navigation changed raw evidence")
    audit = {"mechanics_status": "valid", "files_verified": len(inventory),
        "canonical_bindings": len(rows), "canonical_locations": addresses,
        "lifecycle_versions": versions, "canonical_uses": uses,
        "native_argmax_rows": argmax_rows, "own_feedback_links": feedback,
        "value_payloads_checked": payloads, "exact_three_token_shams": shams,
        "source_carrier_persistence_checks": persistence,
        "frameworks_imported": [m for m in ("torch", "transformers", "saturn") if m in sys.modules],
        "cases": results, "terminal_status": "not-requested; exploratory",
        "scope": "registered native symbols and bounded mediation; semantic names supplied; whole-slot static slices deferred"}
    for name, body in (("audit.json", audit), ("symbol-table.json", rows)):
        (output / name).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    data = json.dumps(rows).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>Registered MDB symbols</title>
<style>body{font:15px system-ui;margin:32px;background:#111827;color:#e5e7eb}input{padding:10px;width:70%}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #374151;vertical-align:top}code{word-break:break-all;font-size:12px}details{max-width:700px}pre{white-space:pre-wrap}small{color:#9ca3af}</style>
<h1>Registered MDB symbol table</h1><p>Canonical bindings, definitions and receipt-joined uses from native Qwen execution. Semantic names are supplied. Filter by context, arm, symbol or native cut.</p><input id="filter" placeholder="Filter symbols"><p id="count"></p><table><thead><tr><th>Session / symbol</th><th>Native point</th><th>Ports / uses</th><th>Evidence</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=DATA; const esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); function draw(){const q=document.querySelector('#filter').value.toLowerCase();const rows=data.filter(r=>JSON.stringify(r).toLowerCase().includes(q));document.querySelector('#count').textContent=rows.length+' / '+data.length+' registered bindings';document.querySelector('#rows').innerHTML=rows.map(r=>'<tr><td>'+esc(r.session_id)+'<br><strong>'+esc(r.symbol)+'</strong></td><td>'+esc(JSON.stringify(r.clock))+'</td><td>'+r.locations.length+' exact locations<br>'+r.uses.length+' native uses</td><td><code>'+esc(r.source_cut)+'</code><details><summary>Canonical evidence</summary><pre>'+esc(JSON.stringify(r,null,2))+'</pre></details></td></tr>').join('')}document.querySelector('#filter').addEventListener('input',draw);draw();</script>""".replace("DATA", data)
    (output / "symbol-table.html").write_text(page)
    print(json.dumps({k: v for k, v in audit.items() if k != "cases"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
