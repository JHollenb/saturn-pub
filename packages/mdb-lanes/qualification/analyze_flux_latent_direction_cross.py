"""Authenticated saved-vector responses for a two-receiver/two-direction cross."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELLS = ("c1-o0", "c0-o1")
ACCOUNT_SHA = "bbe68aee19395c081e2115247ac4731171bc606d2f36ee15f944cbbd4c42add8"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def numeric():
    path = HERE / "analyze_flux_native_corner.py"
    require(sha(path) == ACCOUNT_SHA, "qualified saved-vector account changed")
    return runpy.run_path(str(path))


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), "finite matched vector shapes required")
    length = float(np.sqrt(np.square(a).sum() * np.square(b).sum()))
    return float(np.sum(a * b)) / length if length else None


def factorial(predicted, native):
    """Signed raw-coordinate interaction; no scalar-energy causal allocation."""
    n = numeric()
    h, l = CELLS
    def interaction(table):
        return table[h, h] - table[h, l] - table[l, h] + table[l, l]
    pg, nf = interaction(predicted), interaction(native)
    def conditional(pg, nf):
        return {"prediction_error": n["stats"](pg - nf, nf), "prediction_cosine": cosine(pg, nf)}
    return {"mathematical_interaction": n["stats"](pg), "native_interaction": n["stats"](nf),
        "interaction_prediction_error": n["stats"](pg - nf, nf), "cosine": cosine(pg, nf),
        "receiver_contrasts": {d: {"native": n["stats"](native[h, d] - native[l, d]),
            "mathematical": n["stats"](predicted[h, d] - predicted[l, d]),
            "native_response_cosine": cosine(native[h, d], native[l, d]),
            **conditional(predicted[h, d] - predicted[l, d], native[h, d] - native[l, d])} for d in CELLS},
        "direction_contrasts": {r: {"native": n["stats"](native[r, h] - native[r, l]),
            "mathematical": n["stats"](predicted[r, h] - predicted[r, l]),
            "native_response_cosine": cosine(native[r, h], native[r, l]),
            **conditional(predicted[r, h] - predicted[r, l], native[r, h] - native[r, l])} for r in CELLS},
        "scope": "Fixed-coordinate signed contrasts. Receiver effects include BF16 installation differences; these are neither Jacobians nor semantic factor labels."}


def account(p, q, r, m, input_rms):
    a = numeric()["response_account"](p, q, r, m, input_rms)
    return {"mathematical_response": a["D_G_P_minus_Q"], "native_response": a["D_F_R_minus_M"],
        "prediction_error": a["error_DG_minus_DF"], "cosine": a["cosine_DG_DF"], "dot": a["dot_DG_DF"],
        "effective_input_rms": input_rms, "native_finite_response_ratio": a["native_finite_response_ratio"],
        "mathematical_finite_response_ratio": a["mathematical_finite_response_ratio"], "native_baseline_rms": a["native_field_rms"]}


def construction_audit(parent_data, md, inputs, engine):
    """Reconstruct declared inputs using original arrays and a separate cast."""
    import torch

    n = numeric()
    originals, deltas, records = {}, {}, {}
    for c in CELLS:
        rows = (parent_data / "native" / (c + ".jsonl")).read_text().splitlines()
        native_row = json.loads(rows[27])
        require(inputs["receivers"][c]["cut"] == native_row["capture"], "receiver source cut differs")
        native = n["load_field"](parent_data, native_row["outputs"]["latent"]).astype(np.float64)
        program = n["program"](parent_data, c)
        predicted = n["load_field"](parent_data, program["transitions"][27]["outputs"]["latent"]).astype(np.float64)
        deltas[c], originals[c] = predicted - native, native
        cast = torch.from_numpy((native + deltas[c]).astype(np.float32)).bfloat16().float().numpy()
        require(np.array_equal(cast, predicted), "independent unit-direction reconstruction changed P1")
        records[c] = {"source_direction_rms": n["stats"](deltas[c])["rms"], "unit_P1_reconstruction_exact": True}
    rho = min(np.sqrt(np.mean(v * v)) for v in deltas.values())
    for c in CELLS:
        sham = engine["read_array"](md, inputs["branches"][c + ".sham"]["input_latent"])
        require(np.array_equal(sham, originals[c]), "sealed zero-dose input is not original N1")
        for d in CELLS:
            b = c + ".d-" + d
            scale = float(rho / np.sqrt(np.mean(deltas[d] * deltas[d])))
            independent = torch.from_numpy((originals[c] + scale * deltas[d]).astype(np.float32)).bfloat16().float().numpy()
            supplied = engine["read_array"](md, inputs["branches"][b]["input_latent"])
            require(np.array_equal(independent.view(np.uint32), supplied.view(np.uint32)), "sealed input differs from independent declared construction")
            records[b] = {"scale": scale, "installed_bytes_match": True,
                "actual_effective_change": n["stats"](supplied.astype(np.float64) - originals[c])}
    return {"requested_rms_independently_reconstructed": float(rho), "records": records,
        "cast_authority": "Installed CPU torch FP32→BF16→FP32, independent of producer NumPy bit-round implementation.", "model_calls": 0}


def consumer_audit(data, md, inputs, programs, summary, engine):
    """Independent sealed bytes -> Act -> native history / VAE cut joins."""
    independent = runpy.run_path(str(HERE / "analyze_flux_prompt_factorial.py"))["independent"]()
    evidence = independent["Evidence"](data, data / "unused-parameter-cache", {})
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require(len(emissions) == 516 and [e["sequence"] for e in emissions] == list(range(516)), "physical receipt occurrence census differs")
    receipts = [evidence.receipts[e["receipt"]] for e in emissions]
    require(sum(r["operation"] == "continue" for r in receipts) == 504
        and sum(r["operation"] == "apply" for r in receipts) == 12
        and sum(e["kind"] == "render_acts" for e in emissions) == 6
        and sum(e["kind"] == "native_branch_acts" for e in emissions) == 6, "physical Act/continuation split differs")
    def applied(final, action, parent=None):
        found = [r for r in receipts if r["operation"] == "apply" and r["result"] == final and r["act"] == action]
        require(len(found) == 1 and (parent is None or found[0]["parent"] == parent), "consumer Act receipt/parent/result differs")
    records = {}
    for b in engine["BRANCHES"]:
        c = inputs["branches"][b]["receiver"]
        row = summary["rows"][b]
        parent = inputs["receivers"][c]["cut"]["cut"]
        require(row["source_parent"] == parent, "branch borrowed a different receiver")
        actual = evidence.slot(row["entry"]["cut"], "latent")
        sealed = engine["read_array"](md, inputs["branches"][b]["input_latent"])
        require(np.array_equal(actual, sealed), "native entry differs from sealed intervention bytes")
        before, after = evidence.fields(parent), evidence.fields(row["entry"]["cut"])
        require({k: v for k, v in before.items() if k != "latent"} == {k: v for k, v in after.items() if k != "latent"}, "native entry changed receiver-owned closure")
        applied(row["entry"]["cut"], row["act"], parent)
        history = [json.loads(line) for line in (data / "native" / (b + ".jsonl")).read_text().splitlines()]
        prior = row["entry"]["cut"]
        for pos, item in enumerate(history):
            receipt = item["receipt"]
            require(receipt == evidence.receipts[receipt["fingerprint"]] and receipt["parent"] == prior
                and receipt["operation"] == "continue" and receipt["steps"] == 1
                and receipt["result"] == item["capture"]["cut"] == engine["identity"](item["capture"]["manifest"])
                and (item["position"], item["logical_step"], item["local_index"]) == (pos, 1 + pos // 28, pos % 28), "native ordered receipt/cut chain differs")
            prior = receipt["result"]
        require(len(history) == 84 and prior == row["final"]["cut"], "native terminal is not last consumed state")
        render = row["predicted"]
        applied(render["final"]["cut"], render["act"])
        require(np.array_equal(evidence.slot(render["final"]["cut"], "latent"), engine["read_array"](md, programs[b]["terminal_latent"])), "VAE consumed different mathematical terminal latent")
        fields = evidence.fields(render["final"]["cut"])
        require([evidence.typed["unpack"](fields[k]) for k in ("step", "phase", "cursor")] == [4, "project", 0]
            and all(evidence.typed["unpack"](fields[k]) is None for k in ("image", "text", "context", "noise_prediction")), "mathematical render clock/reset differs")
        require({k: v for k, v in before.items() if k not in ("latent", "step", "phase", "cursor")}
            == {k: v for k, v in fields.items() if k not in ("latent", "step", "phase", "cursor")}, "mathematical render changed receiver conditioning/IDs/schedule")
        records[b] = {"input_install_exact": True, "receiver_closure_exact": True, "ordered_native_transitions": 84,
            "render_consumes_sealed_terminal": True}
    return {"records": records, "native_transitions": 504, "acts": 12, "receipt_occurrences": 516,
        "scope": "Saved public receipts, typed stores and arrays independently joined; no model execution."}


def analyze(run, *, math_run, parent, audit, output):
    from PIL import Image

    from mdb_lanes.forecast_transaction import verify_transaction

    run, math_run, parent, audit, output = map(lambda p: Path(p).resolve(), (run, math_run, parent, audit, output))
    require(not output.exists(), "fresh analysis output required")
    launch = runpy.run_path(str(HERE / "submit_flux_latent_direction_native.py"))
    binding = launch["validate_sources"](parent, math_run, audit=audit)
    req, collection, job = [read(run / n) for n in ("request.json", "collection.json", "job.json")]
    launch["validate_request"](req)
    launch["validate_collected_job"](job, collection["job_id"], req, collection)
    require(job["result"]["diagnostics"]["command"] == read(run / "admission-command.json")["command"], "actual native command differs")
    require(all(sha(run / n) == d for n, d in collection["bindings"].items()), "native collection metadata changed")
    require(all(sha(run / "payload" / n) == d for n, d in {**req["source_hashes"], **req["runtime_wheels"]}.items()), "native payload/runtime changed")
    require(sha(run / "worker.log") == job["result"]["artifacts"]["job_log"]["sha256"], "actual native log changed")
    data, md = run / "extracted" / launch["OUTPUT"], math_run / "extracted" / launch["MATH_OUTPUT"]
    inventory = launch["C"]["inventory"](data)
    require(inventory == read(run / "collection-remote-before.json") == read(run / "collection-remote-after.json")
        == read(run / "collection-local-inventory.json"), "native inventory differs")
    launch["verify_collected_manifest"](data, inventory, job["job_id"], req, collection, run=run)
    math_inventory = launch["C"]["inventory"](md)
    require(math_inventory == read(math_run / "collection-local-inventory.json"), "mathematical inventory changed")
    launch["verify_math_manifest"](md, math_inventory, binding["math_job_id"], read(math_run / "request.json"))
    parent_data = parent / "extracted/flux-prompt-factorial-result"
    require(launch["C"]["inventory"](parent_data) == read(parent / "collection-local-inventory.json"), "original source inventory changed")
    engine = launch["engine"]()
    inputs, programs, math_binding = engine["load_math"](md, parent_job_id=binding["parent_job_id"], parent_request_sha256=binding["parent_request_sha256"])
    require(math_binding["cohort_seal_sha256"] == req["config"]["math_cohort_sha256"], "different sealed math cohort")
    summary = read(data / "summary.json")
    transactions = {c: verify_transaction(data / "transactions" / c, expected_seal_sha256=summary["seals"][c]["seal_sha256"], max_bytes=64 << 20) for c in CELLS}
    events = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    first_native = next(i for i, e in enumerate(events) if e["kind"] == "native_transition")
    first_image = next(i for i, e in enumerate(events) if e["kind"] == "predicted_image")
    predicted_events = [e for e in events[:first_native] if e["kind"] == "predicted_image"]
    require(len(predicted_events) == 6 and {e["branch"] for e in predicted_events} == set(engine["BRANCHES"])
        and {e["receiver"] for e in events[:first_image] if e["kind"] == "receiver_sealed"} == set(CELLS)
        and any(e["kind"] == "cohort_sealed" for e in events[:first_image])
        and all(e["counts"]["adapter_steps"] == 0 and all(v == 0 for v in e["target_calls"].values()) for e in predicted_events), "all unique forecasts/seals/images must precede native targets")
    input_audit = construction_audit(parent_data, md, inputs, engine)
    consumption = consumer_audit(data, md, inputs, programs, summary, engine)
    n = numeric()
    histories = {b: [json.loads(line) for line in (data / "native" / (b + ".jsonl")).read_text().splitlines()] for b in engine["BRANCHES"]}
    require(all(len(v) == 84 for v in histories.values()), "native suffix length differs")
    def field(root, ref):
        return n["load_field"](root, ref).astype(np.float64)
    input_vectors, effective, responses, native_replay = {}, {}, {}, {}
    for c in CELLS:
        base = c + ".sham"
        baseline = engine["read_array"](md, inputs["branches"][base]["input_latent"]).astype(np.float64)
        for d in CELLS:
            b = c + ".d-" + d
            vector = engine["read_array"](md, inputs["branches"][b]["input_latent"]).astype(np.float64) - baseline
            input_vectors[c, d], effective[b] = vector, n["stats"](vector)
            rows = []
            for pos, p in enumerate(programs[b]["transitions"]):
                if not any(v["archive"] is not None for v in p["outputs"].values()):
                    continue
                q, r, m = programs[base]["transitions"][pos], histories[b][pos], histories[base][pos]
                require((p["logical_step"], p["local_index"]) == (r["logical_step"], r["local_index"]) == (m["logical_step"], m["local_index"]), "response clocks differ")
                fields = {k: account(field(md, p["outputs"][k]), field(md, q["outputs"][k]),
                    field(data, r["outputs"][k]), field(data, m["outputs"][k]), effective[b]["rms"]) for k in p["outputs"]}
                rows.append({"position": pos, "logical_step": p["logical_step"], "local_index": p["local_index"], "fields": fields})
            responses[b] = rows
        old = [json.loads(line) for line in (parent / "extracted/flux-prompt-factorial-result/native" / (c + ".jsonl")).read_text().splitlines()]
        checks = [{k: v["descriptor"] == old[28 + i]["outputs"][k]["descriptor"] for k, v in row["outputs"].items()} for i, row in enumerate(histories[base])]
        require(checks == [row["native_replay_diagnostic"] for row in histories[base]], "historical replay diagnostics disagree with independent hashes")
        old_png = read(parent_data / "summary.json")["rows"][c + ".native"]["image"]
        require(sha(parent_data / old_png["path"]) == old_png["sha256"], "historical native PNG changed")
        same_png = sha(data / summary["rows"][base]["image"]["path"]) == old_png["sha256"]
        require(same_png == summary["rows"][base]["baseline_png_exact_prior"], "historical PNG diagnostic differs")
        native_replay[c] = {"fields_compared": sum(len(v) for v in checks), "all_field_hashes_match": all(all(v.values()) for v in checks),
            "terminal_png_exact": same_png}
    geometry = {d: {"effective_input_cosine_across_receivers": cosine(input_vectors[CELLS[0], d], input_vectors[CELLS[1], d]),
        "effective_rms_by_receiver": {c: effective[c + ".d-" + d]["rms"] for c in CELLS}} for d in CELLS}
    contrasts = []
    for row in responses[CELLS[0] + ".d-" + CELLS[0]]:
        pos = row["position"]
        fields = {}
        for k in row["fields"]:
            pg, nf = {}, {}
            for c in CELLS:
                for d in CELLS:
                    b, base = c + ".d-" + d, c + ".sham"
                    pg[c, d] = field(md, programs[b]["transitions"][pos]["outputs"][k]) - field(md, programs[base]["transitions"][pos]["outputs"][k])
                    nf[c, d] = field(data, histories[b][pos]["outputs"][k]) - field(data, histories[base][pos]["outputs"][k])
            fields[k] = factorial(pg, nf)
        contrasts.append({"position": pos, "logical_step": row["logical_step"], "local_index": row["local_index"], "fields": fields})
    pngs = {}
    for b in engine["BRANCHES"]:
        for arm in ("native", "predicted"):
            ref = summary["rows"][b]["image"] if arm == "native" else summary["rows"][b]["predicted"]["image"]
            require(sha(data / ref["path"]) == ref["sha256"], "actual consumer PNG changed")
            with Image.open(data / ref["path"]) as img:
                require(img.mode == "RGB" and img.size == (512, 512), "raw RGB geometry differs")
                pngs[b, arm] = np.array(img, dtype=np.float64)
    endpoints = {b: {"latent": n["stats"](field(md, programs[b]["terminal_latent"]) - field(data, histories[b][-1]["outputs"]["latent"]),
        field(data, histories[b][-1]["outputs"]["latent"])), "RGB": n["stats"](pngs[b, "predicted"] - pngs[b, "native"], pngs[b, "native"])} for b in engine["BRANCHES"]}
    rgb, pg, nf = {}, {}, {}
    for c in CELLS:
        for d in CELLS:
            b, base = c + ".d-" + d, c + ".sham"
            operands = [pngs[b, "predicted"], pngs[base, "predicted"], pngs[b, "native"], pngs[base, "native"]]
            rgb[b] = account(*operands, effective[b]["rms"])
            pg[c, d], nf[c, d] = operands[0] - operands[1], operands[2] - operands[3]
    def image_response(c, d):
        return next(row for row in responses[c + ".d-" + d] if row["position"] == 25)["fields"]["image"]
    def greater_defined(a, b):
        return a > b if a is not None and b is not None else None
    expectations = [{"id": "high-receiver-greater-normalized-image-response", "direction": d,
        "holds": greater_defined(image_response(CELLS[0], d)["native_finite_response_ratio"], image_response(CELLS[1], d)["native_finite_response_ratio"])} for d in CELLS]
    expectations += [{"id": "positive-response-alignment", "branch": c + ".d-" + d,
        "holds": image_response(c, d)["cosine"] > 0 if image_response(c, d)["cosine"] is not None else None} for c in CELLS for d in CELLS]
    projection = {b: {k: {arm: v[arm]["rms"] for arm in ("mathematical_response", "native_response")}
        for k, v in rows[0]["fields"].items() if k != "image"} for b, rows in responses.items()}
    expectations += [{"id": "projection-operand-dependencies", "branch": b, "field": k,
        "holds": v["mathematical_response"] == 0 and v["native_response"] == 0}
        for b, fields in projection.items() for k, v in fields.items()]
    require(launch["C"]["inventory"](data) == inventory and launch["C"]["inventory"](md) == math_inventory, "original evidence changed during analysis")
    output.mkdir(parents=True)
    report = {"schema": "mdb-latent-direction-cross-analysis-v1", "mechanics_status": "authenticated-saved-array-reductions",
        "terminal_status": "unassessed", "scientific_quality_gate": None, "model_calls": 0,
        "native_execution_status": job["state"],
        "manifest_origin": "collector-derived-source-manifest" if collection["schema"] == "mdb-latent-direction-native-recovery-collection-v1" else "producer-remote-manifest",
        "counts": summary["counts"], "native_job_id": job["job_id"], "math_job_id": binding["math_job_id"],
        "inputs": inputs, "effective_input_changes": effective, "input_geometry": geometry, "responses": responses,
        "factorial": contrasts, "RGB": rgb, "RGB_factorial": factorial(pg, nf), "endpoints": endpoints, "native_replay": native_replay,
        "projection_invariance": projection, "expectations": expectations, "original_expectations": read(data / "expectations.json"),
        "native_transactions": transactions, "source_bindings": binding,
        "input_construction_audit": input_audit, "consumer_join_audit": consumption,
        "scope": "Known selected contexts, three newly constructed edited inputs and one known input control. Finite responses include receiver-dependent BF16 installation. Raw-coordinate contrasts are not semantic labels; no independent new recurrence recomputation during this reduction."}
    path = output / "analysis.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    refs = [run / n for n in ("request.json", "job.json", "collection.json")] + [math_run / n for n in ("request.json", "job.json", "collection.json", "result.json")]
    refs += [parent / "collection.json", md / "cohort-seal.json", md / "inputs.json", data / "summary.json"]
    if collection["schema"] == "mdb-latent-direction-native-recovery-collection-v1":
        refs += [run / "collection-source-manifest.json"]
    receipt = {"schema": "mdb-latent-direction-analysis-receipt-v1", "source_sha256": sha(__file__), "report_sha256": sha(path),
        "inputs": {str(p): sha(p) for p in refs}, "model_calls": 0, "independent_predictor_recomputation": False,
        "native_execution_status": report["native_execution_status"], "manifest_origin": report["manifest_origin"],
        "reader_custody_sources": {"launcher": sha(HERE / "submit_flux_latent_direction_native.py"), "custody": launch["CUSTODY_SHA256"]},
        "original_native_and_math_inventories_unchanged": True}
    (output / "analysis-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), "report_sha256": sha(path), "expectations": expectations, "native_replay": native_replay}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--math-run", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.run, math_run=a.math_run, parent=a.parent, audit=a.audit, output=a.out)))
