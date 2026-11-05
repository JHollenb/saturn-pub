"""Saved-only context-transfer audit. No model construction or execution."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_tree(value):
    if isinstance(value, Mapping):
        return {k: json_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_tree(v) for v in value]
    return value


def rgb_score(value):
    if value.dtype != np.uint8 or value.shape != (512, 512, 3):
        raise ValueError("reader requires original512px uint8 RGB")
    value = value.astype(np.float64)
    chroma = (value[..., 2] - value[..., 0]) / (value.sum(-1) + 1.0)
    return float(chroma[:, 256:].mean() - chroma[:, :256].mean())


def difference(value, reference):
    delta = np.asarray(value, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return dict(exact=bool(np.array_equal(value, reference)), rms=float(np.sqrt(np.mean(delta**2))),
                mae=float(np.mean(np.abs(delta))), max_abs=float(np.max(np.abs(delta))),
                changed_scalars=int(np.count_nonzero(delta)), scalars=int(delta.size))


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite BF16 operand")
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def writer_equation(image, ports):
    # Distinct NumPy implementation of the native eager BF16 op boundaries.
    after_attention = bf16(image + bf16(ports["gate_attention"] * ports["attention_used"]))
    return bf16(after_attention + bf16(ports["gate_mlp"] * ports["mlp_used"]))


@lru_cache(maxsize=8)
def load_cut(root, fingerprint):
    from saturn_pub.store import LocalStore
    return LocalStore(root).load(fingerprint, device="cpu")


def read_slot(root, fingerprint, name):
    import torch
    value = load_cut(root, fingerprint).payload[name].detach().cpu()
    # NumPy cannot represent BF16 directly. Widen exactly; retain integer types.
    return (value.float() if value.dtype == torch.bfloat16 else value).numpy()


def analyze(run, output):
    from mdb.forecast_transaction import verify_transaction, load_forecast_tree

    here = Path(__file__).resolve().parent
    helpers = {name: runpy.run_path(str(here / name)) for name in (
        "flux_family_custody.py", "analyze_flux_mlp_forecast.py",
        "summarize_context_window.py")}
    custody = helpers["flux_family_custody.py"]["verify"](run)
    data = run / "extracted/binding-minimal-result"
    config, summary, rows, receipts, prefix = [read(data / n) for n in (
        "plan.json", "summary.json", "rows.json", "receipts.json", "prefix.json")]
    request = read(run / "request.json")
    worker = runpy.run_path(str(run / "payload/flux_context_transfer.py"))
    assert config == request["config"] and summary == read(run / "report.json")["report"]
    assert summary["counts"] == worker["expected_counts"]() and len(receipts) == summary["counts"]["receipts"]
    assert summary["request_sha256"] == request["request_sha256"] == custody["request_sha256"]
    for name, value in request["source_hashes"].items():
        assert sha(run / "payload" / name) == value
    transaction = verify_transaction(data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert transaction["verified"] and transaction["phase"] == "closed"
    frozen = json_tree(load_forecast_tree(data / "transaction",
        read(data / "transaction/forecasts/hypotheses.json")["predictions"]))
    assert frozen["hypotheses"] == config["hypotheses"] and frozen["arms"] == config["arms"]
    assert frozen["counts_at_seal"] == {k: int(k == "loads") for k in summary["counts"]}
    typed = helpers["analyze_flux_mlp_forecast.py"]["audit_store"](data / "store")
    slot = lambda fp, name: read_slot(data / "store", fp, name)
    fields = lambda fp: {v["key"]["value"]: v["value"] for v in typed["cuts"][fp]["payload"]["items"]}
    point = lambda fp: helpers["analyze_flux_mlp_forecast.py"]["unpack"](typed["cuts"][fp]["execution_point"])
    def at(fp, **expected):
        return {key: point(fp)[key] for key in expected} == expected
    public = helpers["summarize_context_window.py"]["public_receipt"]
    by_receipt = {}
    for item in receipts:
        body = item["body"]
        public(body)
        assert body == read(data / "store/receipts" / (body["fingerprint"] + ".json"))
        if body["fingerprint"] in by_receipt:
            assert by_receipt[body["fingerprint"]] == body
        by_receipt[body["fingerprint"]] = body
    assert sum(r["body"]["operation"] == "continue" for r in receipts) == summary["counts"]["native_transitions"]
    assert sum(r["body"]["operation"] == "apply" for r in receipts) == summary["counts"]["acts"]

    def array(record):
        path = data / record["path"]
        assert sha(path) == record["sha256"]
        value = np.load(path, allow_pickle=False)
        assert list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"]
        assert np.isfinite(value).all()
        return value

    def chain(name, parent, final, steps):
        path = [r["body"] for r in receipts if r["row"] == name]
        assert sum(r["operation"] == "continue" for r in path) == steps
        for body in path:
            assert body["parent"] == parent
            parent = body["result"]
        assert parent == final
        return path

    def action(a, spec=None):
        before, after = fields(a["before"]), fields(a["after"])
        name = a["slot"]
        assert a["act"]["writes"] == [name] and before.keys() == after.keys()
        assert all(before[k] == after[k] for k in before if k != name)
        assert a["receipt"] == by_receipt[a["receipt"]["fingerprint"]]
        assert a["receipt"]["parent"] == a["before"] and a["receipt"]["result"] == a["after"]
        if "source_cut" in a:
            assert name in spec["restore"] and a["source_cut"] == rows[0]["marks"]["before_restore"]
            assert point(a["before"]) == point(a["source_cut"])
            expected = slot(a["source_cut"], name)
        elif name == "image":
            assert a["before"] == config["parents"]["base"] or point(a["before"]) == point(config["parents"]["base"])
            expected = slot(config["parents"]["donor"], name)
        else:
            p = point(a["before"])
            assert name == "conditioning" and p["phase"] == "project" and p["logical_step"] in (2, 3)
            expected = slot(config["parents"]["base"], name).copy()
            if spec["edited"] and p["logical_step"] == 2:
                ix = config["conditioning_rows"]
                expected[:, ix] = slot(config["parents"]["donor"], name)[:, ix]
        assert np.array_equal(slot(a["after"], name), expected)

    chain("prefix", config["parents"]["base"], prefix["common"], 23)
    action(prefix["action"])
    assert prefix["parent"] == config["parents"]["base"]
    assert at(prefix["common"], logical_step=2, phase="project", index=None)
    specs = {s["id"]: s for s in config["arms"]}
    specs["validation"] = dict(id="validation", edited=False, restore=[], port=None)
    assert len(rows) == 8 and {r["id"] for r in rows} == set(specs)
    images, latents, ports, outcomes = {}, {}, {}, {}
    for row in rows:
        name = row["id"]
        spec = specs[name]
        assert all(row[k] == v for k, v in spec.items())
        parent = config["parents"]["base"] if name == "validation" else prefix["common"]
        assert row["parent"] == parent
        chain(name, parent, row["final"], 79 if name == "validation" else 56)
        for a in row["actions"]:
            action(a, spec)
        assert len(row["actions"]) == 2 + len(spec["restore"]) + int(name == "validation")
        w = row["writer"]
        assert at(w["entry"], logical_step=2, phase="joint", index=1)
        assert at(w["exit"], logical_step=2, phase="joint", index=2)
        assert any(r["body"]["parent"] == w["entry"] and r["body"]["result"] == w["exit"]
                   and r["body"]["operation"] == "continue" for r in receipts if r["row"] == name)
        ports[name] = {k: array(v) for k, v in w["fields"].items()}
        p = ports[name]
        mod = load_cut(data / "store", w["entry"]).payload["context"]["img_mod"].float().numpy()
        if mod.ndim == 2:
            mod = mod[:, None, :]
        modulation = np.split(mod, 6, axis=-1)
        assert np.array_equal(modulation[2], p["gate_attention"])
        assert np.array_equal(modulation[5], p["gate_mlp"])
        assert np.array_equal(writer_equation(slot(w["entry"], "image"), p), slot(w["exit"], "image"))
        for port in ("attention", "mlp"):
            target = ports["control"][port + "_native"] if spec["port"] == port else p[port + "_native"]
            assert np.array_equal(target, p[port + "_used"])
        assert sha(data / row["png"]) == row["png_sha256"]
        with Image.open(data / row["png"]) as im:
            assert im.mode == "RGB" and im.size == (512, 512)
            images[name] = np.asarray(im).copy()
        latents[name] = array(row["latent"])
        assert np.array_equal(latents[name], slot(row["final"], "latent"))
        outcomes[name] = dict(score=rgb_score(images[name]),
            signed_effect=rgb_score(images[name]) - rgb_score(images["control"]),
            prediction=config["hypotheses"]["categorical"].get(name, "exact control replay"),
            rgb_vs_control=difference(images[name], images["control"]),
            latent_vs_control=difference(latents[name], latents["control"]),
            visual_review="pending; RGB score does not identify objects")
    by = {r["id"]: r for r in rows}
    validation = by["validation"]
    assert fields(validation["marks"]["replayed_prefix"]) == fields(prefix["common"])
    assert fields(validation["final"]) == fields(by["control"]["final"])
    assert validation["png_sha256"] == by["control"]["png_sha256"]
    assert read(data / "prefix-replay.json") == dict(common_payload_exact=True, final_payload_exact=True, png_exact=True,
        shared_control=by["control"]["final"], independent_control=validation["final"],
        scope="Instrument qualification only; lineage fingerprints remain distinct.")
    port_edits = read(data / "port-interventions.json")
    events = [json.loads(line) for line in (data / "port-intervention-events.jsonl").read_text().splitlines()]
    assert len(port_edits) == 2 and len(events) == 4
    assert [e["record"] for e in events if e["phase"] == "prepared"] == port_edits
    for record in port_edits:
        name = record["row"]
        w = by[name]["writer"]
        port = specs[name]["port"]
        assert record["source_row"] == "control" and record["source_entry"] == by["control"]["writer"]["entry"]
        assert record["source_array"] == by["control"]["writer"]["fields"][port + "_native"]
        assert record["destination_entry"] == w["entry"] and record["point"] == point(w["entry"])
        assert record["port"] == ("transformer_blocks.1.attn:return[0]" if port == "attention" else "transformer_blocks.1.ff:return")
        describe = helpers["analyze_flux_mlp_forecast.py"]["descriptor"]
        for state, suffix in (("before", "_native"), ("after", "_used")):
            expected = describe(ports[name][port + suffix], bf16=True)
            assert all(record[state][key] == value for key, value in expected.items())
        consumed = [e for e in events if e["phase"] == "consumed" and e["row"] == name]
        assert len(consumed) == 1
        c = consumed[0]
        assert c["port"] == port and c["destination_exit"] == w["exit"]
        receipt = by_receipt[c["native_receipt"]]
        assert receipt["operation"] == "continue" and receipt["parent"] == w["entry"] and receipt["result"] == w["exit"]
    local = {}
    for name in by:
        w, baseline = by[name]["writer"], by["control"]["writer"]
        local[name] = {key: difference(value, ports["control"][key]) for key, value in ports[name].items()}
        for boundary in ("entry", "exit"):
            for field in ("image", "text"):
                local[name][boundary + "." + field] = difference(slot(w[boundary], field), slot(baseline[boundary], field))
        for field in ("image", "text"):
            local[name]["joint2_exit." + field] = difference(slot(by[name]["marks"]["joint2_exit"], field),
                slot(by["control"]["marks"]["joint2_exit"], field))
        local[name]["final_payload_exact"] = fields(by[name]["final"]) == fields(by["control"]["final"])
        local[name]["project3_payload_exact"] = fields(by[name]["marks"]["project3_input"]) == fields(by["control"]["marks"]["project3_input"])
    interventions = {}
    for name in ("restore-I-A", "restore-I-M"):
        interventions[name] = dict(rgb_vs_restore_I=difference(images[name], images["restore-I"]),
            latent_vs_restore_I=difference(latents[name], latents["restore-I"]),
            signed_consumer_vs_restore_I=rgb_score(images[name]) - rgb_score(images["restore-I"]),
            joint1_image_vs_restore_I=difference(slot(by[name]["writer"]["exit"], "image"),
                                               slot(by["restore-I"]["writer"]["exit"], "image")))
    result = dict(schema="mdb-context-transfer-audit-v1", mechanics_status="verified", instrument_status="applicable; semantic image review separate",
        trend_status="per-arm observations; predictions may miss", terminal_status="unassessed", job_id=summary["job_id"],
        request_sha256=request["request_sha256"], custody=custody, transaction=transaction,
        counts=summary["counts"], unique_receipt_fingerprints=len(by_receipt),
        typed_store=typed["counts"], consumer_contract=config["consumer_contract"],
        outcomes=outcomes, writer_comparisons=local, paired_port_interventions=interventions,
        native_BF16_equations_exact=8, saved_context_gate_joins=16,
        prefix_replay_exact=True, port_interventions_verified=2, new_model_calls_in_audit=0,
        worker_seconds=summary["worker_seconds"],
        audit_sources={p.name: sha(p) for p in [Path(__file__), *(here / n for n in helpers)]})
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    sheet = Image.new("RGB", (4 * 256, 2 * 280), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(images.items()):
        x, y = (i % 4) * 256, (i // 4) * 280
        sheet.paste(Image.fromarray(image).resize((256, 256)), (x, y + 24))
        draw.text((x + 4, y + 6), name, fill="black")
    sheet.save(output.with_suffix(".png"))
    return dict(mechanics_status=result["mechanics_status"], counts=result["counts"],
                effects={k: v["signed_effect"] for k, v in outcomes.items()},
                local_exact={k: dict(image_exit=v["exit.image"]["exact"], mlp_input=v["mlp_input"]["exact"],
                                     native_mlp=v["mlp_native"]["exact"], final=v["final_payload_exact"])
                             for k, v in local.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.run.resolve(), args.output.resolve()), sort_keys=True))
