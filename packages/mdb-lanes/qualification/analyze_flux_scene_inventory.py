"""Model-free custody audit, pixel inventories and signed intervention effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(run_dir, destination):
    import numpy as np
    from PIL import Image, ImageDraw
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import verify_transaction

    helper = runpy.run_path(str(Path(__file__).with_name("flux_scene_inventory.py")))
    custody = runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))
    run = custody["source_run"](run_dir, "flux-scene-inventory-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    result = run_dir/"extracted"/"flux-scene-inventory-result"
    summary, source, manifest, request = (json.loads(p.read_text()) for p in
        (result/"summary.json", result/"source.json", result/"files.json", run_dir/"request.json"))
    plan = request["config"]
    checkpoint_digest = hashlib.sha256(json.dumps(source["checkpoint_files"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if (checkpoint_digest != source["checkpoint_sha256"] or checkpoint_digest != plan["expected_checkpoint_sha256"]
            or source["source_hashes"] != request["source_hashes"]):
        raise ValueError("checkpoint/source identity differs")
    frozen = run["tree"].decode(json.loads((result/"transaction/forecasts/scene_inventory.json").read_text())["predictions"])
    observed = run["tree"].decode(json.loads((result/"transaction/observations/native.json").read_text())["value"])
    if (frozen["hypotheses"] != plan["hypotheses"] or frozen["supports_sha256"] != sha(result/"supports.json")
            or frozen["state_relationships_sha256"] != sha(result/"state-relationships.json")
            or observed["counts"] != summary["counts"] or observed["numerical_RGB_forecast"]):
        raise ValueError("sealed forecasts/observation differ")
    if source["request_sha256"] != request["request_sha256"] or summary["request_sha256"] != request["request_sha256"]:
        raise ValueError("request binding differs")
    for name, digest in manifest.items():
        if sha(result/name) != digest:
            raise ValueError("result manifest differs: "+name)
    for name, digest in request["source_hashes"].items():
        if sha(run_dir/"payload"/name) != digest:
            raise ValueError("staged source differs: "+name)
    store = LocalStore(result/"typed-store")
    receipts, edges = {}, {}
    for path in (result/"typed-store/receipts").glob("*.json"):
        record = json.loads(path.read_text())
        custody["public_receipt"](record)
        if record["fingerprint"] != path.stem or record["model_identity"] != source["model_identity"]:
            raise ValueError("native receipt custody/source differs")
        receipts[path.stem] = record
        if record["operation"] == "continue":
            if record["parent"] in edges and edges[record["parent"]] != record["result"]:
                raise ValueError("native continuation ambiguous")
            edges[record["parent"]] = record["result"]

    def walk(start, stop):
        route, current = [start], start
        while current != stop:
            if current not in edges or len(route) > plan["max_adapter_steps"]:
                raise ValueError("native receipt lineage incomplete")
            current = edges[current]
            route.append(current)
        return route

    for history in source["histories"].values():
        route = walk(history["initial"], history["post"])
        if history["early"] not in route or history["entry"] not in route:
            raise ValueError("observed boundaries absent from native prefix")
    initial_reference = store.load(source["histories"]["s0c0r0"]["initial"],device="cpu")
    for history in source["histories"].values():
        initial = store.load(history["initial"],device="cpu")
        if initial.execution_point != initial_reference.execution_point or set(initial.payload) != set(initial_reference.payload):
            raise ValueError("initial history schema/clock differs")
        for key in initial.payload:
            if key != "conditioning" and describe(initial.payload[key]) != describe(initial_reference.payload[key]):
                raise ValueError("initial history changes a nuisance slot: "+key)
    import torch

    for name, history in source["histories"].items():
        entry, post = (store.load(history[k], device="cpu") for k in ("entry", "post"))
        arrays = {k: torch.from_numpy(np.load(result/v["path"],allow_pickle=False)).to(torch.bfloat16)
                  for k,v in source["ports"][name].items()}
        rebuilt = (entry.payload["image"] + arrays["gate_attention"]*arrays["attention"]) + arrays["gate_mlp"]*arrays["mlp"]
        if not torch.equal(rebuilt, post.payload["image"]):
            raise ValueError("native sequential attention/MLP equation differs")
    timeline = [json.loads(line) for line in (result/"timeline.jsonl").read_text().splitlines()]
    sealed = next(row for row in timeline if row["kind"] == "predictions_sealed")
    if sealed["counts"]["images"] != 0 or sealed["counts"]["vae_calls"] != 0:
        raise ValueError("image outcome precedes prediction seal")
    verification = verify_transaction(result/"transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    supports = json.loads((result/"supports.json").read_text())
    base_cut = store.load(source["histories"]["s0c0r0"]["post"], device="cpu")
    identifiers = np.load(result/source["image_ids"]["path"], allow_pickle=False)
    support_exact, closure_count = {}, 0
    for factor, history in (("shapeA", "s1c0r0"), ("colorB", "s0c1r0"), ("relation", "s0c0r1")):
        donor = store.load(source["histories"][history]["post"], device="cpu")
        recalculated = helper["propose_support"](base_cut.payload["image"].float().numpy(),
            donor.payload["image"].float().numpy(), identifiers)
        if recalculated != supports[factor]:
            raise ValueError("prefix-only support calculation differs")
        support_exact[factor] = True
    rows = [json.loads((result/"rows"/(spec["id"]+".json")).read_text()) for spec in plan["rows"]]
    if rows != observed["rows"]:
        raise ValueError("sealed observation differs from row files")
    images = {r["id"]: np.asarray(Image.open(result/r["image"]["path"]).convert("RGB")) for r in rows}
    inventories, masks = {}, {}
    for row in rows:
        if sha(result/row["image"]["path"]) != row["image"]["sha256"]:
            raise ValueError("image custody differs")
        final = store.load(row["final_cut"], device="cpu")
        parent = store.load(row["parent"], device="cpu")
        if describe(final.payload["conditioning"]) != describe(parent.payload["conditioning"]):
            raise ValueError("recipient conditioning changed")
        for intervention in row["acts"]:
            post = store.load(intervention["post"], device="cpu")
            value = parent.payload["image"].clone()
            if row["kind"] == "patch":
                donor = store.load(row["source_cut"], device="cpu")
                value[:, row["selected_rows"]] = donor.payload["image"][:, row["selected_rows"]]
            if describe(post.payload) != describe({**parent.payload, "image": value}):
                raise ValueError("typed image intervention differs or writes undeclared state")
            receipt = intervention["receipt"]
            if (receipt != receipts[receipt["fingerprint"]] or receipt["parent"] != parent.fingerprint
                    or receipt["result"] != post.fingerprint):
                raise ValueError("typed act receipt lineage differs")
            closure_count += 1
        route = walk(row["acts"][-1]["post"] if row["acts"] else row["parent"], row["final_cut"])
        if any(witness["cut"] not in route for witness in row["schedulers"]):
            raise ValueError("scheduler witness absent from native suffix")
        inv, mask = helper["pixel_inventory"](images[row["id"]])
        inv["sensitivity"] = [{"chroma": chroma, "components": helper["pixel_inventory"](images[row["id"]], chroma=chroma)[0]["components"]}
                              for chroma in (.12,.24,.32)]
        inventories[row["id"]], masks[row["id"]] = inv, mask
    if not np.array_equal(images["native.s0c0r0"], images["sham"]):
        raise ValueError("native/sham pixel mismatch")
    base_image = images["native.s0c0r0"].astype(np.float64)/255
    base_objects = inventories["native.s0c0r0"]["components"]
    region_masks = {}
    for color in ("red", "blue"):
        candidates = [c for c in base_objects if c["color"] == color]
        if candidates:
            region_masks[color] = masks["native.s0c0r0"] == max(candidates,key=lambda c:c["area"])["component_label"]
    effects, predictions, overlaps = {}, {}, {}
    for factor, history in (("shapeA", "s1c0r0"), ("colorB", "s0c1r0"), ("relation", "s0c0r1")):
        donor_image = images["native."+history].astype(np.float64)/255
        direction = donor_image-base_image
        denominator = float(np.sum(direction**2))
        for mode in ("full", "top", "shifted"):
            name = factor+"."+mode
            delta = images[name].astype(np.float64)/255-base_image
            effects[name] = {"source_directed_RGB_projection": float(np.sum(delta*direction)/denominator) if denominator else None,
                             "RGB_mae": float(np.mean(np.abs(delta))),
                             "region_mae": {color: float(np.mean(np.abs(delta[mask]))) for color,mask in region_masks.items()},
                             "projection_scope": "base0 source1 along full RGB difference; not semantic accuracy or fraction of object restored"}
        full, top = (effects[factor+"."+mode]["source_directed_RGB_projection"] for mode in ("full","top"))
        predictions[factor+".full-over-top"] = {"full": full, "top": top,
            "observed": None if full is None or top is None else full > top}
        coords = supports[factor]["coordinates"]
        heights, widths = sorted({c[0] for c in coords}), sorted({c[1] for c in coords})
        overlaps[factor] = {}
        for mode in ("top", "shifted"):
            support_mask = np.zeros(base_image.shape[:2], dtype=bool)
            for index in supports[factor][mode]:
                y,x = coords[index]
                yi,xi = heights.index(y),widths.index(x)
                y0,y1 = round(yi*512/len(heights)),round((yi+1)*512/len(heights))
                x0,x1 = round(xi*512/len(widths)),round((xi+1)*512/len(widths))
                support_mask[y0:y1,x0:x1] = True
            overlaps[factor][mode] = {color: {"object_coverage": float(np.mean(support_mask[mask])),
                "support_precision": float(np.sum(mask&support_mask)/support_mask.sum())} for color,mask in region_masks.items()}
    for factor, color in (("shapeA","red"),("colorB","blue")):
        a, b = (effects[factor+"."+mode]["region_mae"].get(color) for mode in ("top","shifted"))
        predictions[factor+".top-over-shifted"] = {"top": a, "shifted": b,
            "observed": None if a is None or b is None else a > b}
    # Pure state algebra is recomputed from retained native cuts, independent of worker reports.
    recomputed = {}
    for boundary in ("early", "entry", "post"):
        loaded = {tuple(h["factors"]): store.load(h[boundary],device="cpu") for h in source["histories"].values()}
        recomputed[boundary] = {slot: helper["factorial_diagnostics"]({f: c.payload[slot].float().numpy() for f,c in loaded.items()})
                                for slot in ("image","text","conditioning","latent")}
    for slot in ("attention","mlp"):
        recomputed[slot] = helper["factorial_diagnostics"]({tuple(source["histories"][name]["factors"]):
            np.load(result/ports[slot]["path"],allow_pickle=False) for name,ports in source["ports"].items()})
    original_diagnostics = json.loads((result/"state-relationships.json").read_text())
    diagnostic_residuals = []

    def compare_diagnostics(a,b):
        if isinstance(a,dict):
            if not isinstance(b,dict) or set(a) != set(b):
                raise ValueError("state diagnostic structure differs")
            for key in a:
                compare_diagnostics(a[key],b[key])
        elif isinstance(a,(int,float)):
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError("nonfinite diagnostic")
            diagnostic_residuals.append(abs(a-b))
        elif a != b:
            raise ValueError("state diagnostic interpretation differs")

    compare_diagnostics(original_diagnostics,recomputed)
    canvas = Image.new("RGB",(4*320,5*350),"#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for index,row in enumerate(rows):
        x,y = (index%4)*320,(index//4)*350
        canvas.paste(Image.open(result/row["image"]["path"]).resize((320,320)),(x,y+30))
        draw.text((x+5,y+7),row["id"],fill="black")
    figure = destination.parent/"scene-inventory.png"
    canvas.save(figure)
    output = {"schema": "mdb-flux-scene-inventory-analysis-v1", "mechanics_status": "valid",
        "instrument_status": "bounded chromatic-component instrument; sensitivity retained; semantic shape unknown",
        "terminal_status": "unassessed", "job_id": summary["job_id"], "counts": summary["counts"],
        "verified_result_files": len(manifest), "verified_image_acts": closure_count,
        "archive_custody": run["source"], "verified_receipts": len(receipts), "native_joint_equations": 8,
        "prefix_supports_exact": support_exact, "sham_pixels_exact": True, "transaction": verification,
        "inventories": inventories, "effects": effects, "prediction_observations": predictions,
        "support_object_overlap": overlaps, "state_relationships": recomputed,
        "diagnostic_recomputation_max_abs": max(diagnostic_residuals,default=0.),
        "figure": {"path": str(figure), "sha256": sha(figure)},
        "limitations": ["one checkpoint/seed; eight related histories; no held-out semantic generalization",
            "prompt factors are supplied; chromatic components are pixel observations",
            "support energy is observational; intervention suffix is native and retains recipient conditioning",
            "no numerical RGB forecast; no automatic shape semantics or exhaustive scene attribution"]}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as stream:
        json.dump(output,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"job_id": summary["job_id"], "mechanics_status": "valid", "analysis": str(destination),
            "prediction_observations": predictions}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",type=Path)
    p.add_argument("destination",type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.run_dir,args.destination),sort_keys=True))
