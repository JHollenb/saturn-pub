"""Audit spatial latent closure, native receipt paths and visible scene responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run_dir,destination):
    import numpy as np
    from PIL import Image, ImageDraw
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    here = Path(__file__).resolve().parent
    custody = runpy.run_path(str(here/"summarize_context_window.py"))
    helper = runpy.run_path(str(here/"flux_scene_inventory.py"))
    run = custody["source_run"](run_dir,"flux-scene-spatial-result",max_archive_bytes=512 << 20,max_member_bytes=128 << 20)
    data,plan,request = run["data"],run["plan"],run["request"]
    source,summary = (json.loads((data/n).read_text()) for n in ("source.json","summary.json"))
    frozen = run["tree"].decode(json.loads((data/"transaction/forecasts/spatial_state.json").read_text())["predictions"])
    observed = run["tree"].decode(json.loads((data/"transaction/observations/native.json").read_text())["value"])
    if frozen!=plan["hypotheses"] or observed["counts"]!=summary["counts"] or observed["numerical_RGB_forecast"]:
        raise ValueError("sealed prediction/observation differs")
    if source["request_sha256"]!=request["request_sha256"] or source["input_cuts"]!=plan["input_cuts"]:
        raise ValueError("source request bindings differ")
    predecessor = run_dir.parent/"persistent-run-01/extracted/flux-scene-persistent-result"
    hydrated = json.loads((data/"input-custody.json").read_text())
    if source["input_custody_sha256"]!=sha(data/"input-custody.json") or hydrated["files"]!=plan["external_inputs"]:
        raise ValueError("hydrated custody differs")
    for relative,binding in plan["external_inputs"].items():
        path = predecessor/relative
        if path.stat().st_size!=binding["bytes"] or sha(path)!=binding["sha256"]:
            raise ValueError("input differs from independently audited predecessor archive")
    digest = hashlib.sha256(json.dumps(source["checkpoint_files"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest!=source["checkpoint_sha256"] or digest!=plan["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint identity differs")
    store = LocalStore(data/"typed-store")
    parents = {n:store.load(h,device="cpu") for n,h in plan["input_cuts"].items()}
    base = parents["s0c0r0"]
    for name,cut in parents.items():
        if (cut.model_identity!=source["model_identity"] or dict(cut.execution)!=source["execution"]
                or cut.execution_point["logical_step"]!=2 or cut.execution_point["phase"]!="project"
                or dict(cut.execution_point)!=dict(base.execution_point)):
            raise ValueError("source model/clock differs")
        if any(cut.payload[k] is not None for k in ("image","text","context","noise_prediction")):
            raise ValueError("source working state not cleared")
        differing = {k for k in base.payload if describe(base.payload[k])!=describe(cut.payload[k])}
        if differing!=(set() if name=="s0c0r0" else {"latent","conditioning"}):
            raise ValueError("source nuisance closure differs")
    historical_audit = json.loads((run_dir.parent/"persistent-run-01/analysis-v1.json").read_text())
    expected_inventories = {"base":historical_audit["inventories"]["base"],
        **{f:historical_audit["inventories"][f+".11"] for f in ("shapeA","colorB","relation")}}
    if plan["historical_inventories"]!=expected_inventories:
        raise ValueError("historical output-assisted inventories differ")
    spatial = runpy.run_path(str(here/"flux_scene_spatial.py"))
    for factor in ("shapeA","colorB","relation"):
        expected = spatial["supports"](expected_inventories["base"],expected_inventories[factor],
            base.payload["img_ids"].numpy(),height=plan["height"],width=plan["width"])
        if expected!=plan["supports"][factor]:
            raise ValueError("support instrument differs from retained native coordinates")
    receipts,edges = {},{}
    for path in (data/"typed-store/receipts").glob("*.json"):
        record = json.loads(path.read_text())
        custody["public_receipt"](record)
        if record["fingerprint"]!=path.stem or record["model_identity"]!=source["model_identity"]:
            raise ValueError("receipt binding differs")
        receipts[path.stem] = record
        if record["operation"]=="continue":
            if record["parent"] in edges and edges[record["parent"]]!=record["result"]:
                raise ValueError("ambiguous native continuation")
            edges[record["parent"]] = record["result"]

    def walk(start,stop):
        current,length = start,0
        while current!=stop:
            if current not in edges or length>plan["max_adapter_steps"]:
                raise ValueError("native path incomplete")
            current,length = edges[current],length+1

    rows = {r["id"]:r for r in observed["rows"]}
    if set(rows)!={r["id"] for r in plan["rows"]} or len(rows)!=20:
        raise ValueError("row inventory differs")
    inventories,images,acts = {},{},0
    for name,row in rows.items():
        if row!=json.loads((data/"rows"/(name+".json")).read_text()) or row["parent"]!=base.fingerprint:
            raise ValueError("observed row/common parent differs")
        donor = parents[row["source"]] if row["kind"]=="spatial" else base
        mode = row.get("mode","sham")
        modes = (["left","right"] if mode=="both" else [mode]) if row["kind"]!="reference" else []
        if [a["support"] for a in row["acts"]]!=modes:
            raise ValueError("spatial Act sequence differs")
        expected = dict(base.payload)
        current = base.fingerprint
        for intervention,support_mode in zip(row["acts"],modes):
            before,after = (store.load(intervention[k],device="cpu") for k in ("parent","post"))
            if before.fingerprint!=current or dict(before.execution_point)!=dict(base.execution_point) or intervention["slot"]!="latent":
                raise ValueError("Act is not at the common project frontier")
            indices = plan["supports"][row["factor"]]["masks"][support_mode] if row["kind"]=="spatial" else []
            if intervention["indices"]!=indices:
                raise ValueError("Act spatial extent differs")
            value = expected["latent"].clone()
            if indices:
                value[:,indices,:] = donor.payload["latent"][:,indices,:]
            expected["latent"] = value
            if describe(after.payload)!=describe(expected):
                raise ValueError("Act wrote an undeclared row or slot")
            rec = intervention["receipt"]
            if rec!=receipts[rec["fingerprint"]] or rec["parent"]!=before.fingerprint or rec["result"]!=after.fingerprint:
                raise ValueError("Act receipt differs")
            current = after.fingerprint
            acts += 1
        edited = store.load(row["edited_cut"],device="cpu")
        if edited.fingerprint!=current or describe(edited.payload)!=describe(expected):
            raise ValueError("edited state closure differs")
        for boundary in row["schedulers"]:
            walk(current,boundary["cut"])
            current = boundary["cut"]
        walk(current,row["final_cut"])
        final = store.load(row["final_cut"],device="cpu")
        if describe(final.payload["conditioning"])!=describe(expected["conditioning"]):
            raise ValueError("final C closure differs")
        path = data/row["image"]["path"]
        if sha(path)!=row["image"]["sha256"]:
            raise ValueError("PNG checksum differs")
        images[name] = np.asarray(Image.open(path).convert("RGB"))
        inventory = helper["pixel_inventory"](images[name])[0]
        red = [c for c in inventory["components"] if c["color"]=="red"]
        others = [c for c in inventory["components"] if c["color"]!="red"]
        inventory["red_anchor"] = ({"red_position":"right" if red[0]["centroid"][0]>others[0]["centroid"][0] else "left",
            "other_color":others[0]["color"],"other_green_minus_blue":others[0]["median_rgb"][1]-others[0]["median_rgb"][2]}
            if len(red)==len(others)==1 else None)
        inventory["left_to_right_colors"] = [c["color"] for c in inventory["components"]]
        inventory["right_green_minus_blue"] = (inventory["components"][1]["median_rgb"][1]-inventory["components"][1]["median_rgb"][2]
            if len(inventory["components"])==2 else None)
        inventories[name] = inventory
    historical = {n:rows[n]["image"]["sha256"]==h for n,h in plan["historical_pngs"].items()}
    if not all(historical.values()):
        raise ValueError("historical replay differs")
    matched_stacks = {}
    for factor in ("shapeA","colorB","relation"):
        first = rows[factor+".both"]["acts"][0]["post"]
        matched_stacks[factor] = first==rows[factor+".left"]["edited_cut"]
        if not matched_stacks[factor]:
            raise ValueError("stacked right Act did not use the exact left-edited state")
    effects,interactions = {},{}
    base_rgb = images["base"].astype(np.float64)/255
    base_inventory,base_labels = helper["pixel_inventory"](images["base"])
    regions = {side:base_labels==component["component_label"] for side,component in zip(("left","right"),base_inventory["components"])}
    for factor in ("shapeA","colorB","relation"):
        delta = images[factor+".full"].astype(np.float64)/255-base_rgb
        for mode in ("full","left","right","both","outside","shifted"):
            name = factor+"."+mode
            change = images[name].astype(np.float64)/255-base_rgb
            effects[name] = {"source_directed_RGB_projection":float(np.sum(change*delta)/np.sum(delta**2)),
                "RGB_mae":float(np.mean(np.abs(change))),
                "base_region_mae":{side:float(np.mean(np.abs(change[mask]))) for side,mask in regions.items()}}
        mixed = (images[factor+".both"].astype(np.float64)-images[factor+".left"].astype(np.float64)
                 -images[factor+".right"].astype(np.float64)+images["base"].astype(np.float64))/255
        interactions[factor] = {"RGB_mixed_rms":float(np.sqrt(np.mean(mixed**2))),
            "RGB_mixed_signed_mean":float(mixed.mean()),"RGB_mixed_max_abs":float(np.abs(mixed).max()),
            "per_base_region_signed_mean":{side:mixed[mask].mean(0).tolist() for side,mask in regions.items()}}
    predictions = {}
    colors = {n:v["left_to_right_colors"] for n,v in inventories.items()}
    predictions["H-order-left"] = {"observed":colors["relation.left"]==["blue","blue"],"colors":colors["relation.left"]}
    predictions["H-order-right"] = {"observed":colors["relation.right"]==["red","red"],"colors":colors["relation.right"]}
    predictions["H-order-both"] = {"pixel_observed":colors["relation.both"]==["blue","red"],"colors":colors["relation.both"],
        "shape_binding_observed":None,"shape_instrument":"separate informal arm-blinded observation"}
    for mode in ("outside","shifted"):
        predictions["H-location."+mode] = {"observed":colors["relation."+mode]==["red","blue"],"colors":colors["relation."+mode]}
    ca,cb = (inventories["colorB."+mode]["right_green_minus_blue"] for mode in ("right","left"))
    predictions["H-color"] = {"rank_observed":None if ca is None or cb is None else ca>cb,"right":ca,"left":cb,
        "right_colors":colors["colorB.right"],"left_colors":colors["colorB.left"]}
    predictions["H-shape"] = {"observed":None,"instrument":"separate informal arm-blinded observation"}
    canvas = Image.new("RGB",(7*240,3*265),"#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i,factor in enumerate(("shapeA","colorB","relation")):
        for j,name in enumerate(("base",*(factor+"."+mode for mode in ("full","left","right","both","outside","shifted")))):
            x,y = j*240,i*265
            canvas.paste(Image.fromarray(images[name]).resize((240,240)),(x,y+25))
            draw.text((x+5,y+7),name,fill="black")
    figure = destination.parent/"spatial-state.png"
    canvas.save(figure)
    report = {"schema":"mdb-flux-scene-spatial-analysis-v1","mechanics_status":"valid","terminal_status":"unassessed",
        "job_id":summary["job_id"],"counts":summary["counts"],"archive_custody":run["source"],
        "historical_controls":historical,"matched_left_edited_parents":matched_stacks,"verified_acts":acts,
        "verified_receipts":len(receipts),"inventories":inventories,"effects":effects,"output_interactions":interactions,"prediction_observations":predictions,
        "figure":{"path":str(figure),"sha256":sha(figure)},
        "limitations":["Fifteen new dependent donor-assisted futures, five exact replay controls; one checkpoint/seed",
            "Historical RGB supplies geometric windows; no latent-derived semantic object IDs, new-prompt support prediction or learned parser",
            "No independent numerical RGB forecast; shape and binding instrument remains informal"]}

    with destination.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"analysis":str(destination),"job_id":summary["job_id"],"predictions":predictions}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",type=Path)
    p.add_argument("destination",type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.run_dir,args.destination),sort_keys=True))
