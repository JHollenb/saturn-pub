"""Offline audit and pixel observations of the I/T/future-C state factorial."""
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
    run = custody["source_run"](run_dir,"flux-scene-state-closure-result",max_archive_bytes=512 << 20,max_member_bytes=128 << 20)
    data,plan,request = run["data"],run["plan"],run["request"]
    source,summary = (json.loads((data/name).read_text()) for name in ("source.json","summary.json"))
    frozen = run["tree"].decode(json.loads((data/"transaction/forecasts/state_closure.json").read_text())["predictions"])
    observed = run["tree"].decode(json.loads((data/"transaction/observations/native.json").read_text())["value"])
    if frozen != plan["hypotheses"] or observed["counts"] != summary["counts"] or observed["numerical_RGB_forecast"]:
        raise ValueError("sealed prediction/observation differs")
    if source["request_sha256"] != request["request_sha256"] or source["input_cuts"] != plan["input_cuts"]:
        raise ValueError("source bindings differ")
    input_custody = json.loads((data/"input-custody.json").read_text())
    predecessor = run_dir.parent/"run-03/extracted/flux-scene-inventory-result"
    if source["input_custody_sha256"] != sha(data/"input-custody.json") or input_custody["files"] != plan["external_inputs"]:
        raise ValueError("hydrated input declarations differ")
    for relative,binding in plan["external_inputs"].items():
        if (predecessor/relative).stat().st_size != binding["bytes"] or sha(predecessor/relative)!=binding["sha256"]:
            raise ValueError("hydrated inputs differ from independently audited predecessor archive")
    digest = hashlib.sha256(json.dumps(source["checkpoint_files"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest != source["checkpoint_sha256"] or digest != plan["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    store = LocalStore(data/"typed-store")
    receipts,edges = {},{}
    for path in (data/"typed-store/receipts").glob("*.json"):
        record = json.loads(path.read_text())
        custody["public_receipt"](record)
        if record["fingerprint"] != path.stem or record["model_identity"] != source["model_identity"]:
            raise ValueError("receipt source/custody differs")
        receipts[path.stem] = record
        if record["operation"]=="continue":
            if record["parent"] in edges and edges[record["parent"]] != record["result"]:
                raise ValueError("native continuation ambiguous")
            edges[record["parent"]] = record["result"]

    def walk(start,stop):
        current,route = start,[start]
        while current != stop:
            if current not in edges or len(route)>plan["max_adapter_steps"]:
                raise ValueError("native continuation incomplete")
            current = edges[current]
            route.append(current)
        return route

    rows = {r["id"]:r for r in observed["rows"]}
    if set(rows) != {r["id"] for r in plan["rows"]} or len(rows)!=25:
        raise ValueError("state coalition row inventory differs")
    inventories,images,acts,matched_cuts = {},{},0,{}
    for name,row in rows.items():
        if row != json.loads((data/"rows"/(name+".json")).read_text()):
            raise ValueError("row file differs from observation")
        current = row["parent"]
        bits = row.get("bits",[0,0,0])
        donor_id = plan["input_cuts"][row["source"]] if row["kind"]=="coalition" else current
        donor = store.load(donor_id,device="cpu")
        for intervention in row["acts"]:
            walk(current,intervention["parent"])
            before,after = (store.load(intervention[k],device="cpu") for k in ("parent","post"))
            slot = intervention["slot"]
            if describe(after.payload) != describe({**before.payload,slot:donor.payload[slot]}):
                raise ValueError("state intervention differs or changes undeclared closure")
            point = before.execution_point
            if slot=="conditioning":
                if point["logical_step"]!=2 or point["phase"]!="project":
                    raise ValueError("future C changed at a wrong read boundary")
                matched_cuts[name] = before.fingerprint
            elif point["logical_step"]!=1 or point["phase"]!="single" or point["index"]!=0:
                raise ValueError("I/T edit at a wrong boundary")
            rec = intervention["receipt"]
            if rec != receipts[rec["fingerprint"]] or rec["parent"]!=before.fingerprint or rec["result"]!=after.fingerprint:
                raise ValueError("Act receipt chain differs")
            current = after.fingerprint
            acts += 1
        walk(current,row["final_cut"])
        expected_slots = [slot for slot,bit in zip(("image","text","conditioning"),bits) if bit]
        if [a["slot"] for a in row["acts"]] != expected_slots:
            raise ValueError("coalition slot inventory differs")
        final = store.load(row["final_cut"],device="cpu")
        initial = store.load(row["parent"],device="cpu")
        expected_c = donor.payload["conditioning"] if bits[2] else initial.payload["conditioning"]
        if describe(final.payload["conditioning"]) != describe(expected_c):
            raise ValueError("final persistent conditioning differs")
        path = data/row["image"]["path"]
        if sha(path)!=row["image"]["sha256"]:
            raise ValueError("image checksum differs")
        images[name] = np.asarray(Image.open(path).convert("RGB"))
        inventories[name] = helper["pixel_inventory"](images[name])[0]
    for name,parent in matched_cuts.items():
        row = rows[name]
        counterpart = row["factor"]+"."+"".join(map(str,row["bits"][:2]))+"0" if any(row["bits"][:2]) else "base"
        if parent != rows[counterpart]["schedulers"][0]["cut"]:
            raise ValueError("future-C arms do not share the exact same pre-C state")
    historical = {name:rows[name]["image"]["sha256"]==digest for name,digest in plan["historical_pngs"].items()}
    if not all(historical.values()):
        raise ValueError("historical native/edited replay differs")
    base = images["base"].astype(np.float64)/255
    projections = {}
    for factor in ("shapeA","colorB","relation"):
        delta = images["reference."+factor].astype(np.float64)/255-base
        denom = float(np.sum(delta**2))
        for bits in ("001","010","011","100","101","110","111"):
            name = factor+"."+bits
            change = images[name].astype(np.float64)/255-base
            projections[name] = {"source_directed_RGB_projection":float(np.sum(change*delta)/denom) if denom else None,
                                 "RGB_mae":float(np.mean(np.abs(change)))}
    predictions = {}
    for factor in ("shapeA","colorB","relation"):
        a,b = (projections[factor+"."+bits]["source_directed_RGB_projection"] for bits in ("111","100"))
        predictions[factor+".coherence-rank"] = {"111":a,"100":b,"observed":None if a is None or b is None else a>b}
    for name in rows:
        components = inventories[name]["components"]
        red = [c for c in components if c["color"]=="red"]
        others = [c for c in components if c["color"]!="red"]
        inventories[name]["red_anchor"] = {"other_color":others[0]["color"],
            "red_position":"right" if red[0]["centroid"][0]>others[0]["centroid"][0] else "left"} if len(red)==len(others)==1 else None
    predictions["colorB.future-C-only"] = {"observed":None if inventories["colorB.001"]["red_anchor"] is None else inventories["colorB.001"]["red_anchor"]["other_color"]=="green"}
    predictions["relation.future-C-only"] = {"observed":None if inventories["relation.001"]["red_anchor"] is None else inventories["relation.001"]["red_anchor"]["red_position"]=="right"}
    canvas = Image.new("RGB",(5*300,5*330),"#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i,spec in enumerate(plan["rows"]):
        name = spec["id"]
        x,y = (i%5)*300,(i//5)*330
        canvas.paste(Image.fromarray(images[name]).resize((300,300)),(x,y+30))
        draw.text((x+5,y+8),name,fill="black")
    figure = destination.parent/"state-closure.png"
    canvas.save(figure)
    report = {"schema":"mdb-flux-scene-state-closure-analysis-v1","mechanics_status":"valid","terminal_status":"unassessed",
        "job_id":summary["job_id"],"counts":summary["counts"],"archive_custody":run["source"],
        "historical_controls":historical,"verified_acts":acts,"verified_receipts":len(receipts),"matched_pre_C_states":matched_cuts,
        "inventories":inventories,"effects":projections,"prediction_observations":predictions,
        "figure":{"path":str(figure),"sha256":sha(figure)},
        "limitations":["Same four source histories/checkpoint/seed;18 new dependent coalition outcomes plus7 replayed controls",
            "Donor-assisted state-coalition test; recipient latent/history retained; no independent RGB forecast",
            "Coarse colored components/centroid ordering; shape semantics require separate informal visual observations",
            "Working-text replacement and future-C replacement have different intervention times by declared native ABI"]}
    with destination.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"job_id":summary["job_id"],"mechanics_status":"valid","analysis":str(destination),"prediction_observations":predictions}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",type=Path)
    p.add_argument("destination",type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.run_dir,args.destination),sort_keys=True))
