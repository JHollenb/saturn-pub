"""Independently audit L/C closure, native receipt paths and visible responses."""
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
    run = custody["source_run"](run_dir,"flux-scene-persistent-result",max_archive_bytes=512 << 20,max_member_bytes=128 << 20)
    data,plan,request = run["data"],run["plan"],run["request"]
    source,summary = (json.loads((data/n).read_text()) for n in ("source.json","summary.json"))
    frozen = run["tree"].decode(json.loads((data/"transaction/forecasts/persistent_state.json").read_text())["predictions"])
    observed = run["tree"].decode(json.loads((data/"transaction/observations/native.json").read_text())["value"])
    if frozen!=plan["hypotheses"] or observed["counts"]!=summary["counts"] or observed["numerical_RGB_forecast"]:
        raise ValueError("sealed prediction/observation differs")
    if source["request_sha256"]!=request["request_sha256"] or source["input_cuts"]!=plan["input_cuts"]:
        raise ValueError("source request bindings differ")
    predecessor = run_dir.parent/"closure-run-02/extracted/flux-scene-state-closure-result"
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
    if set(rows)!={r["id"] for r in plan["rows"]} or len(rows)!=10:
        raise ValueError("row inventory differs")
    inventories,images,coherent,acts = {},{},{},0
    for name,row in rows.items():
        if row!=json.loads((data/"rows"/(name+".json")).read_text()) or row["parent"]!=base.fingerprint:
            raise ValueError("observed row/common parent differs")
        bits = row.get("bits",[0,0])
        donor = parents[row["source"]] if row["kind"]=="cross" else base
        expected = dict(base.payload)
        current = base.fingerprint
        if [a["slot"] for a in row["acts"]]!=[s for s,b in zip(("latent","conditioning"),bits) if b]:
            raise ValueError("Act slot inventory differs")
        for intervention in row["acts"]:
            before,after = (store.load(intervention[k],device="cpu") for k in ("parent","post"))
            slot = intervention["slot"]
            if before.fingerprint!=current or dict(before.execution_point)!=dict(base.execution_point):
                raise ValueError("Act not at matched project frontier")
            expected[slot] = donor.payload[slot]
            if describe(after.payload)!=describe(expected):
                raise ValueError("Act payload closure differs")
            rec = intervention["receipt"]
            if rec!=receipts[rec["fingerprint"]] or rec["parent"]!=before.fingerprint or rec["result"]!=after.fingerprint:
                raise ValueError("Act receipt differs")
            current = after.fingerprint
            acts += 1
        edited = store.load(row["edited_cut"],device="cpu")
        if edited.fingerprint!=current or describe(edited.payload)!=describe(expected):
            raise ValueError("edited state closure differs")
        if bits==[1,1]:
            coherent[name] = describe(edited.payload)==describe(donor.payload)
            if not coherent[name]:
                raise ValueError("coherent corner not source-equivalent")
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
        inventories[name] = inventory
    historical = {n:rows[n]["image"]["sha256"]==h for n,h in plan["historical_pngs"].items()}
    if not all(historical.values()):
        raise ValueError("historical replay differs")
    effects = {}
    base_rgb = images["base"].astype(np.float64)/255
    for factor in ("shapeA","colorB","relation"):
        delta = images[factor+".11"].astype(np.float64)/255-base_rgb
        for bits in ("01","10","11"):
            name = factor+"."+bits
            change = images[name].astype(np.float64)/255-base_rgb
            effects[name] = {"source_directed_RGB_projection":float(np.sum(change*delta)/np.sum(delta**2)),
                             "RGB_mae":float(np.mean(np.abs(change)))}
    predictions = {"H-layout":{"observed":None if inventories["relation.10"]["red_anchor"] is None else inventories["relation.10"]["red_anchor"]["red_position"]=="right"}}
    ca,cb = (inventories["colorB."+bits]["red_anchor"] for bits in ("10","01"))
    predictions["H-color"] = {"observed":None if ca is None or cb is None else ca["other_green_minus_blue"]>cb["other_green_minus_blue"],
                              "10":ca,"01":cb}
    predictions["H-shape"] = {"observed":None,"instrument":"separate informal arm-blinded visual observation required"}
    canvas = Image.new("RGB",(4*300,3*330),"#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i,factor in enumerate(("shapeA","colorB","relation")):
        for j,name in enumerate(("base",*(factor+"."+bits for bits in ("01","10","11")))):
            x,y = j*300,i*330
            canvas.paste(Image.fromarray(images[name]).resize((300,300)),(x,y+30))
            draw.text((x+5,y+8),name,fill="black")
    figure = destination.parent/"persistent-state.png"
    canvas.save(figure)
    report = {"schema":"mdb-flux-scene-persistent-analysis-v1","mechanics_status":"valid","terminal_status":"unassessed",
        "job_id":summary["job_id"],"counts":summary["counts"],"archive_custody":run["source"],
        "historical_controls":historical,"source_equivalent_coherent_corners":coherent,"verified_acts":acts,
        "verified_receipts":len(receipts),"inventories":inventories,"effects":effects,"prediction_observations":predictions,
        "figure":{"path":str(figure),"sha256":sha(figure)},
        "limitations":["Three new dependent donor-assisted futures, seven exact replay controls; one checkpoint/seed",
            "Whole latent swap tests carried history, not an isolated object or relationship register",
            "No independent numerical RGB forecast or new raw-input semantic equation; shape instrument remains informal"]}
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
