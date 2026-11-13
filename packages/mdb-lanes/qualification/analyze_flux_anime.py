"""Audit native anime executions and geometric effects without a semantic oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run_dir,destination,predecessor=None):
    import numpy as np
    from PIL import Image, ImageDraw
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    here = Path(__file__).resolve().parent
    custody = runpy.run_path(str(here/"summarize_context_window.py"))
    run = custody["source_run"](run_dir,"flux-anime-result",max_archive_bytes=512 << 20,max_member_bytes=128 << 20)
    data,plan,request = run["data"],run["plan"],run["request"]
    source,summary = (json.loads((data/n).read_text()) for n in ("source.json","summary.json"))
    frozen = run["tree"].decode(json.loads((data/"transaction/forecasts/anime.json").read_text())["predictions"])
    observed = run["tree"].decode(json.loads((data/"transaction/observations/native.json").read_text())["value"])
    if frozen!=plan["hypotheses"] or observed["counts"]!=summary["counts"] or observed["numerical_RGB_forecast"]:
        raise ValueError("sealed forecast/observation binding differs")
    if source["request_sha256"]!=request["request_sha256"] or source["source_hashes"]!=request["source_hashes"]:
        raise ValueError("source/request implementation binding differs")
    digest = hashlib.sha256(json.dumps(source["checkpoint_files"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest!=source["checkpoint_sha256"] or digest!=plan["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    store = LocalStore(data/"typed-store")
    parents = {n:store.load(h,device="cpu") for n,h in source["parents"].items()}
    for cut in parents.values():
        if (cut.model_identity!=source["model_identity"] or dict(cut.execution)!=source["execution"]
                or cut.execution_point["logical_step"]!=2 or cut.execution_point["phase"]!="project"
                or any(cut.payload[k] is not None for k in ("image","text","context","noise_prediction"))):
            raise ValueError("parent identity/clock/cleared-state contract differs")
    if plan["stage"]=="spatial":
        if predecessor is None:
            raise ValueError("spatial audit needs independently audited predecessor directory")
        old_data = predecessor/"extracted/flux-anime-result"
        old_source = json.loads((old_data/"source.json").read_text())
        if sha(predecessor/"analysis-v1.json")!=plan["predecessor_audit_sha256"] or old_source["parents"]!=source["parents"]:
            raise ValueError("input source cuts/audit differ")
        hydrated = json.loads((data/"input-custody.json").read_text())
        if hydrated["files"]!=plan["external_inputs"]:
            raise ValueError("input hydration declarations differ")
        for relative,binding in plan["external_inputs"].items():
            path = old_data/relative
            if path.stat().st_size!=binding["bytes"] or sha(path)!=binding["sha256"]:
                raise ValueError("input CAS differs from retained predecessor archive")
        geometry = runpy.run_path(str(here/"flux_scene_spatial.py"))
        for scene in plan["scenes"]:
            key = scene["id"]
            annotation = plan["annotations"][key]
            for order,label in (("base","base"),("swap","source")):
                row = json.loads((old_data/"rows"/(key+".native."+order+".json")).read_text())
                if annotation[label+"_image_sha256"]!=row["image"]["sha256"]:
                    raise ValueError("manual reference annotation source differs")
            expected = geometry["supports"](annotation["base"],annotation["source"],parents[key+".base"].payload["img_ids"].numpy(),height=512,width=512)
            if expected!=plan["supports"][key]:
                raise ValueError("manual windows/native coordinate lowering differs")
    receipts,edges = {},{}
    for path in (data/"typed-store/receipts").glob("*.json"):
        record = json.loads(path.read_text())
        custody["public_receipt"](record)
        if record["fingerprint"]!=path.stem or record["model_identity"]!=source["model_identity"]:
            raise ValueError("public receipt custody differs")
        receipts[path.stem] = record
        if record["operation"]=="continue":
            if record["parent"] in edges and edges[record["parent"]]!=record["result"]:
                raise ValueError("native continuation ambiguous")
            edges[record["parent"]] = record["result"]

    def walk(start,stop):
        current,length = start,0
        while current!=stop:
            if current not in edges or length>plan["max_adapter_steps"]:
                raise ValueError("native continuation incomplete")
            current,length = edges[current],length+1

    if plan["stage"]=="references":
        for scene in plan["scenes"]:
            initials = []
            for history in scene["histories"]:
                prefix = source["prefixes"][history["id"]]
                if prefix["prompt"]!=history["prompt"] or prefix["seed"]!=history["seed"] or prefix["cut"]!=source["parents"][history["id"]]:
                    raise ValueError("prefix prompt/seed/cut binding differs")
                walk(prefix["initial"],prefix["cut"])
                initials.append(store.load(prefix["initial"],device="cpu"))
            for slot in ("latent","img_ids","timesteps","sigmas"):
                if describe(initials[0].payload[slot])!=describe(initials[1].payload[slot]):
                    raise ValueError("within-scene common initial noise/geometry differs")
    rows = {r["id"]:r for r in observed["rows"]}
    if set(rows)!={r["id"] for r in plan["rows"]} or len(rows)!=(6 if plan["stage"]=="references" else 16):
        raise ValueError("finite row inventory differs")
    images,acts = {},0
    for name,row in rows.items():
        if row!=json.loads((data/"rows"/(name+".json")).read_text()) or row["parent"]!=source["parents"][row["parent_name"]]:
            raise ValueError("observed row/parent differs")
        parent = parents[row["parent_name"]]
        donor = parents[row["source"]] if "source" in row else parent
        expected = dict(parent.payload)
        current = parent.fingerprint
        mode = row["mode"]
        modes = [] if mode=="native" else ["left","right"] if mode=="both" else [mode]
        if [a["support"] for a in row["acts"]]!=modes:
            raise ValueError("declared action sequence differs")
        for action,support_mode in zip(row["acts"],modes):
            before,after = (store.load(action[k],device="cpu") for k in ("parent","post"))
            indices = (list(range(parent.payload["latent"].shape[1])) if support_mode=="full" else [] if support_mode=="sham"
                else plan["supports"][row["scene"]]["masks"][support_mode])
            if before.fingerprint!=current or action["indices"]!=indices or action["slot"]!="latent" or dict(before.execution_point)!=dict(parent.execution_point):
                raise ValueError("Act source/clock/extent differs")
            value = expected["latent"].clone()
            if indices:
                value[:,indices,:] = donor.payload["latent"][:,indices,:]
            expected["latent"] = value
            if describe(after.payload)!=describe(expected):
                raise ValueError("undeclared latent row/slot changed")
            rec = action["receipt"]
            if rec!=receipts[rec["fingerprint"]] or rec["parent"]!=before.fingerprint or rec["result"]!=after.fingerprint:
                raise ValueError("Act receipt differs")
            current = after.fingerprint
            acts += 1
        if current!=row["edited_cut"]:
            raise ValueError("edited cut differs")
        for boundary in row["schedulers"]:
            walk(current,boundary["cut"])
            current = boundary["cut"]
        walk(current,row["final_cut"])
        if describe(store.load(row["final_cut"],device="cpu").payload["conditioning"])!=describe(parent.payload["conditioning"]):
            raise ValueError("recipient C changed")
        path = data/row["image"]["path"]
        if sha(path)!=row["image"]["sha256"]:
            raise ValueError("image checksum differs")
        images[name] = np.asarray(Image.open(path).convert("RGB"))
    historical = {n:rows[n]["image"]["sha256"]==h for n,h in plan.get("historical_pngs",{}).items()}
    if not all(historical.values()):
        raise ValueError("historical replay differs")
    effects,matched = {},{}
    if plan["stage"]=="spatial":
        for scene in plan["scenes"]:
            key = scene["id"]
            matched[key] = rows[key+".both"]["acts"][0]["post"]==rows[key+".left"]["edited_cut"]
            if not matched[key]:
                raise ValueError("stacked Act does not use exact left-edited parent")
            base_rgb = images[key+".native"].astype(np.float64)/255
            delta = images[key+".full"].astype(np.float64)/255-base_rgb
            coords = parents[key+".base"].payload["img_ids"].numpy()[0]
            regions = {}
            for side in ("left","right","outside"):
                mask = np.zeros((512,512),dtype=bool)
                for i in plan["supports"][key]["masks"][side]:
                    h,w = (int(v) for v in coords[i,1:3])
                    mask[16*h:16*(h+1),16*w:16*(w+1)] = True
                regions[side] = mask
            for mode in ("full","left","right","both","outside","shifted"):
                change = images[key+"."+mode].astype(np.float64)/255-base_rgb
                effects[key+"."+mode] = {"RGB_mae":float(np.abs(change).mean()),
                    "source_directed_RGB_projection":float(np.sum(change*delta)/np.sum(delta**2)) if np.sum(delta**2) else None,
                    "geometric_region_mae":{s:float(np.abs(change[m]).mean()) if m.any() else None for s,m in regions.items()}}
    columns = 3 if plan["stage"]=="references" else 8
    canvas = Image.new("RGB",(columns*256,2*282),"#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i,scene in enumerate(plan["scenes"]):
        group = [r for r in plan["rows"] if r["scene"]==scene["id"]]
        for j,row in enumerate(group):
            x,y = j*256,i*282
            canvas.paste(Image.fromarray(images[row["id"]]).resize((256,256)),(x,y+26))
            draw.text((x+5,y+8),row["id"],fill="black")
    figure = destination.parent/(plan["stage"]+".png")
    canvas.save(figure)
    report = {"schema":"mdb-flux-anime-analysis-v1","stage":plan["stage"],"job_id":summary["job_id"],
        "mechanics_status":"valid","terminal_status":"unassessed","counts":summary["counts"],"archive_custody":run["source"],
        "parents":source["parents"],"historical_controls":historical,"verified_acts":acts,"verified_receipts":len(receipts),
        "matched_left_edited_parents":matched,"effects":effects,"semantic_prediction_observations":None,
        "figure":{"path":str(figure),"sha256":sha(figure)},
        "limitations":["Visual appearance and character identity require separately saved informal anonymous observations",
            "Manual reference windows and geometric pixel differences do not establish semantic person segmentation or facial identity",
            "Two new scenes/seeds at one checkpoint/clock; dependent donor-assisted futures, no numerical RGB forecast"]}
    with destination.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"job_id":summary["job_id"],"stage":plan["stage"],"mechanics_status":"valid","counts":summary["counts"],"historical_controls":historical}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("--predecessor",type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.run_dir,args.destination,args.predecessor),sort_keys=True))
