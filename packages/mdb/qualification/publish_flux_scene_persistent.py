"""Curate source-qualified carried-state response records, retaining misses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root):
    run = root/"persistent-run-01"
    data = run/"extracted/flux-scene-persistent-result"
    audit = json.loads((run/"analysis-v1.json").read_text())
    visual = json.loads((root/"persistent-review/visual-review.json").read_text())
    plan,source,summary = (json.loads((data/n).read_text()) for n in ("plan.json","source.json","summary.json"))
    binding = {"model_identity":source["model_identity"],"checkpoint_sha256":source["checkpoint_sha256"],
        "execution":source["execution"],"job_id":summary["job_id"],"seed":plan["seed"],
        "audit_sha256":sha(run/"analysis-v1.json"),"source_sha256":sha(data/"source.json"),
        "visual_review_sha256":sha(root/"persistent-review/visual-review.json"),"seal":summary["seal"],
        "terminal_status":"unassessed","replication":"One checkpoint/seed; three new dependent L-source/C-base crosses, seven historical controls"}
    records = []
    for i,factor in enumerate(("shapeA","colorB","relation")):
        for bits in ("01","10","11"):
            name = factor+"."+bits
            row = json.loads((data/"rows"/(name+".json")).read_text())
            records.append({"id":f"flux.scene.carried-response.factor_{i}.{bits}","binding":binding,
                "supplied_factor_alias":factor,"semantic_operator":None,
                "source_cut":plan["input_cuts"][row["source"]],"recipient_cut":row["parent"],
                "clock":"step2 project before native input projection",
                "policy":{"source_latent_L":int(bits[0]),"source_conditioning_C":int(bits[1]),
                    "nuisance":"All other slots exact; I/T/context/noise_prediction cleared; source whole latent, not object-local patch"},
                "novelty":"prospective new crossed future" if bits=="10" else "exact historical replay control",
                "actual_acts":row["acts"],"native_continuation":row["schedulers"],"effects":audit["effects"][name],
                "pixel_inventory":audit["inventories"][name],"visual":visual["rows"][name],
                "applicability":"Only named source/recipient cuts, clock, checkpoint, seed and numerical profile; untested elsewhere"})
    hypotheses = {**audit["prediction_observations"],"H-shape":{"observed":visual["rows"]["shapeA.10"]["red_sphere_like"],
        "instrument":"informal arm-blinded assistant inspection; no calibrated shape classifier"}}
    value = {"schema":"mdb-flux-scene-carried-response-records-v1","binding":binding,"records":records,
        "predictions":hypotheses,"mechanics_status":"valid","terminal_status":"unassessed",
        "automatic_semantic_decomposition":False,"automatic_equation_atlas_ingestion":False,
        "unknowns":["Internal object identity/address binding","Minimal relation-specific carried subspace",
                    "New-seed/context applicability","Independent numerical RGB prediction"]}
    with (root/"carried-response-records.json").open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"records":len(records),"predictions":hypotheses}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    print(json.dumps(publish(p.parse_args().root),sort_keys=True))
