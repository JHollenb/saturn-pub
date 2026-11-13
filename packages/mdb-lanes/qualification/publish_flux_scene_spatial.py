"""Curate spatial response occurrences and componentwise prediction observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root):
    run = root/"spatial-run-01"
    data = run/"extracted/flux-scene-spatial-result"
    audit = json.loads((run/"analysis-v1.json").read_text())
    visual = json.loads((root/"spatial-review/visual-review.json").read_text())
    plan,source,summary = (json.loads((data/n).read_text()) for n in ("plan.json","source.json","summary.json"))
    binding = {"model_identity":source["model_identity"],"checkpoint_sha256":source["checkpoint_sha256"],
        "execution":source["execution"],"job_id":summary["job_id"],"seed":plan["seed"],
        "audit_sha256":sha(run/"analysis-v1.json"),"source_sha256":sha(data/"source.json"),
        "visual_review_sha256":sha(root/"spatial-review/visual-review.json"),"seal":summary["seal"],
        "terminal_status":"unassessed","replication":"One checkpoint/seed;15new dependent spatial futures,5historical controls"}
    records = []
    for i,factor in enumerate(("shapeA","colorB","relation")):
        for mode in ("full","left","right","both","outside","shifted"):
            name = factor+"."+mode
            row = json.loads((data/"rows"/(name+".json")).read_text())
            records.append({"id":f"flux.scene.spatial-response.factor_{i}.{mode}","binding":binding,
                "supplied_factor_alias":factor,"semantic_operator":None,
                "source_cut":plan["input_cuts"][row["source"]],"recipient_cut":row["parent"],
                "clock":"step2 project before native input projection","source_policy":"Source latent rows at selected native H/W addresses; recipient C and all other state retained",
                "support":{k:v for k,v in plan["supports"][factor].items() if k!="masks"},
                "actual_indices":plan["supports"][factor]["masks"][mode],
                "novelty":"historical replay control" if mode=="full" else "prospective new spatial future",
                "actual_acts":row["acts"],"native_continuation":row["schedulers"],"effects":audit["effects"][name],
                "pixel_inventory":audit["inventories"][name],"visual":visual["rows"][name],
                "applicability":"Named source/recipient cuts, output-assisted windows, clock, checkpoint and numerical profile only"})
    predictions = dict(audit["prediction_observations"])
    v = visual["rows"]
    predictions["H-shape"] = {"left_sphere_like":v["shapeA.left"]["left_sphere_like"],
        "both_sphere_like":v["shapeA.both"]["left_sphere_like"], "right_retains_cube":v["shapeA.right"]["left_cube_like"],
        "instrument":"Informal arm-blinded assistant shape observations; mixed shapes preserved in visual records"}
    predictions["H-order-both"]["shape_binding_observed"] = {"left_sphere_like":v["relation.both"]["left_sphere_like"],
        "right_cube_like":v["relation.both"]["right_cube_like"]}
    value = {"schema":"mdb-flux-scene-spatial-response-records-v1","binding":binding,"records":records,
        "predictions":predictions,"output_interactions":audit["output_interactions"],
        "mechanics_status":"valid","terminal_status":"unassessed","automatic_semantic_decomposition":False,
        "automatic_equation_atlas_ingestion":False,
        "unknowns":["Minimal latent semantic support","Absolute positions versus a relational predicate",
            "Binding rule outside these histories","New seed/context applicability","Independent numerical RGB prediction"]}
    with (root/"spatial-response-records.json").open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"records":len(records),"predictions":predictions}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    print(json.dumps(publish(p.parse_args().root),sort_keys=True))
