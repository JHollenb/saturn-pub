"""Curate source-qualified state-response candidates and retained prediction misses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root):
    run = root/"closure-run-02"
    data = run/"extracted/flux-scene-state-closure-result"
    audit = json.loads((run/"analysis-v1.json").read_text())
    visual = json.loads((root/"closure-review/visual-review.json").read_text())
    plan,source,summary = (json.loads((data/name).read_text()) for name in ("plan.json","source.json","summary.json"))
    rows = {spec["id"]:json.loads((data/"rows"/(spec["id"]+".json")).read_text()) for spec in plan["rows"]}
    records,readouts = [],{}
    for name,inventory in audit["inventories"].items():
        others = [c for c in inventory["components"] if c["color"]!="red"]
        readouts[name] = {"components":inventory["components"],"red_anchor":inventory["red_anchor"],
            "other_green_minus_blue":others[0]["median_rgb"][1]-others[0]["median_rgb"][2] if len(others)==1 else None,
            "visual":visual["rows"][name]}
    binding = {"model_identity":source["model_identity"],"checkpoint_sha256":source["checkpoint_sha256"],
        "execution":source["execution"],"job_id":summary["job_id"],"seed":plan["seed"],
        "audit_sha256":sha(run/"analysis-v1.json"),"source_sha256":sha(data/"source.json"),
        "visual_review_sha256":sha(root/"closure-review/visual-review.json"),"seal":summary["seal"],
        "terminal_status":"unassessed","replication":"one checkpoint/seed, three directed source contrasts, dependent state coalitions"}
    for factor_number,factor in enumerate(("shapeA","colorB","relation")):
        for bits in ("001","010","011","100","101","110","111"):
            name = factor+"."+bits
            records.append({"id":f"flux.scene.state-response.factor_{factor_number}.{bits}","binding":binding,
                "supplied_factor_alias":factor,"semantic_operator":None,
                "source_cut":plan["input_cuts"][rows[name]["source"]],"recipient_cut":rows[name]["parent"],
                "source_policy":{"image_I":int(bits[0]),"working_text_T":int(bits[1]),"future_conditioning_C":int(bits[2]),
                                 "I_T_clock":"step1 joint4 exit","C_clock":"step2 preprojection"},
                "retained_state":"Recipient scheduler latent/history and all nonwritten slots; actual matched pre-C parent verified",
                "actual_acts":rows[name]["acts"],"native_continuation":rows[name]["schedulers"],
                "effects":audit["effects"][name],"pixel_readout":readouts[name],
                "evidence_status":"causal-intervention observation; exact-context response-family candidate",
                "applicability":"Only named source/recipient histories, clocks, model and numerical profile; no held-out expansion"})
    semantic_misses = {"colorB.future-C-only":{"status":"counterexample","pixel":audit["prediction_observations"]["colorB.future-C-only"],
        "visual":visual["rows"]["colorB.001"],"trend":"partial cyan shift, not absence of color influence"},
        "relation.future-C-only":{"status":"counterexample","pixel":audit["prediction_observations"]["relation.future-C-only"],"visual":visual["rows"]["relation.001"]},
        "shapeA.future-C-only":{"status":"counterexample under informal visual observation","visual":visual["rows"]["shapeA.001"]}}
    value = {"schema":"mdb-flux-scene-state-response-records-v1","binding":binding,"records":records,
        "readouts":readouts,"frozen_numerical_predictions":audit["prediction_observations"],"semantic_prediction_misses":semantic_misses,
        "working_inferences":["Color response jointly depends on current text and later conditioning in this panel; intermediate cyan outcomes retain a continuous trend",
            "Shape responds more strongly to coherent I/T/C than image alone, with an imperfect near-sphere endpoint",
            "All seven late ordering coalitions retain recipient ordering; carried latent/history is a live untested explanation, alongside state/address mismatch"],
        "automatic_semantic_decomposition":False,"automatic_equation_atlas_ingestion":False}
    with (root/"state-response-records.json").open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return {"records":len(records),"semantic_prediction_misses":semantic_misses,
            "green_minus_blue":{name:readouts[name]["other_green_minus_blue"] for name in readouts if name.startswith("colorB.")}}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    print(json.dumps(publish(p.parse_args().root),sort_keys=True))
