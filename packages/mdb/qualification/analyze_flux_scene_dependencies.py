"""Audit saved L/I/T restorations and complete conditional Euler futures."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(a,b):
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    return {"mae":float(np.abs(d).mean()),"l2":float(np.sqrt(np.sum(d*d))),"max_abs":float(np.abs(d).max()),"exact_values":int(np.count_nonzero(d==0)),"values":int(d.size)}


def analyze(run,destination):
    from mdb.forecast_transaction import verify_transaction
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/"extracted/scene-dependencies-result"
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    summary,config,rows,forecasts,receipts=[read(data/n) for n in ("summary.json","plan.json","rows.json","latent-forecasts.json","receipts.json")]
    request=read(run/"request.json")
    assert summary==read(run/"report.json")["report"] and config==request["config"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    for n,h in request["source_hashes"].items():assert sha(run/"payload"/n)==h
    verifications=[verify_transaction(data/n,expected_seal_sha256=summary[k]["seal_sha256"])
        for n,k in (("transaction","semantic_seal"),("latent-transaction","latent_seal"))]
    assert all(v["verified"] and v["phase"]=="closed" for v in verifications)
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"))
    routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    store=typed["audit_store"](data/"store")
    def slot(cut,key):return routes["slot_array"](data/"store",cut,key,typed)
    def fields(cut):return {i["key"]["value"]:i["value"] for i in store["cuts"][cut]["payload"]["items"]}
    def arr(r):return load_tensor_archive(data,r)
    for r in receipts:public(r)
    assert len(receipts)==183 and sum(r["operation"]=="continue" for r in receipts)==164
    assert sum(r["operation"]=="apply" for r in receipts)==19
    by_receipt={r["fingerprint"]:r for r in receipts}
    images,latents={},{}
    for row in rows:
        assert sha(data/row["png"])==row["png_sha256"]
        with Image.open(data/row["png"]) as im:
            assert im.mode=="RGB" and im.size==(512,512)
            images[row["id"]]=np.asarray(im).copy()
        latents[row["id"]]=arr(row["latent"])
        assert np.array_equal(latents[row["id"]],slot(row["final_cut"],"latent"))
        previous=None
        for r in row["path"]:
            assert r==by_receipt[r["fingerprint"]]
            if previous:assert r["parent"]==previous
            previous=r["result"]
        if previous:assert previous==row["final_cut"]
        for act in row["actions"]:
            assert act["receipt"]==by_receipt[act["receipt"]["fingerprint"]]
            before,after=fields(act["parent"]),fields(act["post_act"])
            assert before.keys()==after.keys() and act["act"]["writes"]==[act["slot"]]
            assert all(before[k]==after[k] for k in before if k!=act["slot"])
    assert len(images)==12
    controls={role:sha(data/("native."+role+".png"))==config["historical_pngs"][role] for role in ("base","joint")}
    controls["old_M_transfer"]=sha(data/"L0I0T0.png")==config["historical_pngs"]["old_M_transfer"]
    assert all(controls.values())
    curves=read(data/"curves.json")
    raw_by_id={r["id"]:r for r in rows}
    equation_errors,validation={},{}
    for name,spec in forecasts.items():
        latent,velocity=arr(spec["initial_latent"]),arr(spec["velocity"])
        q=typed["quantize_bf16"]
        rebuilt=q((latent+q(q(np.asarray(spec["sigma_delta"],np.float32))*velocity)).astype(np.float32))
        prediction=arr(spec["latent"])
        equation_errors[name]=metric(prediction,rebuilt)
        assert np.array_equal(prediction,rebuilt)
        predicted_row=raw_by_id[name+".predicted"]
        assert predicted_row["kind"]=="forecast-latent-native-VAE" and not predicted_row["path"]
        if name+".validation" in raw_by_id:
            observed=raw_by_id[name+".validation"]
            comparison=metric(latents[name+".validation"],prediction)
            low=raw_by_id[spec["velocity_source"]]
            validation[name]={"latent_error":comparison,"velocity_exact":np.array_equal(arr(observed["noise"]),arr(low["noise"])),
                "png_exact":np.array_equal(images[name+".validation"],images[name+".predicted"]),
                "paired_descriptor_curve_exact":curves[name+".validation"]==curves[spec["velocity_source"]]}
            assert observed["forecast_exact"]==(comparison["exact_values"]==comparison["values"])
            assert observed["png_exact"]==validation[name]["png_exact"]
    comparisons={name:{"to_base":metric(im,images["native.base"]),"to_joint":metric(im,images["native.joint"])}
        for name,im in images.items()}
    report={"schema":"mdb-scene-dependencies-audit-v1","job_id":summary["job_id"],"request_sha256":request["request_sha256"],
        "mechanics_status":"verified","instrument_status":"applicable","trend_status":"preserve-per-case","terminal_status":"unassessed",
        "counts":summary["counts"],"custody":custody,"store_counts":store["counts"],"transactions":verifications,
        "historical_controls":controls,"equation_errors":equation_errors,"native_validations":validation,
        "rgb_comparisons":comparisons,"worker_seconds":summary["worker_seconds"],
        "coverage":"Four complete Euler references independently reconstructed. Native velocity/final-latent/PNG comparisons cover00/11 corners. Middle01/10 are forecast-state/nativeVAE only. Intermediate paired curves have worker-emitted full tensor hashes, without retained complete intermediate arrays. Derived project frontiers are reconstitutions with native12-step payload checks, not fabricated historical cuts."}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return {"audit":str(destination),"native_validations":validation,"rgb_comparisons":comparisons}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve()),sort_keys=True))
