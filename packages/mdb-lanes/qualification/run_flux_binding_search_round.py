"""Resume one sealed MDB/mrun edit-search round through native evidence audit.

No automatic retries, semantic verdicts or cancellation. Queue timeout detaches
with the job identity. Partial collections are retained and refuse overwrite.
"""
import argparse
import json
import runpy
import time
from pathlib import Path


def run_round(run,source,panel,collection,audit,*,vram_mb=11500,wait_seconds=1800,metadata_recovery=None):
    from mdb_lanes.job_support import scheduler_api
    from mdb_lanes.edit_search import file_digest
    here=Path(__file__).resolve().parent
    worker=runpy.run_path(str(here/"flux_binding_minimal.py"))
    recovery=runpy.run_path(str(here/"recover_flux_directory.py"))
    auditor=runpy.run_path(str(here/"analyze_flux_binding_search.py"))
    custody=runpy.run_path(str(here/"flux_family_custody.py"))
    read=lambda p:json.loads(p.read_text())
    def event(phase,**fields):
        value=dict(phase=phase,**fields)
        print(json.dumps(value,sort_keys=True),flush=True)
        with(run/"controller-events.jsonl").open("a")as f:f.write(json.dumps(value,sort_keys=True)+"\n")
    if not run.exists():worker["stage"](run,source,"search","I",panel,vram_mb)
    request=read(run/"request.json")
    if request["config"]["panel"]!="search" or request["config"]["equation_forecasts_sha256"]!=file_digest(panel):raise ValueError("existing run is bound to another panel")
    custody["_request"](run)
    if not(run/"submission.json").exists():worker["submit"](run)
    job_id=read(run/"submission.json")["job_id"];event("admitted",job_id=job_id,request_sha256=request["request_sha256"])
    api=scheduler_api();started=time.monotonic();previous=None
    while True:
        status,raw,_=api.request("GET","/api/jobs/"+job_id,timeout_s=30)
        if status>=400:raise ValueError("scheduler read failed; resume with the same run")
        job=json.loads(raw)
        if job["state"]!=previous:event("scheduler",job_id=job_id,state=job["state"]);previous=job["state"]
        if job["state"]in custody["TERMINAL"]:break
        if time.monotonic()-started>=wait_seconds:
            return dict(status="detached",job_id=job_id,run=str(run),reason="wait budget reached; existing job retained")
        time.sleep(5)
    metadata=None
    if job["state"]!="succeeded":
        if not(run/"report.json").exists():custody["collect"](run,job_id)
        report=read(run/"report.json")["report"]if(run/"report.json").exists()else{}
        if job["state"]!="failed" or report.get("error")!="ValueError: same-parent historical control differs" or worker["has_historical_control"](request["config"]["arms"]):
            raise ValueError("non-successful native attempt retained: "+job_id+" "+job["state"])
        metadata=metadata_recovery or audit.parent/(audit.stem+"-index-recovery")
        if not metadata.exists():runpy.run_path(str(here/"recover_flux_search_indexes.py"))["recover"](run,metadata)
        collection=run;custody["verify"](collection)
        event("recovered-inapplicable-post-completion-check",job_id=job_id,metadata=str(metadata),original_state=job["state"])
    elif not collection.exists():recovery["recover"](run,collection)
    else:custody["verify"](collection)
    event("collected",job_id=job_id,collection=str(collection))
    if not audit.exists():auditor["analyze"](collection,source,audit,metadata)
    result=read(audit)
    if result["mechanics_status"]!="verified" or result["custody"]["archive_sha256"]!=read(collection/"archive-custody.json")["archive_sha256"]:raise ValueError("audit is bound to different evidence")
    event("audited",job_id=job_id,audit=str(audit))
    return dict(status="verified-awaiting-consumer-assessment",job_id=job_id,run=str(run),collection=str(collection),audit=str(audit),
        original_job_state=job["state"],metadata_recovery=None if metadata is None else str(metadata),
        image_sheet=str(audit.with_suffix(".png")),counts=result["counts"],colors={k:v["observed_colors"]for k,v in result["outcomes"].items()},
        projection_forecasts=result["projection_predictions"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in("run","source","panel","collection","audit"):p.add_argument(n,type=Path)
    p.add_argument("--vram-mb",type=int,default=11500);p.add_argument("--wait-seconds",type=int,default=1800)
    p.add_argument("--metadata-recovery",type=Path)
    a=p.parse_args()
    print(json.dumps(run_round(*[getattr(a,n).resolve()for n in("run","source","panel","collection","audit")],vram_mb=a.vram_mb,wait_seconds=a.wait_seconds,metadata_recovery=a.metadata_recovery.resolve()if a.metadata_recovery else None),sort_keys=True))
