"""Close the two audited carrier panels without further model execution."""
import argparse
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from mdb.computation_atlas import ComputationAtlas
from mdb.edit_search import EditSearch,file_digest,digest


def verify(root):
    here=Path(__file__).resolve().parent
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"]
    read=lambda p:json.loads(p.read_text())
    total={};panels=[];unique_entries=set();references=[]
    for panel in(root,root/"rows"):
        run=panel/"collection-01";data=run/"extracted/binding-minimal-result"
        checked=custody(run);a=read(panel/"audit-v1.json");summary=read(data/"summary.json");job=read(run/"job.json")
        assert a["mechanics_status"]=="verified" and a["custody"]["archive_sha256"]==checked["archive_sha256"]
        assert job["state"]=="succeeded" and a["counts"]==summary["counts"]
        review=read(panel/"image-review-v1.json");rows=read(data/"rows.json")
        for row in rows:
            assert review["rows"][row["id"]]["png_sha256"]==row["png_sha256"]==file_digest(data/row["png"])
        status=EditSearch(panel/"search").status();assert len(status["records"])==len(rows)and all(r["verdict"]!="pending"for r in status["records"])
        for w in read(data/"writes.json"):
            unique_entries.add(digest(read(data/"store/cuts"/(w["entry"]+".json"))["slots"]))
        for k,v in a["counts"].items():total[k]=total.get(k,0)+v
        panels.append(dict(job_id=job["job_id"],custody=checked,counts=a["counts"],block_entries=a["target_entries"]["blocks"],
            categorical_matched=sum(v["color_prediction_matched"]for v in a["outcomes"].values()),categorical_total=len(rows),
            runner_seconds=job["result"]["elapsed_s"],worker_seconds=a["worker_seconds"],
            scheduler_log_bytes=(run/"scheduler-metadata.log").stat().st_size,collection=read(run/"collection-timing.json")))
        references.extend(panel/n for n in("audit-v1.json","image-review-v1.json","panel.json"))
    atlas=ComputationAtlas(root/"public-records-v1/atlas");records=atlas.find();publication=read(root/"public-records-v1/provenance.json")
    assert len(records)==publication["equation_occurrences"]==12
    for record in records:
        context=record["equation"]["tested_context"]
        assert len(atlas.find(applicable_to=context))==1
        assert not atlas.find(applicable_to={**context,"clock":dict(step=2,joint=1)})
        assert not atlas.find(applicable_to={**context,"receiving_state":"unknown-parent"})
        for ref in record["provenance"]["artifacts"]:assert file_digest(ref["path"])==ref["sha256"]
    storage=read(root/"storage-verification-v2.json");assert storage["mechanics_status"]=="verified"
    tests=ET.parse(root/"installed-tests-v1.xml").getroot();suites=tests.findall("testsuite")
    assert sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0))for s in suites)==0
    first=read(root/"audit-v1.json");restoration=first["carrier_restoration"]["restore-IT"]
    result=dict(schema="mdb-carrier-study-verification-v1",mechanics_status="verified",instrument_status="hue/position diagnostics plus explicit native-image review; fine details not certified",terminal_status="unassessed",
        panels=panels,counts=total,block_entries=sum(p["block_entries"]for p in panels),categorical_matched=sum(p["categorical_matched"]for p in panels),categorical_total=sum(p["categorical_total"]for p in panels),
        prediction_scope="One fixed checkpoint/receiver/seed, two adaptive panels; includes four repeat/historical controls. No new prompts, independent whole-image numerical prediction or general semantic rule claim.",
        joint_restoration=dict(final_payload_exact=restoration["final_payload_exact"],png_exact=restoration["png_exact"],post_restore_different_fields=restoration["post_restore_different_fields"],project3_different_fields=restoration["project3_different_fields"]),
        atlas=dict(occurrences=len(records),unique_J00_numeric_payloads=len(unique_entries),exact_queries=12,changed_context_abstentions=24,
            equation_scope="Post-hoc conditional BF16 writer closure from observed native ports. Repeated numeric states and later restored futures are retained separately, not new discovered laws."),
        reuse=dict(native_suffix_candidates=read(root/"reuse-plan-v1.json")["reuse_candidates"],native_work_actually_skipped=0),
        storage=storage,duplicate_inode_payload_bytes_released=sum(read(root/("storage-compaction-v"+str(i)+".json"))["released_inode_logical_bytes"]for i in(1,2)),
        installed_tests=sum(int(s.attrib["tests"])for s in suites),new_model_calls_in_verification=0,
        evidence=[dict(path=str(p),sha256=file_digest(p))for p in references+[root/"reuse-plan-v1.json",root/"public-records-v1/provenance.json",root/"storage-verification-v2.json",root/"installed-tests-v1.xml"]],
        qualification_sources={n:file_digest(here/n)for n in("flux_binding_minimal.py","flux_carrier_restoration.py","analyze_flux_binding_search.py","plan_flux_carrier_reuse.py","publish_flux_carrier_atlas.py","verify_flux_carrier_study.py")})
    with(root/"verification-v1.json").open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    return {k:result[k]for k in("mechanics_status","counts","categorical_matched","categorical_total","installed_tests","duplicate_inode_payload_bytes_released","new_model_calls_in_verification")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);a=p.parse_args();print(json.dumps(verify(a.root.resolve()),sort_keys=True))
