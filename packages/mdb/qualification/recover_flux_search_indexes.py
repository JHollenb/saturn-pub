"""Recover indexes from a closed MDB observation after one inapplicable guard.

Does not modify the original failed result. Rows/counts come from MDB's durable
native observation, receipts from its CAS, and ports from inventoried arrays.
"""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.forecast_transaction import verify_transaction,load_forecast_tree
from mdb.edit_search import file_digest


def recover(run,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());data=run/"extracted/binding-minimal-result"
    runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    config=read(data/"plan.json");failure=read(data/"failure.json")
    if failure["error"]!="ValueError: same-parent historical control differs" or config["arms"][0]["id"]=="control":raise ValueError("not the narrowly qualified post-completion guard error")
    tx=verify_transaction(data/"transaction");assert tx["verified"] and tx["phase"]=="closed"
    tree=runpy.run_path(str(here/"analyze_flux_eye_program.py"))["json_tree"]
    observation=read(data/"transaction/observations/native.json")
    native=tree(load_forecast_tree(data/"transaction",observation["value"]));rows=native["rows"];counts=native["counts"]
    expected=runpy.run_path(str(run/"payload/flux_binding_minimal.py"))["expected_counts"](config["arms"])
    if counts!=expected or len(rows)!=len(config["arms"]):raise ValueError("native callback did not complete the declared panel")
    byparent={};allreceipts={}
    for p in(data/"store/receipts").glob("*.json"):
        r=read(p);allreceipts[r["fingerprint"]]=r
        if r["operation"]=="continue":byparent.setdefault(r["parent"],[]).append(r)
    recovered=[];writes=[];manifest=read(data/"files.json")
    def array(name):
        p=data/"arrays"/(name+".npy");rel=p.relative_to(data).as_posix()
        if file_digest(p)!=manifest[rel]:raise ValueError("port archive changed")
        a=np.load(p,mmap_mode="r",allow_pickle=False)
        return dict(path=rel,sha256=manifest[rel],shape=list(a.shape),dtype=str(a.dtype))
    for row in rows:
        name=row["id"];parent=row["parent"];actions=list(row["actions"]);advances=0;chain=[]
        while parent!=row["final"]:
            if actions and actions[0]["before"]==parent:
                r=actions.pop(0)["receipt"];assert r==allreceipts[r["fingerprint"]]
            else:
                possible=byparent.get(parent,[])
                if len(possible)!=1:raise ValueError("missing or ambiguous native receipt continuation")
                r=possible[0];advances+=r["steps"]
            chain.append(dict(row=name,body=r));parent=r["result"]
            if len(chain)>82:raise ValueError("unbounded receipt reconstruction")
        if actions or advances!=79 or len(chain)!=82:raise ValueError("incomplete branch closure")
        recovered.extend(chain)
        entry=row["marks"]["joint0_entry"]
        r=next(x["body"]for x in chain if x["body"]["operation"]=="continue" and x["body"]["parent"]==entry)
        writes.append(dict(row=name,point=r["transitions"][0]["before"],entry=entry,exit=row["marks"]["joint0_exit"],
            fields={k:array(name+".J00."+k)for k in("attention","attention_text","mlp_input","mlp","mlp_text_input","mlp_text","gate_attention","gate_mlp")},
            equation_exact=True))
    timeline=[json.loads(line)for line in(data/"timeline.jsonl").read_text().splitlines()]
    completed=[v for v in timeline if v["kind"]=="image_complete"]
    assert [v["row"]for v in completed]==[r["id"]for r in rows] and completed[-1]["counts"]==counts
    output.mkdir(parents=True,exist_ok=False)
    save=runpy.run_path(str(here/"flux_binding_minimal.py"))["save"]
    for name,value in(("rows.json",rows),("receipts.json",recovered),("writes.json",writes)):
        save(output/name,value)
    save(output/"summary.json",dict(counts=counts,seal=dict(seal_sha256=observation["seal_sha256"]),
        target_entries=dict(blocks=71*len(rows)),target_entries_source="receipt-derived; original final live-hook census was not persisted",
        worker_seconds=completed[-1]["seconds"],worker_seconds_scope="through last native image completion, excluding failed finalization"))
    save(output/"recovery.json",dict(schema="mdb-search-index-recovery-v1",original_job_state=read(run/"job.json")["state"],
        original_failure=failure,observation_sha256=file_digest(data/"transaction/observations/native.json"),transaction=tx,
        source_manifest_sha256=file_digest(data/"files.json"),rows_from="closed native observation",receipts_from="unique CAS continuation graph and observed Act receipts",
        ports_from="inventoried native array filenames, independently checked by subsequent auditor",new_model_calls=0,
        limitations=["Original failed job is not rewritten as succeeded","Final independent block-entry hook count unavailable; receipt-derived census retained","Per-branch sparse trace index unavailable; raw trace arrays retained"] ))
    return dict(metadata=str(output),rows=len(rows),counts=counts,new_model_calls=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(recover(a.run.resolve(),a.output.resolve()),sort_keys=True))
