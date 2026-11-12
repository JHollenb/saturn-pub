"""Build saved-only dependency plans for an audited joint-carrier restoration."""
import argparse
import json
from pathlib import Path
from mdb.dependency_reuse import plan_suffix_reuse,verify_reuse_plan
from mdb.edit_search import digest,file_digest


def plan(run,audit,output):
    read=lambda p:json.loads(p.read_text())
    evidence=read(audit)
    if evidence["mechanics_status"]!="verified":raise ValueError("audited native evidence required")
    data=run/"extracted/binding-minimal-result";rows={r["id"]:r for r in read(data/"rows.json")}
    source=rows["control"];target=rows["restore-IT"]
    receipts=[r["body"]for r in read(data/"receipts.json")if r["row"]==target["id"]]
    start=next(i for i,r in enumerate(receipts)if r["parent"]==target["marks"]["post_restore"])
    tail=receipts[start:];reset=next(i for i,r in enumerate(tail)if r["operation"]=="apply")
    if tail[reset]["act"]["writes"]!=["conditioning"]:raise ValueError("unexpected external action in suffix")
    first,second=tail[:reset],tail[reset+1:]
    assert all(r["operation"]=="continue" for r in first+second)
    execution=first[0]["execution"];assert all(r["execution"]==execution for r in first+second)
    observation=evidence["carrier_restoration"][target["id"]]
    segments=[]
    for label,chain,changed,sp,tp in[
        ("before-conditioning-reset",first,observation["post_restore_different_fields"],source["marks"]["restore_boundary"],target["marks"]["post_restore"]),
        ("after-conditioning-reset",second,observation["project3_different_fields"],source["marks"]["project3_input"],target["marks"]["project3_input"])]:
        transitions=[t["native_operation"]for r in chain for t in r["transitions"]]
        p=plan_suffix_reuse(transitions,changed_slots=changed,model_identity=chain[0]["model_identity"],
            execution_identity=digest(execution),source_parent=sp,target_parent=tp)
        segments.append(dict(label=label,plan=p,verification=verify_reuse_plan(p),native_receipt_fingerprints=[r["fingerprint"]for r in chain]))
    result=dict(schema="mdb-carrier-reuse-occurrence-v1",audit=dict(path=str(audit),sha256=file_digest(audit)),segments=segments,
        reset_receipt=tail[reset]["fingerprint"],native_reference_transitions=len(first)+len(second),
        reuse_candidates=sum(s["plan"]["reuse_candidates"]for s in segments),
        final_payload_exact=observation["final_payload_exact"],png_exact=observation["png_exact"],
        new_model_calls=0,native_work_actually_skipped=0,
        scope="Post-hoc dependency planning on real native receipts and complete audited field differences. The constant C reset is separately audited; its declared read footprint is not silently weakened. No new execution policy or measured speedup.")
    with output.open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    return {k:result[k]for k in("native_reference_transitions","reuse_candidates","final_payload_exact","png_exact","new_model_calls","native_work_actually_skipped")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in("run","audit","output"):p.add_argument(n,type=Path)
    a=p.parse_args();print(json.dumps(plan(*[getattr(a,n).resolve()for n in("run","audit","output")]),sort_keys=True))
