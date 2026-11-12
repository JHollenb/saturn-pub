"""Seal a matched image/text carrier restoration panel for the MDB worker."""
import argparse
import json
import runpy
from pathlib import Path
from mdb.edit_search import EditSearch,file_digest


def prepare(root,source,index=0,prior=None):
    if index not in range(5):raise ValueError("qualified joint index required")
    data=source/"run-02/extracted/six-areas-result"
    read=lambda p:json.loads(p.read_text())
    old=read(data/"plan.json");observed=read(data/"observed.json")
    groups=None
    if prior is not None:
        previous=read(prior)
        if previous["mechanics_status"]!="verified":raise ValueError("audited prior carrier panel required")
        tokens=read(data/"tokens.json")["b0r0"];valid={i for i,v in enumerate(tokens["attention_mask"])if v}
        groups=dict(color=[4,11],nearby=[5,12],valid_other=sorted(valid-{4,11,5,12}),padding=sorted(set(range(512))-valid))
        if not {4,11,5,12}<=valid:raise ValueError("source token layout differs")
    root.mkdir(parents=True,exist_ok=False)
    binding=dict(parent=observed["b0r0"]["joint4_entry"],
        donor=observed["b1r0"]["joint4_entry"],model=old["checkpoint_sha256"],
        execution=dict(device="cuda",dtype="bfloat16",batch=1,seed=109,height=512,width=512,steps=4,
                       initial="step1 joint4 donor I pulse",restore_clock=dict(step=2,joint_exit=index),
                       source="first in-lease baseline control exit; source-assisted restoration, not a predicted operand"),
        consumer="unchanged native suffix and VAE RGB",source_inventory=file_digest(data/"files.json"))
    if prior is not None:binding["prior_carrier_evidence"]=dict(path=str(prior),sha256=file_digest(prior))
    s=EditSearch.create(root/"search",binding=binding,
        atoms=["early.C4","early.C11","retain.image"]+(["retain.text."+g for g in groups]if groups else["retain.text"]),
        observable="blue sphere left of red cube, recognizable objects; image/text retention under fixed early conditioning, native RGB review")
    cases=[("control",[],[],"red sphere left of blue cube"),
           ("positive",[4,11],[],"blue sphere left of red cube"),
           ("restore-I",[4,11],["image"],"blue sphere left of red cube"),
           ("restore-T",[4,11],["text"],"blue sphere left of red cube"),
           ("restore-IT",[4,11],["image","text"],"red sphere left of blue cube")]
    keep={}
    if groups:
        keep=dict(color=["color"],noncolor=["nearby","valid_other","padding"],nearby=["nearby"],padding=["padding"],valid=["color","nearby","valid_other"])
        cases=[("control",[],[],"red sphere left of blue cube"),("whole-T",[4,11],["image"],"blue sphere left of red cube")]
        cases.extend((name,[4,11],["image","text"],"blue sphere left of red cube"if name in{"color","valid"}else"red sphere left of blue cube")for name in keep)
    arms=[];freezes={};categorical={}
    for name,crows,restored,prediction in cases:
        atoms=["early.C"+str(r)for r in crows]
        if "image"not in restored:atoms.append("retain.image")
        if groups:atoms.extend("retain.text."+g for g in(keep[name]if name in keep else groups if"text"not in restored else[]))
        elif "text"not in restored:atoms.append("retain.text")
        freezes[name]=s.freeze(atoms,dict(categorical=prediction,
            exact_baseline_future=name=="restore-IT",
            scope=("Working row-local persistence hypotheses; no independent RGB forecast. Retain every miss."if groups else"Working single-carrier persistence hypotheses; joint restore predicts exact payload/RGB after project3 C reset. Retain every miss.")))
        arms.append(dict(id=name,slots=["image"],C="scheduled",rows_by_step={"2":crows,"3":[]},restore_slots=restored,restore_index=index))
        if name in keep:arms[-1]["text_keep_rows"]=sorted(i for g in keep[name]for i in groups[g])
        categorical[name]=prediction
    panel=dict(schema="mdb-binding-search-panel-v1",kind="carrier-restoration",search_plan=s.plan,freezes=freezes,arms=arms,
        hypotheses=dict(categorical=categorical,
            numerical="Joint I/T restoration predicts complete baseline payload except C at the restore cut, complete payload equality at project3 after base C reset, and exact baseline final payload/PNG. Compare payloads rather than lineage fingerprints.",
            alternatives="Either carrier may sustain the swap; restoring both should erase it if the declared live/external closure is complete. Single-carrier predictions are tentative; no independent pixel calculation.",
            source_assistance="Matched baseline exit is generated as the first control in this lease and used as a restoration operand. All hypotheses seal before the first target block. No claim of independently predicting source tensors."))
    if groups:
        panel.update(kind="carrier-row-restoration",text_groups=groups)
        panel["hypotheses"].update(numerical="Restore baseline image at joint0 exit; baseline text outside the explicitly retained native edited rows. Later C resets to base. No independent pixel forecast.",
            alternatives="Tentative localized address hypothesis: original color rows or valid text suffice, other rows do not. Noncolor/nearby/padding success would expose distributed transport rather than a unique semantic address. Retain misses.")
    worker=runpy.run_path(str(Path(__file__).with_name("flux_binding_minimal.py")))
    worker["validate_search"](arms)
    path=root/"panel.json"
    with path.open("x")as f:json.dump(panel,f,indent=2,sort_keys=True);f.write("\n")
    return dict(panel=str(path.resolve()),counts=worker["expected_counts"](arms))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("source",type=Path)
    p.add_argument("--index",type=int,default=0);p.add_argument("--prior",type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.source.resolve(),a.index,a.prior.resolve()if a.prior else None),sort_keys=True))
