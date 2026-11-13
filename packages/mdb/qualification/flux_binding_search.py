"""Freeze bounded row x clock edits using MDB's durable search ledger."""
import argparse
import json
import runpy
from pathlib import Path
from mdb.edit_search import EditSearch, file_digest

ATOMS=["row4.step2","row11.step2","row4.step3","row11.step3"]


def prepare(root, source):
    data=source/"run-02/extracted/six-areas-result"
    read=lambda p:json.loads(p.read_text())
    old=read(data/"plan.json");observed=read(data/"observed.json")
    s=EditSearch.create(root/"search",binding=dict(parent=observed["b0r0"]["joint4_entry"],
        donor=observed["b1r0"]["joint4_entry"],model=old["checkpoint_sha256"],execution=dict(device="cuda",dtype="bfloat16",batch=1,seed=109,height=512,width=512,steps=4,initial="step1 joint4 donor I pulse"),
        consumer="unchanged native suffix and VAE RGB",source_inventory=file_digest(data/"files.json")),
        atoms=ATOMS,observable="blue sphere left of red cube, with both objects still recognizable; chromatic diagnostics and manual native-image review; fine shape/shading collateral remains separate")
    cases=[("control",[],"red sphere left of blue cube"),
           ("positive",ATOMS,"blue sphere left of red cube"),
           ("row4",[a for a in ATOMS if a.startswith("row4.")],"blue sphere left of blue cube"),
           ("row11",[a for a in ATOMS if a.startswith("row11.")],"red sphere left of red cube"),
           ("step2",[a for a in ATOMS if a.endswith("step2")],"blue sphere left of red cube"),
           ("step3",[a for a in ATOMS if a.endswith("step3")],"red sphere left of blue cube")]
    return freeze_panel(root,s,cases,"round1")


def freeze_panel(root, search, cases, name, projection_source=None):
    arms=[];categorical={};freezes={}
    for label,atoms,prediction in cases:
        frozen=search.freeze(atoms,dict(categorical=prediction,scope="prospective working hypothesis, not an RGB pixel prediction; retain misses"))
        freezes[label]=frozen
        arms.append(dict(id=label,slots=["image"],C="scheduled",rows_by_step={str(step):sorted(int(a.split(".")[0][3:]) for a in atoms if a.endswith("step"+str(step)))for step in (2,3)}))
        categorical[label]=prediction
    panel=dict(schema="mdb-binding-search-panel-v1",search_plan=search.plan,freezes=freezes,arms=arms,
        hypotheses=dict(categorical=categorical,numerical="Project2 image input depends on unchanged I-primed Z2; C row replacement is exact and restricted to declared rows at each project. No independent full-suffix or pixel prediction.",
                        alternatives="one row versus contextual coalition; early-only versus late-only versus both reads; retain mixed colors and counterexamples"))
    if projection_source is not None:
        import hashlib
        import numpy as np
        here=Path(__file__).resolve().parent
        readslot=runpy.run_path(str(here/"analyze_flux_binding_minimal.py"))["read_slot"]
        source=projection_source/"subsets-01/extracted/binding-minimal-result"
        observed={r["id"]:r for r in json.loads((source/"rows.json").read_text())}
        if json.loads((projection_source/"subsets-analysis-v1.json").read_text())["mechanics_status"]!="verified":raise ValueError("audited projection operands required")
        base=observed["I"]["marks"]["joint0_entry"];edited=observed["IC.color"]["marks"]["joint0_entry"]
        I=readslot(source/"store",base,"image");T=readslot(source/"store",base,"text");Td=readslot(source/"store",edited,"text")
        assert np.array_equal(I,readslot(source/"store",edited,"image"))
        fields={};directory=root/(name+"-projection-forecasts");directory.mkdir()
        def publish(value):
            key=hashlib.sha256(value.tobytes()).hexdigest();p=directory/(key+".npy")
            if not p.exists():np.save(p,value,allow_pickle=False)
            return dict(path=str(p.resolve()),sha256=file_digest(p),shape=list(value.shape),dtype=str(value.dtype))
        for arm in arms:
            text=T.copy();ix=arm["rows_by_step"]["2"];text[:,ix]=Td[:,ix]
            fields[arm["id"]]=dict(image=publish(I),text=publish(text))
        panel["projection_forecasts"]=dict(fields=fields,source=str(source.resolve()),source_rows_sha256=file_digest(source/"rows.json"),
            base_cut=base,edited_cut=edited,model_calls=0,
            scope="Source-assisted cached projected row operands plus row-local projection law; complete I/T prediction at step2 joint0 entry. No independent weight evaluation, later attention, full suffix or RGB forecast.")
    path=root/(name+"-panel.json")
    with path.open("x")as f:json.dump(panel,f,indent=2,sort_keys=True);f.write("\n")
    return dict(panel=str(path),candidates=len(arms),counts=runpy.run_path(str(Path(__file__).with_name("flux_binding_minimal.py")))["expected_counts"](arms))


def record_round(root, run, audit, review, next_round=None, projection_source=None, metadata=None):
    """Consumer review is explicit data; a distance is never semantic authority."""
    read=lambda p:json.loads(p.read_text())
    analysis=read(audit);inspection=read(review);data=run/"extracted/binding-minimal-result"
    panel=read(data/"plan.json")["search"];search=EditSearch(root/"search")
    if analysis["mechanics_status"]!="verified" or panel["search_plan"]!=search.plan:raise ValueError("unverified or different search")
    if metadata is not None and analysis.get("post_completion_index_recovery",{}).get("metadata")!=str(metadata):raise ValueError("audit does not qualify recovered metadata")
    rows={r["id"]:r for r in read((data if metadata is None else metadata)/"rows.json")}
    if set(rows)!=set(inspection["rows"]):raise ValueError("complete consumer review required")
    existing={r["candidate"]:r for r in search.status()["records"]}
    # Validate the entire assessment before publishing any outcomes. Re-running
    # an interrupted collection is idempotent only for exactly equal evidence.
    pending=[]
    for name,row in rows.items():
        observed=inspection["rows"][name]
        if observed["png_sha256"]!=row["png_sha256"] or file_digest(data/row["png"])!=row["png_sha256"]:raise ValueError("review image differs")
        candidate=panel["freezes"][name]["candidate"]
        measurements=dict(native_image_sha256=row["png_sha256"],observation=observed["observation"],diagnostics=analysis["outcomes"][name])
        refs=[audit,review,run/"archive-custody.json"]
        if observed["verdict"]not in {"sufficient","insufficient","uncertain"}:raise ValueError("invalid assessment")
        if existing[candidate]["verdict"]!="pending":
            previous=read(search.root/"candidates"/candidate/"outcome.json")
            if previous["verdict"]!=observed["verdict"] or previous["measurements"]!=measurements or previous["evidence"]!=[dict(path=str(p.resolve()),sha256=file_digest(p))for p in refs]:raise ValueError("changed assessment; retain immutable outcomes")
        else:pending.append((candidate,refs,observed["verdict"],measurements))
    for candidate,refs,verdict,measurements in pending:
        search.observe(candidate,evidence=refs,verdict=verdict,measurements=measurements)
    result=dict(status=search.status())
    if next_round:
        if (root/(next_round+"-panel.json")).exists():
            saved=read(root/(next_round+"-panel.json"))
            if saved["search_plan"]!=search.plan or any(read(search.root/"candidates"/v["candidate"]/"freeze.json")!=v for v in saved["freezes"].values()):raise ValueError("saved next panel differs")
            result["next"]=dict(panel=str(root/(next_round+"-panel.json")),candidates=len(saved["arms"]),resumed=True)
            return result
        proposed=search.propose(6);cases=[]
        for i,item in enumerate(proposed):
            atoms=item["atoms"];rows_used={int(a.split(".")[0][3:]) for a in atoms}
            predicted=("blue sphere left of red cube" if rows_used=={4,11} else "blue sphere left of blue cube" if rows_used=={4} else "red sphere left of red cube" if rows_used=={11} else "red sphere left of blue cube")
            cases.append((f"candidate{i+1}",atoms,predicted))
        if cases:result["next"]=freeze_panel(root,search,cases,next_round,projection_source)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("--source",type=Path)
    p.add_argument("--record",action="store_true");p.add_argument("--run",type=Path);p.add_argument("--audit",type=Path);p.add_argument("--review",type=Path);p.add_argument("--next")
    p.add_argument("--projection-source",type=Path)
    p.add_argument("--metadata",type=Path)
    a=p.parse_args()
    result=(record_round(a.root.resolve(),a.run.resolve(),a.audit.resolve(),a.review.resolve(),a.next,a.projection_source.resolve() if a.projection_source else None,a.metadata.resolve()if a.metadata else None) if a.record else prepare(a.root.resolve(),a.source.resolve()))
    print(json.dumps(result,sort_keys=True))
