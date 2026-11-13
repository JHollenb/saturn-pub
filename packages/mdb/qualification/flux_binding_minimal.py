"""Receiver sufficiency, lexical conditioning and native first-writer capture."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def sha(p):
    d=hashlib.sha256()
    with Path(p).open("rb") as s:
        for b in iter(lambda:s.read(8<<20),b""):d.update(b)
    return d.hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def bind_environment(environment):
    """Reuse the recorded remote lane, never infer worker paths from the laptop."""
    from mdb.execution_environment import intent
    for key,name in dict(uv_cache_dir="MDB_WORKER_UV_CACHE_DIR",torch_home="MDB_WORKER_TORCH_HOME",
                         extra_model_roots="MDB_WORKER_MODEL_ROOTS",python="MDB_WORKER_PYTHON",
                         uv_link_mode="MDB_WORKER_UV_LINK_MODE").items():
        if name in os.environ and os.environ[name]!=environment[key]:raise ValueError("explicit worker environment differs from source lane: "+name)
        os.environ[name]=environment[key]
    if intent()!=environment:raise ValueError("installed execution requirements differ from recorded lane")
def specs(panel="streams", prime="I"):
    slots={"I":["image"],"T":["text"],"IT":["image","text"]}[prime]
    if panel=="streams":
        return [dict(id="sham",slots=["image","text"],C="base"),
            dict(id="I",slots=["image"],C="base"),
            dict(id="IC",slots=["image"],C="all"),
            dict(id="TC",slots=["text"],C="all"),
            dict(id="ITC",slots=["image","text"],C="all")]
    if panel=="subsets":
        return [dict(id=prime,slots=slots,C="base"),
            *[dict(id=prime+"C."+group,slots=slots,C=group) for group in ("all","valid","padding","color")]]
    raise ValueError("unknown finite panel")


def validate_search(arms):
    if not 1<=len(arms)<=8 or len({a["id"] for a in arms})!=len(arms):raise ValueError("finite unique search arms required")
    for a in arms:
        if a["slots"]!=["image"] or a["C"]!="scheduled" or set(a["rows_by_step"])!={"2","3"}:raise ValueError("unqualified search boundary")
        if not all(isinstance(v,list) and v==sorted(set(v)) and set(v)<={4,11} for v in a["rows_by_step"].values()):raise ValueError("unqualified search rows")
        if not a["id"].replace(".","").replace("-","").isalnum():raise ValueError("unsafe search name")
    if any("restore_slots" in a for a in arms):
        if not has_historical_control(arms) or arms[0]["id"]!="control":raise ValueError("matched first control required for carrier restoration")
        indices={a.get("restore_index") for a in arms}
        if len(indices)!=1 or next(iter(indices)) not in range(5):raise ValueError("one qualified joint restoration boundary required")
        for a in arms:
            slots=a.get("restore_slots")
            if not isinstance(slots,list) or slots!=[s for s in ("image","text") if s in slots]:raise ValueError("qualified restoration slots required")
            if "text_keep_rows" in a:
                ix=a["text_keep_rows"]
                if "text" not in slots or not isinstance(ix,list) or any(type(i)is not int or not 0<=i<512 for i in ix) or ix!=sorted(set(ix)):raise ValueError("qualified retained text rows required")
        if arms[0]["restore_slots"]:raise ValueError("control cannot restore from itself")


def expected_counts(arms):
    acts=sum(len(a["slots"])+(len(a["rows_by_step"]) if a["C"]=="scheduled" else a["C"]!="base")+len(a.get("restore_slots",[])) for a in arms)
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=79*len(arms),
                acts=acts,receipts=79*len(arms)+acts,images=len(arms))


def has_historical_control(arms):
    first=arms[0]
    return first["id"]in {"sham","I","T","IT","control"} and (first["C"]=="base" or first["C"]=="scheduled" and not any(first["rows_by_step"].values()))


def worker():
    setup_started=time.monotonic()
    def setup_event(phase):print("MDB_BINDING_SETUP "+json.dumps(dict(phase=phase,seconds=time.monotonic()-setup_started),sort_keys=True),flush=True)
    payload=Path.cwd();config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    if request["config"]!=config or (config["panel"]!="search" and config["arms"]!=specs(config["panel"],config["prime"])) or any(sha(payload/n)!=v for n,v in request["source_hashes"].items()):raise ValueError("source/config binding differs")
    if config["panel"]=="search":validate_search(config["arms"])
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path("binding-minimal-result");out.mkdir();save(out/"plan.json",config);save(out/"request.json",request)
    source=Path(config["input_root"]).resolve(strict=True)
    for name,meta in config["input_files"].items():
        p=source/name
        if p.stat().st_size!=meta["bytes"] or sha(p)!=meta["sha256"]:raise ValueError("input custody differs")
        q=Path("input-data")/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    setup_event("source-custody-closed")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    setup_event("runtime-imported")
    started=time.monotonic();counts=dict(loads=0,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0)
    receipts=[];rows=[];traces=[];writes=[];active={"id":"setup"};restoration_source=None
    def event(kind,**fields):
        v=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),**fields)
        with (out/"timeline.jsonl").open("a") as f:f.write(json.dumps(v,sort_keys=True)+"\n")
        print("MDB_BINDING_PROGRAM "+json.dumps(v,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>600:raise RuntimeError("finite binding-program wall budget")
    def array(name,value):
        p=out/"arrays"/(name+".npy");p.parent.mkdir(exist_ok=True);a=value.detach().float().cpu().numpy();np.save(p,a,allow_pickle=False)
        return dict(path=p.relative_to(out).as_posix(),sha256=sha(p),shape=list(a.shape),dtype=str(a.dtype))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError("CUDA admission required")
    root=Path(config["model_path"]);inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["checkpoint_sha256"]:raise ValueError("checkpoint differs")
    setup_event("checkpoint-verified")
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True);counts["loads"]+=1
    for c in (pipe.transformer,pipe.vae):c.eval().requires_grad_(False).to("cuda")
    target_entries={"blocks":0}
    def target_entry(module,args):target_entries["blocks"]+=1
    entry_handles=[b.register_forward_pre_hook(target_entry) for b in
                   [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    setup_event("model-resident")
    original=LocalStore("input-data/store");store=LocalStore(out/"store");parents={k:original.load(v,device="cuda") for k,v in config["parents"].items()}
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=1,phase="joint",index=4).items()):raise ValueError("parent lane/clock differs")
        store.save(cut)
    for key in parents["base"].payload:
        if key not in {"image","text","conditioning","latent"} and describe(parents["base"].payload[key])!=describe(parents["donor"].payload[key]):raise ValueError("parent nuisance differs: "+key)
    def receipt(value):
        wall();store.receipt(value);body=value.to_dict();receipts.append(dict(row=active["id"],body=body));counts["receipts"]+=1
        counts["native_transitions" if body["operation"]=="continue" else "acts"]+=body.get("steps",1)
        if counts["native_transitions"]>expected_counts(config["arms"])["native_transitions"] or counts["acts"]>expected_counts(config["arms"])["acts"]:raise RuntimeError("finite operation budget")
    def capture(n):c=n.session.capture(retain=False);store.save(c);return c
    def apply(n,slot,value,label):
        before=capture(n);act=Act.replace(slot,value,name=label);r=n.apply(act);after=capture(n)
        if describe(after.payload)!=describe({**before.payload,slot:value}):raise ValueError("undeclared restoration")
        return dict(slot=slot,before=before.fingerprint,after=after.fingerprint,act=act.manifest(),receipt=r.to_dict())
    tx=ForecastTransaction(out/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),source_identity=identity(request["source_hashes"]),
        frontier=dict(kind="architectural",observed="Two previously audited step1/joint4 parents and prior source panels declared in config. Current candidate predictions seal before this worker's first target entry; historical controls remain explicitly source-assisted."),
        execution_scope="Externally admitted native FLUX.2 Klein4B CUDA BF16 B1; model-entry census recorded separately",
        budgets=dict(max_native_calls=1,max_wall_ms=600000))
    tx.predict("program_hypotheses",lambda _:dict(predictions=dict(hypotheses=config["hypotheses"],arms=config["arms"],counts_at_seal=dict(counts),target_entries_before_seal=dict(target_entries),equation_forecasts=config["equation_forecasts"])))
    seal=tx.seal();event("sealed",seal=seal)
    with tx:
        def execute(_):
            nonlocal restoration_source
            for spec in config["arms"]:
                name=spec["id"];active["id"]=name;n=NativeCutNavigator(Session.from_cut(adapter,parents["base"]),max_steps=79,on_receipt=receipt);actions=[]
                event("branch_started",row=name)
                for slot in spec["slots"]:actions.append(apply(n,slot,parents["base" if name=="sham" else "donor"].payload[slot],name+"."+slot))
                commits=[];marks={}
                while n.point["logical_step"]<4:
                    wall();point=n.point
                    if point["logical_step"]==2 and point["phase"]=="project":
                        marks["project2_before_C"]=capture(n).fingerprint
                        event("boundary",row=name,point=point)
                    if point["logical_step"]==2 and point["phase"]=="project" and spec["C"]!="base":
                        value=parents["base"].payload["conditioning"].clone();donor=parents["donor"].payload["conditioning"]
                        indices=(spec["rows_by_step"]["2"] if spec["C"]=="scheduled" else list(range(value.shape[1])) if spec["C"]=="all" else config["conditioning_rows"][spec["C"]])
                        value[:,indices]=donor[:,indices];actions.append(apply(n,"conditioning",value,name+".futureC"))
                    if point["logical_step"]==3 and point["phase"]=="project" and spec["C"]=="scheduled":
                        value=parents["base"].payload["conditioning"].clone();indices=spec["rows_by_step"]["3"]
                        value[:,indices]=parents["donor"].payload["conditioning"][:,indices]
                        actions.append(apply(n,"conditioning",value,name+".C.step3"))
                    if point["phase"]=="project" and point["logical_step"] in (2,3):
                        marks["project"+str(point["logical_step"])+"_input"]=capture(n).fingerprint
                    if point["phase"] in {"joint","single","readout"}:
                        sample={}
                        for slot in ("image","text"):
                            v=n.session.read(slot);ix=np.unique(np.linspace(0,v.shape[1]-1,8).astype(int)).tolist()
                            if slot=="text":ix=sorted(set(ix+config["conditioning_rows"]["color"]))
                            sample[slot]=dict(array(name+f".trace{len(traces)}."+slot,v[:,ix]),rows=ix)
                        traces.append(dict(row=name,point=point,fields=sample))
                    if point["logical_step"]==2 and point["phase"]=="project":
                        marks["project2_input"]=capture(n).fingerprint
                    handles=[];native_ports={}
                    if point["logical_step"]==2 and point["phase"]=="joint" and point["index"]==0:
                        entry=capture(n);marks["joint0_entry"]=entry.fingerprint
                        block=pipe.transformer.transformer_blocks[0]
                        def attn_hook(module,args,output):
                            native_ports["attention"]=output[0].detach().clone()
                            native_ports["attention_text"]=output[1].detach().clone()
                        def ff_hook(module,args,output):
                            native_ports["mlp_input"]=args[0].detach().clone()
                            native_ports["mlp"]=output.detach().clone()
                        def fft_hook(module,args,output):
                            native_ports["mlp_text_input"]=args[0].detach().clone()
                            native_ports["mlp_text"]=output.detach().clone()
                        handles=[block.attn.register_forward_hook(attn_hook),block.ff.register_forward_hook(ff_hook),
                                 block.ff_context.register_forward_hook(fft_hook)]
                    try:n.step()
                    finally:
                        for h in handles:h.remove()
                    if native_ports:
                        from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
                        (_,_,ga),(_,_,gm)=Flux2Modulation.split(entry.payload["context"]["img_mod"],2)
                        actual=capture(n);marks["joint0_exit"]=actual.fingerprint
                        equation=(entry.payload["image"]+ga*native_ports["attention"])+gm*native_ports["mlp"]
                        if not torch.equal(equation,actual.payload["image"]):raise ValueError("observed writer equation differs")
                        native_ports.update(gate_attention=ga,gate_mlp=gm)
                        writes.append(dict(row=name,point=point,entry=entry.fingerprint,exit=actual.fingerprint,
                            fields={k:array(name+".J00."+k,v) for k,v in native_ports.items()},equation_exact=True))
                    if "restore_slots" in spec and point["logical_step"]==2 and point["phase"]=="joint" and point["index"]==spec["restore_index"]:
                        boundary=capture(n);marks["restore_boundary"]=boundary.fingerprint
                        if name=="control":restoration_source=boundary.fingerprint
                        if restoration_source is None:raise ValueError("matched restoration source absent")
                        marks["restoration_source"]=restoration_source
                        baseline=store.load(restoration_source,device="cuda")
                        for slot in spec["restore_slots"]:
                            value=baseline.payload[slot]
                            if slot=="text" and "text_keep_rows" in spec:
                                value=value.clone();ix=spec["text_keep_rows"];value[:,ix]=n.session.read(slot)[:,ix]
                            action=apply(n,slot,value,name+".restore."+slot)
                            if slot=="text" and "text_keep_rows" in spec:action["preserved_rows"]=spec["text_keep_rows"]
                            action["source_cut"]=restoration_source;actions.append(action)
                        marks["post_restore"]=capture(n).fingerprint
                        event("carriers_restored",row=name,slots=spec["restore_slots"],source=restoration_source)
                    if point["logical_step"]==2 and point["phase"]=="joint" and point["index"]==4:
                        marks["joint4_exit"]=capture(n).fingerprint
                    if point["phase"]=="scheduler":
                        c=capture(n);commits.append(dict(step=n.point["logical_step"],cut=c.fingerprint,latent=array(name+".z"+str(n.point["logical_step"]),c.payload["latent"])))
                final=capture(n);p=out/(name+".png");adapter.decode(n.session,height=512,width=512).save(p);counts["images"]+=1
                row=dict(**spec,parent=parents["base"].fingerprint,actions=actions,final=final.fingerprint,commits=commits,marks=marks,png=p.name,png_sha256=sha(p),latent=array(name+".final",final.payload["latent"]))
                rows.append(row);save(out/("completed-"+name+".json"),dict(row=row,writes=[w for w in writes if w["row"]==name],
                    traces=[t for t in traces if t["row"]==name],receipts=[r for r in receipts if r["row"]==name]));event("image_complete",row=name)
            return dict(rows=rows,counts=dict(counts))
        tx.execute("native",execute)
    save(out/"rows.json",rows);save(out/"receipts.json",receipts);save(out/"traces.json",traces);save(out/"writes.json",writes)
    if has_historical_control(config["arms"]) and rows[0]["png_sha256"]!=config["control_png_sha256"]:raise ValueError("same-parent historical control differs")
    if config["panel"]=="streams" and rows[-1]["png_sha256"]!=config["ITC_png_sha256"]:raise ValueError("positive control differs")
    expected=expected_counts(config["arms"])
    if counts!=expected:raise ValueError("physical census differs: "+str(counts))
    if target_entries["blocks"]!=71*len(rows):raise ValueError("native block-entry census differs")
    for handle in entry_handles:handle.remove()
    summary=dict(schema="mdb-binding-minimal-v1",job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],counts=counts,seal=seal,
        transaction=verify_transaction(tx.root),target_entries=target_entries,worker_seconds=time.monotonic()-started,output_root=str(scratch),mechanics_status="pending-independent-audit",terminal_status="unassessed",sham_exact=True)
    save(out/"summary.json",summary);event("complete");save(out/"files.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    if config.get("directory_custody"):
        print("MDB_DIRECTORY_READY "+json.dumps(dict(job_id=summary["job_id"],request_sha256=request["request_sha256"],
            output_root=str((scratch/out).resolve()),files_manifest_sha256=sha(out/"files.json"),summary_sha256=sha(out/"summary.json")),sort_keys=True),flush=True)
    else:runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](out,summary,compression_level=1)


def stage(run,source,panel,prime,forecasts,vram_mb=13000):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    if vram_mb not in (11500,13000):raise ValueError("VRAM reservation outside measured lane bounds")
    here=Path(__file__).resolve().parent;data=source/"run-02/extracted/six-areas-result";read=lambda p:json.loads(p.read_text())
    audit=read(source/"analysis-v1.json");old=read(data/"plan.json");summary=read(data/"summary.json");observed=read(data/"observed.json");rows={r["id"]:r for r in read(data/"rows.json")};tokens=read(data/"tokens.json")
    bind_environment(read(source/"binding-02/request.json")["execution_environment"])
    if audit["mechanics_status"]!="verified":raise ValueError("audited source required")
    parents=dict(base=observed["b0r0"]["joint4_entry"],donor=observed["b1r0"]["joint4_entry"]);manifest=read(data/"files.json");names=set()
    for fp in parents.values():
        name="store/cuts/"+fp+".json";names.add(name);cut=read(data/name)
        names.update("store/pages/"+v+".json" for v in cut["slots"].values());names.update("store/blobs/"+v+".safetensors" for v in cut["physical_blobs"])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError("selected CAS differs")
        files[name]=dict(bytes=(data/name).stat().st_size,sha256=manifest[name])
    t=tokens["b0r0"];crows=dict(padding=t["groups"]["padding"],valid=[i for i,v in enumerate(t["attention_mask"]) if v],color=sorted(set(t["groups"]["red"]+t["groups"]["blue"])))
    if tokens["b1r0"]["attention_mask"]!=t["attention_mask"] or [i for i,(a,b) in enumerate(zip(t["input_ids"],tokens["b1r0"]["input_ids"])) if a!=b]!=crows["color"]:
        raise ValueError("color-only token contrast differs")
    config=dict(schema="mdb-binding-minimal-plan-v1",**{k:old[k] for k in ("model","model_path","checkpoint_sha256","output_scratch")},parents=parents,
        input_root=summary["output_root"]+"/six-areas-result",input_files=files,conditioning_rows=crows,arms=(read(forecasts)["arms"] if panel=="search" else specs(panel,prime)),panel=panel,prime=prime,
        base_png_sha256=rows["b0r0"]["png_sha256"],source_audit_sha256=sha(source/"analysis-v1.json"),
        control_png_sha256=(rows["b0r0"]["png_sha256"] if panel=="streams" else read(source/"binding-02/extracted/binding-program-result/rows.json")[[v["id"] for v in read(source/"binding-02/extracted/binding-program-result/rows.json")].index(prime)]["png_sha256"]),
        ITC_png_sha256=next(v["png_sha256"] for v in read(source/"binding-02/extracted/binding-program-result/rows.json") if v["id"]=="ITC"),
        equation_forecasts=read(forecasts),equation_forecasts_sha256=sha(forecasts),
        hypotheses=(dict(
            categorical=dict(IC="blue sphere left of red cube",TC="red sphere left of blue cube"),
            signed_consumer="IC has a larger positive RGB projection toward the natural binding donor than TC.",
            numerical="At project2, IC and TC Z2 match prior I and T histories exactly. Full project2 and J00 forecasts use those observed prefixes and independent equations before new targets.",
            source_closure="With full donor C, IC/TC/ITC native project2 text is exactly equal; initial image projection depends only on receiving Z2.",
            scope="One checkpoint/seed/parent; I-only and ITC are historical controls. No independent edited-RGB forecast. Retain every miss.")
            if panel=="streams" else dict(
            categorical={prime+"C.all":"blue sphere left of red cube",prime+"C.valid":"blue sphere left of red cube",
                         prime+"C.color":"blue sphere left of red cube",prime+"C.padding":"red sphere left of blue cube"},
            signed_consumer="Predict valid/color C rows move RGB positively toward the binding donor; padding-only response is smaller than valid.",
            scope="Prospective tentative row-sufficiency hypotheses on a primed receiver; retain misses. Native projected padding is contextual and may contradict this hypothesis.")))

    if panel=="search":
        search=read(forecasts);validate_search(search["arms"])
        config.update(arms=search["arms"],hypotheses=search["hypotheses"],search=search,directory_custody=True)
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config);worker=Path(__file__).resolve();extras=[(here/"flux_family_custody.py","flux_family_custody.py"),(run/"plan.json","plan.json")]
    pre=run/"preflight.v1.json";save(pre,dict(schema_version="mrun-preflight-config-v1",hashed_roots=[worker.name,"flux_family_custody.py","plan.json",pre.name],path_token_allowlist=["binding-minimal-result","input-data",config["input_root"],config["model_path"],config["output_scratch"]]));extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request=dict(schema="mdb-binding-minimal-request-v1",config=config,model=config["model"],task_family="forward",backend="diffusers",device="cuda",dtype="bfloat16",batch_size=1,height=512,width=512,needs=dict(cuda=True,host="beast"),execution_environment=intent(),config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=len(config["arms"]),logical_branches=len(config["arms"]),logical_checkpoint_replays=len(config["arms"]),max_local_work_items=len(config["arms"]),checkpoint_policy="Five serial same-parent futures, observed prefixes and sealed independent equations, native J00 ports,0encodes/noise,no retries."),resource_exception=dict(ram_mb=10000,vram_mb=13000,disk_gb=1,calibration_job=summary["job_id"],measured_peak_rss_mb=8332.1,measured_peak_vram_mb=9606,reason="Same no-encoder resident BF16 Klein4B lane;9 futures and sparse witnesses/typed cuts; no independent parameter banks. Existing noencoder10GBRAM13GBVRAM bound;1GBroot admission/HDDscratch/cache/output."))
    request["resource_exception"].update(vram_mb=vram_mb,reason=f"Same no-encoder BF16 Klein4B B1 lane: prior peak VRAM9606MB, current six-arm peak9268MB. Declared{vram_mb}MB plus standard guard ceiling; no backend/batch/math change. HDD scratch/cache/output, 1GBroot admission.")
    request["scheduling_contract"]["checkpoint_policy"]=f"{len(config['arms'])} serial same-parent futures; sealed predictions, native J00 ports,0encodes/noise,no retries."
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request);stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request["request_sha256"])


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    r=json.loads((run/"request.json").read_text())
    bind_environment(r["execution_environment"])
    if r["execution_environment"]!=intent() or (run/"submission.json").exists() or any(sha(run/"payload"/n)!=v for n,v in r["source_hashes"].items()):raise ValueError("mutated/admitted stage or environment")
    cmd=worker_command("flux_binding_minimal.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    vram=r["resource_exception"]["vram_mb"]
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-flux-binding-minimal",payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],ram_mb=10000,vram_mb=vram,disk_gb=1,cpu_threads=4,est_wall_s=300,timeout_s=780,retry_on_kill=False,preflight=True,priority=70,
        note=f"Seed109 Klein BF16 B1 512px,{len(r['config']['arms'])} saved-parent futures,{79*len(r['config']['arms'])} native transitions,4-step native RGB. Sealed row/clock or receiver predictions; J00 native ports. 0encodes/noise,1 resident load,serial,no retries. Stop declared image budget/600worker780outer; calibrated10GBRAM{vram}MBVRAM,HDDscratch.")
    save(run/"submission.json",dict(job_id=str(job)));return dict(job_id=str(job),run=str(run))


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--source",type=Path);p.add_argument("--panel",choices=["streams","subsets","search"],default="streams");p.add_argument("--prime",choices=["I","T","IT"],default="I");p.add_argument("--forecasts",type=Path);p.add_argument("--vram-mb",type=int,default=13000);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve(),a.panel,a.prime,a.forecasts.resolve(),a.vram_mb)
    print(json.dumps(r,sort_keys=True))


if __name__=="__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER")=="1":
        worker_root=Path.cwd()
        try:
            worker()
        except Exception as exc:
            output=Path("binding-minimal-result")
            if output.is_dir() and not (output/"files.json").exists():
                request=json.loads((output/"request.json").read_text())
                failure=dict(schema="mdb-binding-minimal-failure-v1",job_id=os.environ.get("MRUN_JOB_ID"),request_sha256=request["request_sha256"],error=f"{type(exc).__name__}: {exc}",mechanics_status="errored")
                save(output/"failure.json",failure);save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
                runpy.run_path(str(worker_root/"flux_family_custody.py"))["emit"](output,failure,compression_level=1)
            raise
    else:
        cli()
