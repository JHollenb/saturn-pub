"""Prospective mathematical image-write corrections and native recomputation."""
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
def specs(panel="writers", prime="I"):
    return [dict(id="IC.sham",slots=["image"],C="all"),
            *[dict(id=name,slots=["image"],C="all",restore=name)
              for name in ("A.fixed","M.fixed","AM.predicted","A.recomputed.native")]]


def expected_counts(arms):
    acts=sum(len(a["slots"])+(a["C"]!="base")+("restore" in a) for a in arms)
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=79*len(arms),
                acts=acts,receipts=79*len(arms)+acts,images=len(arms),candidate_blocks=sum("restore" in a for a in arms))


def worker():
    payload=Path.cwd();config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    if request["config"]!=config or config["arms"]!=specs(config["panel"],config["prime"]) or any(sha(payload/n)!=v for n,v in request["source_hashes"].items()):raise ValueError("source/config binding differs")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path("binding-writers-result");out.mkdir();save(out/"plan.json",config);save(out/"request.json",request)
    source=Path(config["input_root"]).resolve(strict=True)
    for name,meta in config["input_files"].items():
        p=source/name
        if p.stat().st_size!=meta["bytes"] or sha(p)!=meta["sha256"]:raise ValueError("input custody differs")
        q=Path("input-data")/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for record in config["writer_manifest"]["ports"].values():
        p=Path(config["writer_manifest"]["writer_root"])/record["path"]
        if p.stat().st_size!=record["bytes"] or sha(p)!=record["sha256"]:raise ValueError("writer operand custody differs")
    for name,record in config["writer_manifest"]["files"].items():
        if (payload/name).stat().st_size!=record["bytes"] or sha(payload/name)!=record["sha256"]:raise ValueError("sealed write vector differs")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    started=time.monotonic();counts=dict(loads=0,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0,candidate_blocks=0)
    receipts=[];rows=[];traces=[];writes=[];candidates=[];active={"id":"setup"}
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
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True);counts["loads"]+=1
    for c in (pipe.transformer,pipe.vae):c.eval().requires_grad_(False).to("cuda")
    target_entries={"blocks":0}
    def target_entry(module,args):target_entries["blocks"]+=1
    entry_handles=[b.register_forward_pre_hook(target_entry) for b in
                   [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    original=LocalStore("input-data/store");store=LocalStore(out/"store");parents={k:original.load(v,device="cuda") for k,v in config["parents"].items()}
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=1,phase="joint",index=4).items()):raise ValueError("parent lane/clock differs")
        store.save(cut)
    for key in parents["base"].payload:
        if key not in {"image","text","conditioning","latent"} and describe(parents["base"].payload[key])!=describe(parents["donor"].payload[key]):raise ValueError("parent nuisance differs: "+key)
    from mdb.tensor_archive import load_tensor_archive
    def oldport(label):
        record=config["writer_manifest"]["ports"][label]
        a=np.load(Path(config["writer_manifest"]["writer_root"])/record["path"],allow_pickle=False)
        return torch.from_numpy(a.copy()).to(device="cuda",dtype=torch.bfloat16)
    def vector(name):
        return torch.from_numpy(load_tensor_archive(payload,config["writer_manifest"]["vectors"][name]).copy()).to(device="cuda",dtype=torch.bfloat16)
    def receipt(value):
        wall();store.receipt(value);body=value.to_dict();receipts.append(dict(row=active["id"],body=body));counts["receipts"]+=1
        counts["native_transitions" if body["operation"]=="continue" else "acts"]+=body.get("steps",1)
        if counts["native_transitions"]>expected_counts(config["arms"])["native_transitions"] or counts["acts"]>32:raise RuntimeError("finite operation budget")
    def capture(n):c=n.session.capture(retain=False);store.save(c);return c
    def apply(n,slot,value,label):
        before=capture(n);act=Act.replace(slot,value,name=label);r=n.apply(act);after=capture(n)
        if describe(after.payload)!=describe({**before.payload,slot:value}):raise ValueError("undeclared restoration")
        return dict(slot=slot,before=before.fingerprint,after=after.fingerprint,act=act.manifest(),receipt=r.to_dict())
    tx=ForecastTransaction(out/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),source_identity=identity(request["source_hashes"]),
        frontier=dict(kind="architectural",observed="two previously audited step1/joint4 parents and prior natural RGB; prefixes observed in prior runs; IC and TC suffixes unexecuted when initial predictions seal"),
        budgets=dict(max_native_calls=1,max_wall_ms=600000))
    tx.predict("program_hypotheses",lambda _:dict(predictions=dict(hypotheses=config["hypotheses"],arms=config["arms"],counts_at_seal=dict(counts),target_entries_before_seal=dict(target_entries),equation_forecasts=config["equation_forecasts"])))
    seal=tx.seal();event("sealed",seal=seal)
    with tx:
        def execute(_):
            for spec in config["arms"]:
                name=spec["id"];active["id"]=name;n=NativeCutNavigator(Session.from_cut(adapter,parents["base"]),max_steps=79,on_receipt=receipt);actions=[]
                for slot in spec["slots"]:actions.append(apply(n,slot,parents["base" if name=="sham" else "donor"].payload[slot],name+"."+slot))
                commits=[];marks={}
                while n.point["logical_step"]<4:
                    wall();point=n.point
                    if point["logical_step"]==2 and point["phase"]=="project":
                        marks["project2_before_C"]=capture(n).fingerprint
                    if point["logical_step"]==2 and point["phase"]=="project" and spec["C"]!="base":
                        value=parents["base"].payload["conditioning"].clone();donor=parents["donor"].payload["conditioning"]
                        indices=list(range(value.shape[1])) if spec["C"]=="all" else config["conditioning_rows"][spec["C"]]
                        value[:,indices]=donor[:,indices];actions.append(apply(n,"conditioning",value,name+".futureC"))
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
                        candidate=None;predicted=None
                        if "restore" in spec:
                            mode=spec["restore"]
                            attention=oldport("base_attention") if mode!="M.fixed" else None
                            mlp=oldport("edited_mlp") if mode=="A.fixed" else oldport("base_mlp") if mode in ("M.fixed","AM.predicted") else None
                            candidate=runpy.run_path(str(payload/"flux_joint_writes.py"))["joint_candidate"](block,entry,attention=attention,mlp=mlp)
                            counts["candidate_blocks"]+=1
                            predicted=vector("AM.predicted" if mode=="A.recomputed.native" else mode)
                            candidates.append(dict(row=name,entry=entry.fingerprint,mode=mode,predicted_image_exact=bool(torch.equal(predicted,candidate["image"])),
                                predicted=array(name+".predicted_exit",predicted),native_candidate=array(name+".native_candidate_exit",candidate["image"]),
                                native_candidate_text=array(name+".native_candidate_text",candidate["text"]),equation_exact=candidate["equation_exact"]))
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
                        if "restore" in spec:
                            if describe(candidate["text"])!=describe(actual.payload["text"]):raise ValueError("candidate text closure differs")
                            installed=candidate["image"] if spec["restore"]=="A.recomputed.native" else predicted
                            actions.append(apply(n,"image",installed,name+".writer_exit"))
                            marks["post_writer_Act"]=capture(n).fingerprint
                    if point["logical_step"]==2 and point["phase"]=="joint" and point["index"]==4:
                        marks["joint4_exit"]=capture(n).fingerprint
                    if point["phase"]=="scheduler":
                        c=capture(n);commits.append(dict(step=n.point["logical_step"],cut=c.fingerprint,latent=array(name+".z"+str(n.point["logical_step"]),c.payload["latent"])))
                final=capture(n);p=out/(name+".png");adapter.decode(n.session,height=512,width=512).save(p);counts["images"]+=1
                row=dict(**spec,parent=parents["base"].fingerprint,actions=actions,final=final.fingerprint,commits=commits,marks=marks,png=p.name,png_sha256=sha(p),latent=array(name+".final",final.payload["latent"]))
                rows.append(row);event("image_complete",row=name)
            return dict(rows=rows,counts=dict(counts))
        tx.execute("native",execute)
    if rows[0]["png_sha256"]!=config["control_png_sha256"]:raise ValueError("same-parent historical control differs")
    if config["panel"]=="streams" and rows[-1]["png_sha256"]!=config["ITC_png_sha256"]:raise ValueError("positive control differs")
    expected=expected_counts(config["arms"])
    if counts!=expected:raise ValueError("physical census differs: "+str(counts))
    if target_entries["blocks"]!=71*len(rows)+counts["candidate_blocks"]:raise ValueError("native block-entry census differs")
    for handle in entry_handles:handle.remove()
    save(out/"rows.json",rows);save(out/"receipts.json",receipts);save(out/"traces.json",traces);save(out/"writes.json",writes);save(out/"candidates.json",candidates)
    summary=dict(schema="mdb-binding-writers-v1",job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],counts=counts,seal=seal,
        transaction=verify_transaction(tx.root),target_entries=target_entries,worker_seconds=time.monotonic()-started,output_root=str(scratch),mechanics_status="pending-independent-audit",terminal_status="unassessed",sham_exact=True)
    save(out/"summary.json",summary);event("complete");save(out/"files.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](out,summary,compression_level=1)


def stage(run,source,panel,prime,forecasts):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
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
    config=dict(schema="mdb-binding-writers-plan-v1",**{k:old[k] for k in ("model","model_path","checkpoint_sha256","output_scratch")},parents=parents,
        input_root=summary["output_root"]+"/six-areas-result",input_files=files,conditioning_rows=crows,arms=specs(panel,prime),panel=panel,prime=prime,
        base_png_sha256=rows["b0r0"]["png_sha256"],source_audit_sha256=sha(source/"analysis-v1.json"),
        control_png_sha256=read(forecasts)["positive_control_png_sha256"],
        ITC_png_sha256=None,equation_forecasts=read(forecasts),equation_forecasts_sha256=sha(forecasts),
        writer_manifest=read(forecasts),hypotheses=read(forecasts)["hypotheses"])

    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config);worker=Path(__file__).resolve();extras=[(here/"flux_family_custody.py","flux_family_custody.py"),(here/"flux_joint_writes.py","flux_joint_writes.py"),(run/"plan.json","plan.json")]+[(forecasts.parent/name,name) for name in config["writer_manifest"]["files"]]
    pre=run/"preflight.v1.json";save(pre,dict(schema_version="mrun-preflight-config-v1",hashed_roots=[worker.name,"flux_family_custody.py","flux_joint_writes.py","plan.json",pre.name,*config["writer_manifest"]["files"]],path_token_allowlist=["binding-minimal-result","input-data",config["input_root"],config["model_path"],config["output_scratch"],config["writer_manifest"]["writer_root"]]));extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request=dict(schema="mdb-binding-writers-request-v1",config=config,model=config["model"],task_family="forward",backend="diffusers",device="cuda",dtype="bfloat16",batch_size=1,height=512,width=512,needs=dict(cuda=True,host="beast"),execution_environment=intent(),config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=len(config["arms"]),logical_branches=len(config["arms"]),logical_checkpoint_replays=len(config["arms"]),max_local_work_items=len(config["arms"]),checkpoint_policy="Five same-parent writer futures,395 native transitions,4 direct candidate blocks,14 Acts,5 images.3 full mathematical correction fields and native A recomputation.0encodes/noise,serial,noretries."),resource_exception=dict(ram_mb=10000,vram_mb=13000,disk_gb=1,calibration_job=summary["job_id"],measured_peak_rss_mb=8332.1,measured_peak_vram_mb=9606,reason="Same no-encoder resident BF16 Klein4B lane;9 futures and sparse witnesses/typed cuts; no independent parameter banks. Existing noencoder10GBRAM13GBVRAM bound;1GBroot admission/HDDscratch/cache/output."))
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request);stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request["request_sha256"])


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    r=json.loads((run/"request.json").read_text())
    bind_environment(r["execution_environment"])
    if r["execution_environment"]!=intent() or (run/"submission.json").exists() or any(sha(run/"payload"/n)!=v for n,v in r["source_hashes"].items()):raise ValueError("mutated/admitted stage or environment")
    cmd=worker_command("flux_binding_writers.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-flux-binding-writers",payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],ram_mb=10000,vram_mb=13000,disk_gb=1,cpu_threads=4,est_wall_s=300,timeout_s=780,retry_on_kill=False,preflight=True,priority=70,
        note="Klein BF16 seed109 same-parent I+C.5 futures:IC control,A fixed-current-M,M restore,AM predicted,A native recompute.395 transitions14Acts4 explicit candidate blocks5RGB.Three full BF16 write vectors sealed offline from saved ports; compare native candidate fields and futures.0encodes/noise,1load,serial,noretries. Stop5images/600worker780outer.")
    save(run/"submission.json",dict(job_id=str(job)));return dict(job_id=str(job),run=str(run))


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--source",type=Path);p.add_argument("--panel",choices=["writers"],default="writers");p.add_argument("--prime",choices=["I","T","IT"],default="I");p.add_argument("--forecasts",type=Path);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve(),a.panel,a.prime,a.forecasts.resolve())
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
                failure=dict(schema="mdb-binding-writers-failure-v1",job_id=os.environ.get("MRUN_JOB_ID"),request_sha256=request["request_sha256"],error=f"{type(exc).__name__}: {exc}",mechanics_status="errored")
                save(output/"failure.json",failure);save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
                runpy.run_path(str(worker_root/"flux_family_custody.py"))["emit"](output,failure,compression_level=1)
            raise
    else:
        cli()

