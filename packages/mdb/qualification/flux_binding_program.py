"""Common-parent binding edits: current I/T versus repeated future C sources."""
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
def specs():
    return [dict(id="sham",slots=["image","text"],C="base"),dict(id="I",slots=["image"],C="base"),
        dict(id="T",slots=["text"],C="base"),dict(id="IT",slots=["image","text"],C="base"),
        dict(id="C.all",slots=[],C="all"),dict(id="C.valid",slots=[],C="valid"),
        dict(id="C.padding",slots=[],C="padding"),dict(id="C.color",slots=[],C="color"),
        dict(id="ITC",slots=["image","text"],C="all")]


def worker():
    payload=Path.cwd();config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    if request["config"]!=config or config["arms"]!=specs() or any(sha(payload/n)!=v for n,v in request["source_hashes"].items()):raise ValueError("source/config binding differs")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path("binding-program-result");out.mkdir();save(out/"plan.json",config);save(out/"request.json",request)
    source=Path(config["input_root"]).resolve(strict=True)
    for name,meta in config["input_files"].items():
        p=source/name
        if p.stat().st_size!=meta["bytes"] or sha(p)!=meta["sha256"]:raise ValueError("input custody differs")
        q=Path("input-data")/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    started=time.monotonic();counts=dict(loads=0,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0)
    receipts=[];rows=[];traces=[];active={"id":"setup"}
    def event(kind,**fields):
        v=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),**fields)
        with (out/"timeline.jsonl").open("a") as f:f.write(json.dumps(v,sort_keys=True)+"\n")
        print("MDB_BINDING_PROGRAM "+json.dumps(v,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>480:raise RuntimeError("finite binding-program wall budget")
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
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    original=LocalStore("input-data/store");store=LocalStore(out/"store");parents={k:original.load(v,device="cuda") for k,v in config["parents"].items()}
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=1,phase="joint",index=4).items()):raise ValueError("parent lane/clock differs")
        store.save(cut)
    for key in parents["base"].payload:
        if key not in {"image","text","conditioning","latent"} and describe(parents["base"].payload[key])!=describe(parents["donor"].payload[key]):raise ValueError("parent nuisance differs: "+key)
    def receipt(value):
        wall();store.receipt(value);body=value.to_dict();receipts.append(dict(row=active["id"],body=body));counts["receipts"]+=1
        counts["native_transitions" if body["operation"]=="continue" else "acts"]+=body.get("steps",1)
        if counts["native_transitions"]>711 or counts["acts"]>20:raise RuntimeError("finite operation budget")
    def capture(n):c=n.session.capture(retain=False);store.save(c);return c
    def apply(n,slot,value,label):
        before=capture(n);act=Act.replace(slot,value,name=label);r=n.apply(act);after=capture(n)
        if describe(after.payload)!=describe({**before.payload,slot:value}):raise ValueError("undeclared restoration")
        return dict(slot=slot,before=before.fingerprint,after=after.fingerprint,act=act.manifest(),receipt=r.to_dict())
    tx=ForecastTransaction(out/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),source_identity=identity(request["source_hashes"]),
        frontier=dict(kind="architectural",observed="two previously audited step1/joint4 parents and prior natural RGB; no outcomes of this continuation panel"),
        budgets=dict(max_native_calls=1,max_wall_ms=480000))
    tx.predict("program_hypotheses",lambda _:dict(predictions=dict(hypotheses=config["hypotheses"],arms=config["arms"],counts_at_seal=dict(counts))))
    seal=tx.seal();event("sealed",seal=seal)
    with tx:
        def execute(_):
            for spec in config["arms"]:
                name=spec["id"];active["id"]=name;n=NativeCutNavigator(Session.from_cut(adapter,parents["base"]),max_steps=79,on_receipt=receipt);actions=[]
                for slot in spec["slots"]:actions.append(apply(n,slot,parents["base" if name=="sham" else "donor"].payload[slot],name+"."+slot))
                commits=[]
                while n.point["logical_step"]<4:
                    wall();point=n.point
                    if point["logical_step"]==2 and point["phase"]=="project" and spec["C"]!="base":
                        value=parents["base"].payload["conditioning"].clone();donor=parents["donor"].payload["conditioning"]
                        indices=list(range(value.shape[1])) if spec["C"]=="all" else config["conditioning_rows"][spec["C"]]
                        value[:,indices]=donor[:,indices];actions.append(apply(n,"conditioning",value,name+".futureC"))
                    if point["phase"] in {"joint","single","readout"}:
                        sample={}
                        for slot in ("image","text"):
                            v=n.session.read(slot);ix=np.unique(np.linspace(0,v.shape[1]-1,8).astype(int)).tolist();sample[slot]=array(name+f".trace{len(traces)}."+slot,v[:,ix])
                        traces.append(dict(row=name,point=point,fields=sample))
                    n.step()
                    if point["phase"]=="scheduler":
                        c=capture(n);commits.append(dict(step=n.point["logical_step"],cut=c.fingerprint,latent=array(name+".z"+str(n.point["logical_step"]),c.payload["latent"])))
                final=capture(n);p=out/(name+".png");adapter.decode(n.session,height=512,width=512).save(p);counts["images"]+=1
                row=dict(**spec,parent=parents["base"].fingerprint,actions=actions,final=final.fingerprint,commits=commits,png=p.name,png_sha256=sha(p),latent=array(name+".final",final.payload["latent"]))
                rows.append(row);event("image_complete",row=name)
            return dict(rows=rows,counts=dict(counts))
        tx.execute("native",execute)
    if rows[0]["png_sha256"]!=config["base_png_sha256"]:raise ValueError("same-parent historical sham image differs")
    expected=dict(loads=1,encodes=0,noise_draws=0,native_transitions=711,acts=13,receipts=724,images=9)
    if counts!=expected:raise ValueError("physical census differs: "+str(counts))
    save(out/"rows.json",rows);save(out/"receipts.json",receipts);save(out/"traces.json",traces)
    summary=dict(schema="mdb-binding-program-v1",job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],counts=counts,seal=seal,
        transaction=verify_transaction(tx.root),worker_seconds=time.monotonic()-started,output_root=str(scratch),mechanics_status="pending-independent-audit",terminal_status="unassessed",sham_exact=True)
    save(out/"summary.json",summary);event("complete");save(out/"files.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](out,summary,compression_level=1)


def stage(run,source):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;data=source/"run-02/extracted/six-areas-result";read=lambda p:json.loads(p.read_text())
    audit=read(source/"analysis-v1.json");old=read(data/"plan.json");summary=read(data/"summary.json");observed=read(data/"observed.json");rows={r["id"]:r for r in read(data/"rows.json")};tokens=read(data/"tokens.json")
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
    config=dict(schema="mdb-binding-program-plan-v1",**{k:old[k] for k in ("model","model_path","checkpoint_sha256","output_scratch")},parents=parents,
        input_root=summary["output_root"]+"/six-areas-result",input_files=files,conditioning_rows=crows,arms=specs(),base_png_sha256=rows["b0r0"]["png_sha256"],
        source_audit_sha256=sha(source/"analysis-v1.json"),hypotheses=dict(
            H1="Predict C.all and ITC switch to a blue sphere left of a red cube while retaining two shapes/order. Exact image match to donor is not predicted; incoming latent remains native base.",
            H2="Predict current I/T-only pulses (I,T,IT) keep the final red sphere/blue cube under native base future conditioning, although geometry/shading collateral may occur. Counterexamples retained.",
            competing_sources="No selected winner for valid-token, padding and two color-token C rows. All are explicit exploratory source-localization arms, not established semantic symbols.",
            scope="All arms share one native step1/joint4 parent and base carried latent. C edits first occur at project2 and persist through project3. 0 new encodes/noise. Whole source C/T/I are observed prior states; no independent numerical forecast of these edited futures."))
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config);worker=Path(__file__).resolve();extras=[(here/"flux_family_custody.py","flux_family_custody.py"),(run/"plan.json","plan.json")]
    pre=run/"preflight.v1.json";save(pre,dict(schema_version="mrun-preflight-config-v1",hashed_roots=[worker.name,"flux_family_custody.py","plan.json",pre.name],path_token_allowlist=["binding-program-result","input-data",config["input_root"],config["model_path"],config["output_scratch"]]));extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request=dict(schema="mdb-binding-program-request-v1",config=config,model=config["model"],task_family="forward",backend="diffusers",device="cuda",dtype="bfloat16",batch_size=1,height=512,width=512,needs=dict(cuda=True,host="beast"),execution_environment=intent(),config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=9,logical_branches=9,logical_checkpoint_replays=9,max_local_work_items=9,checkpoint_policy="9 same-parent native futures; currentI/T versus repeatedfutureC row groups.711 native transitions13 Acts,0 encodes/noise;one finite serial lease,no retries."),resource_exception=dict(ram_mb=10000,vram_mb=13000,disk_gb=1,calibration_job=summary["job_id"],measured_peak_rss_mb=8332.1,measured_peak_vram_mb=9606,reason="Same no-encoder resident BF16 Klein4B lane;9 futures and sparse witnesses/typed cuts; no independent parameter banks. Existing noencoder10GBRAM13GBVRAM bound;1GBroot admission/HDDscratch/cache/output."))
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request);stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request["request_sha256"])


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    r=json.loads((run/"request.json").read_text())
    if r["execution_environment"]!=intent() or (run/"submission.json").exists() or any(sha(run/"payload"/n)!=v for n,v in r["source_hashes"].items()):raise ValueError("mutated/admitted stage or environment")
    cmd=worker_command("flux_binding_program.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-flux-binding-program",payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],ram_mb=10000,vram_mb=13000,disk_gb=1,cpu_threads=4,est_wall_s=300,timeout_s=600,retry_on_kill=False,preflight=True,priority=70,
        note="Saved seed109 Klein BF16 step1/joint4 commonparent.9 futures:sham,I,T,IT,then futureC all/valid/padding/color rows,ITC. Freeze semantic predictions before branches.711transitions13Acts9VAE,0encodes/noise. C firstatproject2/persist3; baseZ retained, no lateZAct.1residentload,serial,noretries;10GBRAM13GBVRAM1GBdisk,HDDscratch/cache. Stop9images/480worker600outer.")
    save(run/"submission.json",dict(job_id=str(job)));return dict(job_id=str(job),run=str(run))


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--source",type=Path);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))


if __name__=="__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER")=="1":
        worker_root=Path.cwd()
        try:
            worker()
        except Exception as exc:
            output=Path("binding-program-result")
            if output.is_dir() and not (output/"files.json").exists():
                request=json.loads((output/"request.json").read_text())
                failure=dict(schema="mdb-binding-program-failure-v1",job_id=os.environ.get("MRUN_JOB_ID"),request_sha256=request["request_sha256"],error=f"{type(exc).__name__}: {exc}",mechanics_status="errored")
                save(output/"failure.json",failure);save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
                runpy.run_path(str(worker_root/"flux_family_custody.py"))["emit"](output,failure,compression_level=1)
            raise
    else:
        cli()
