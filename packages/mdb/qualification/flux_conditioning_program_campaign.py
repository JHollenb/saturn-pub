"""Conditioning interaction interventions at one legal common project frontier."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/"flux_predicted_future_campaign.py"));save,sha,identity=[u[k] for k in ("save","sha","identity")]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    if config!=request["config"] or any(sha(p)!=h for p,h in request["source_hashes"].items()):raise ValueError("staging differs")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path("conditioning-program-result");output.mkdir();save(output/"plan.json",config);save(output/"request.json",request)
    root=Path(config["input_root"]).resolve(strict=True)
    for name,meta in config["input_files"].items():
        src=root/name;dst=Path("input-data")/name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=meta["bytes"] or sha(src)!=meta["sha256"]:raise ValueError("source custody differs")
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open("xb") as f:f.write(src.read_bytes())
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import write_tensor_archive
    helper=runpy.run_path(str(payload/"flux_conditioning_program.py"));cuts_helper=runpy.run_path(str(payload/"flux_scene_dependencies.py"))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={"loads":0,"encodes":0,"native_transitions":0,"acts":0,"receipts":0,"images":0};receipts=[];rows=[]
    def event(kind,**kw):
        row={"kind":kind,"seconds":time.monotonic()-started,"counts":dict(counts),**kw}
        with (output/"timeline.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
        print("MDB_CONDITIONING "+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        row=r.to_dict();receipts.append(row);counts["receipts"]+=1;counts["native_transitions" if row["operation"]=="continue" else "acts"]+=1
        if counts["native_transitions"]>234 or counts["acts"]>8 or time.monotonic()-started>config["max_wall_seconds"]:raise RuntimeError("finite conditioning budget exceeded")
    def array(name,v):return write_tensor_archive(output/"arrays"/(name+".npy"),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    model=Path(config["model_path"]);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob("*")) if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["checkpoint_sha256"]:raise ValueError("checkpoint differs")
    event("loading")
    pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts["loads"]+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda");store=LocalStore(output/"store")
    sources={role:LocalStore(Path("input-data/store")).load(fingerprint,device="cuda") for role,fingerprint in config["input_cuts"].items()}
    for cut in sources.values():
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):raise ValueError("source numeric profile differs")
    frontier=cuts_helper["project_frontier"](adapter,sources["base"]);store.save(frontier)
    values=helper["conditioning_cases"]({k:cpu(v.payload["conditioning"]) for k,v in sources.items()})
    projector=pipe.transformer.context_embedder
    binding={"derived_project_frontier":frontier.fingerprint,"source_cuts":config["input_cuts"],"conditioning":{k:array("E."+k,torch.from_numpy(v).to(torch.bfloat16)) for k,v in values.items()},
        "weight":array("context.weight",projector.weight),"bias":None if projector.bias is None else array("context.bias",projector.bias),
        "scope":"Incoming conditioning and actual native linear projector only; no current projectedT or laterdenoiser outputs supplied. CommonbaseL/time/noise context, derivedlegalproject boundary, source-assisted E compositions."}
    save(output/"binding.json",binding);release={"native":False}
    def guard(module,args):
        if not release["native"]:raise ValueError("context projector called before seal")
    handle=projector.register_forward_pre_hook(guard)
    tx=ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
        frontier={"kind":"architectural","binding":binding},execution_scope="Conditioning-to-initialT independentlinear forecasts; causalE edit with commonlatent history and native finalstep/VAE",
        budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    predicted={};pred_records={};projected={};native_fields={}
    with tx:
        tx.frontier("binding",binding)
        def forecast(_):
            for role,value in values.items():
                p=helper["predict_projection"](value,cpu(projector.weight),None if projector.bias is None else cpu(projector.bias))
                predicted[role]=torch.from_numpy(p).to("cuda",torch.bfloat16);pred_records[role]=array("predictedT."+role,predicted[role])
            save(output/"forecasts.json",pred_records);return {"predictions":{"initial_text":pred_records,"hypotheses":config["hypotheses"]}}
        tx.predict("conditioning_to_text",forecast);seal=tx.seal();release["native"]=True;handle.remove();event("sealed",seal=seal)
        def execute(_):
            reconstruction=NativeCutNavigator(Session.from_cut(adapter,frontier),max_steps=12,on_receipt=on_receipt)
            for _ in range(12):reconstruction.step()
            rebuilt=reconstruction.capture();store.save(rebuilt)
            if describe(rebuilt.payload)!=describe(sources["base"].payload):raise ValueError("derivedfrontier fails historical pre6 reconstruction")
            save(output/"reconstruction.json",{"derived_frontier":frontier.fingerprint,"reconstructed":rebuilt.fingerprint,"historical_pre6":sources["base"].fingerprint,"payload_exact":True,"native_transitions":12})
            def finish(name,nav,actions,initial=()):
                path=list(initial)
                while nav.point["logical_step"]<4:path.append(nav.step().to_dict())
                final=nav.capture();store.save(final);png=output/(name+".png");adapter.decode(nav.session,height=512,width=512).save(png);counts["images"]+=1
                if name=="native.base" and sha(png)!=config["historical_base_png"]:raise ValueError("historical basePNG differs")
                rows.append({"id":name,"actions":actions,"path":path,"final_cut":final.fingerprint,"latent":array(name+".latent",final.payload["latent"]),"png":png.name,"png_sha256":sha(png)});event("image_complete",branch=name)
            for role,value in values.items():
                nav=NativeCutNavigator(Session.from_cut(adapter,frontier),max_steps=28,on_receipt=on_receipt)
                actions=cuts_helper["restore_slots"](nav,{"conditioning":torch.from_numpy(value).to("cuda",torch.bfloat16)},label="E."+role,store=store)
                first=nav.step().to_dict();projected[role]=nav.capture();store.save(projected[role]);native_fields[role]=array("nativeT."+role,projected[role].payload["text"])
                finish("native."+role,nav,actions,[first])
            save(output/"native-fields.json",native_fields)
            for role in ("additive","joint"):
                nav=NativeCutNavigator(Session.from_cut(adapter,projected[role]),max_steps=27,on_receipt=on_receipt)
                actions=cuts_helper["restore_slots"](nav,{"text":predicted[role]},label="predictedT."+role,store=store);finish("predicted."+role,nav,actions)
            return {"counts":dict(counts),"rows":len(rows)}
        tx.execute("native_conditioning_futures",execute)
    save(output/"rows.json",rows);save(output/"receipts.json",receipts)
    if counts!={"loads":1,"encodes":0,"native_transitions":234,"acts":8,"receipts":242,"images":8}:raise ValueError("conditioning census differs: "+str(counts))
    summary={"schema":"mdb-conditioning-program-v1","job_id":os.environ["MRUN_JOB_ID"],"request_sha256":request["request_sha256"],"model_identity":adapter.model_identity,"execution":dict(adapter.execution),"counts":counts,
        "seal":seal,"transaction":verify_transaction(tx.root),"worker_seconds":time.monotonic()-started,"output_root":str(Path.cwd()),"mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary);event("complete");save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](output,summary,compression_level=1)


if __name__=="__main__":worker()
