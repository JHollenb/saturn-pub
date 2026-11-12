"""Text-program RAM factorial and complete entry-only text A/M forecasts."""
import itertools
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/"flux_predicted_future_campaign.py"));save,sha,identity=[u[k] for k in ("save","sha","identity")]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    if config!=request["config"] or any(sha(p)!=v for p,v in request["source_hashes"].items()):raise ValueError("staged binding differs")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path("text-program-result");output.mkdir();save(output/"plan.json",config);save(output/"request.json",request)
    source=Path(config["input_root"]).resolve(strict=True)
    for name,meta in config["input_files"].items():
        src=source/name;dst=Path("input-data")/name
        if src.is_symlink() or not src.resolve().is_relative_to(source) or src.stat().st_size!=meta["bytes"] or sha(src)!=meta["sha256"]:raise ValueError("historical custody differs")
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open("xb") as f:f.write(src.read_bytes())
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="4"
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
    helper=runpy.run_path(str(payload/"flux_text_program.py"));capture=runpy.run_path(str(payload/"flux_single_contraction.py"))["capture_candidate"]
    restore=runpy.run_path(str(payload/"flux_scene_dependencies.py"))["restore_slots"]
    a=runpy.run_path(str(payload/"flux_attention_reference.py"))["predict_attention_field"];m=runpy.run_path(str(payload/"flux_mlp_replacement.py"))["predict_mlp_field"]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={"loads":0,"encodes":0,"native_transitions":0,"acts":0,"receipts":0,"source_blocks":0,"validation_blocks":0,"projections":0,"images":0};receipts=[];rows=[]
    def event(kind,**fields):
        row={"kind":kind,"seconds":time.monotonic()-started,"counts":dict(counts),**fields}
        with (output/"timeline.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
        print("MDB_TEXT_PROGRAM "+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        row=r.to_dict();receipts.append(row);counts["receipts"]+=1;counts["native_transitions" if row["operation"]=="continue" else "acts"]+=1
        if counts["native_transitions"]>171 or counts["acts"]>11 or time.monotonic()-started>config["max_wall_seconds"]:raise RuntimeError("finite text-program budget exceeded")
    def cpu(value):return value.detach().float().cpu().numpy()
    def array(name,value):return write_tensor_archive(output/"arrays"/(name+".npy"),value,root=output)
    root=Path(config["model_path"]);inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["checkpoint_sha256"]:raise ValueError("checkpoint differs")
    event("loading")
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts["loads"]+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    store=LocalStore(output/"store");cuts={}
    for role,fingerprint in config["input_cuts"].items():
        cut=LocalStore(Path("input-data/store")).load(fingerprint,device="cuda")
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):raise ValueError("saved numerical profile differs")
        if any(cut.execution_point[k]!=v for k,v in {"logical_step":3,"phase":"single","index":8}.items()):raise ValueError("requires pre8")
        cuts[role]=cut;store.save(cut)
    low,high=cuts["low"],cuts["high"]
    if describe(low.payload)!=describe({**high.payload,"text":low.payload["text"]}):raise ValueError("source parents differ beyondT")
    block=pipe.transformer.single_transformer_blocks[8];fused=block.attn.to_qkv_mlp_proj;h=low.payload["image"].shape[-1];t=low.payload["text"].shape[1]
    binding={"entries":{k:v.fingerprint for k,v in cuts.items()},"inputs":{k:{"text":array(k+".text",v.payload["text"]),"image":array(k+".image",v.payload["image"])} for k,v in cuts.items()},
        "mod":array("mod",low.payload["context"]["single_mod"]),"rope_cos":array("rope.cos",low.payload["context"]["rope"][0]),"rope_sin":array("rope.sin",low.payload["context"]["rope"][1]),
        "qkv_weight":array("qkv.weight",fused.weight[:3*h]),"mlp_weight":array("mlp.weight",fused.weight[3*h:]),
        "qkv_bias":None if fused.bias is None else array("qkv.bias",fused.bias[:3*h]),"mlp_bias":None if fused.bias is None else array("mlp.bias",fused.bias[3*h:]),
        "norm_q":array("norm.q",block.attn.norm_q.weight),"norm_k":array("norm.k",block.attn.norm_k.weight),"eps":block.norm.eps,"qk_eps":block.attn.norm_q.eps,"heads":block.attn.heads,
        "scope":"Complete text A/M from observed incomingT/I/modulation/actualweights/position. Attention pack [I;T] movesRoPE withrows and keepsallkeys. Currenttargetoutputs unavailable beforeseal."}
    save(output/"binding.json",binding);release={"native":False}
    def guard(module,args):
        if not release["native"]:raise ValueError("block8 entered before text-field forecast seal")
    handle=block.register_forward_pre_hook(guard)
    tx=ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
        frontier={"kind":"architectural","binding":binding},execution_scope="TextA/M forecast beforeblock8; native projection/residual and source-assisted RAMfactorial followed by fixed-image native continuation",
        budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    predicted={};forecast_records={};native={};post={}
    with tx:
        tx.frontier("binding",binding)
        def forecast(_):
            for role,cut in cuts.items():
                result=helper["predict_text_fields"](cpu(cut.payload["text"]),cpu(cut.payload["image"]),cpu(cut.payload["context"]["single_mod"]),
                    qkv_weight=cpu(fused.weight[:3*h]),qkv_bias=None if fused.bias is None else cpu(fused.bias[:3*h]),mlp_weight=cpu(fused.weight[3*h:]),mlp_bias=None if fused.bias is None else cpu(fused.bias[3*h:]),
                    norm_q=cpu(block.attn.norm_q.weight),norm_k=cpu(block.attn.norm_k.weight),eps=block.norm.eps,qk_eps=block.attn.norm_q.eps,heads=block.attn.heads,
                    rope_cos=cpu(cut.payload["context"]["rope"][0]),rope_sin=cpu(cut.payload["context"]["rope"][1]),attention_predictor=a,mlp_predictor=m)
                predicted[role]={k:torch.from_numpy(np.array(result[k],copy=True)).to("cuda",torch.bfloat16) for k in ("A","M")}
                forecast_records[role]={k:array("predicted."+role+"."+k,value) for k,value in predicted[role].items()}
            save(output/"forecasts.json",forecast_records)
            return {"predictions":{"fields":forecast_records,"hypotheses":config["hypotheses"]}}
        tx.predict("text_fields_and_RAM",forecast);seal=tx.seal();release["native"]=True;handle.remove();event("sealed",seal=seal)
        def execute(_):
            observed={}
            for role,cut in cuts.items():
                native[role]=capture(block,cut,single_index=8);counts["source_blocks"]+=1
                cat=native[role]["ports"]["original_concat"]
                observed[role]={"A":array("native."+role+".A",cat[:,:t,:h]),"M":array("native."+role+".M",cat[:,:t,h:]),"text_exit":array("native."+role+".T",native[role]["text"])}
            save(output/"native-fields.json",observed)
            def finish(name,nav,actions,initial=()):
                path=list(initial)
                while nav.point["logical_step"]<4:path.append(nav.step().to_dict())
                final=nav.capture();store.save(final);png=output/(name+".png");adapter.decode(nav.session,height=512,width=512).save(png);counts["images"]+=1
                if name in config["historical_pngs"] and sha(png)!=config["historical_pngs"][name]:raise ValueError("historical PNG differs")
                rows.append({"id":name,"actions":actions,"path":path,"final_cut":final.fingerprint,"latent":array(name+".latent",final.payload["latent"]),"png":png.name,"png_sha256":sha(png)})
                event("image_complete",branch=name)
            for role,cut in cuts.items():
                nav=NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=14,on_receipt=on_receipt);first=nav.step().to_dict();post[role]=nav.capture();store.save(post[role])
                if not torch.equal(post[role].payload["text"],native[role]["text"]) or not torch.equal(post[role].payload["image"],native[role]["image"]):raise ValueError("direct/native source mismatch")
                finish("native."+role,nav,[],[first])
            parts={role:{"R":cut.payload["text"],"A":native[role]["ports"]["original_concat"][:,:t,:h],"M":native[role]["ports"]["original_concat"][:,:t,h:]} for role,cut in cuts.items()}
            writes={}
            for ai,mi in itertools.product((0,1),repeat=2):
                writes[ai,mi]=helper["text_write"](block,native["high"],attention=parts[("low","high")[ai]]["A"],mlp=parts[("low","high")[mi]]["M"]);counts["projections"]+=1
            values={"R%dA%dM%d"%(ri,ai,mi):parts[("low","high")[ri]]["R"]+writes[ai,mi] for ri,ai,mi in itertools.product((0,1),repeat=3)}
            validations={}
            for ri,ai,mi in ((0,1,0),(1,0,1)):
                name="R%dA%dM%d"%(ri,ai,mi);role=("low","high")[ri]
                actual=helper["check_text_override"](block,cuts[role],attention=parts[("low","high")[ai]]["A"],mlp=parts[("low","high")[mi]]["M"],single_index=8);counts["validation_blocks"]+=1
                validations[name]={"text_exact_fullblock":torch.equal(actual,values[name])}
            validations["R0A0M0"]={"text_exact_native_low":torch.equal(values["R0A0M0"],native["low"]["text"])}
            validations["R1A1M1"]={"text_exact_native_high":torch.equal(values["R1A1M1"],native["high"]["text"])}
            save(output/"writer-validations.json",validations)
            for role in ("low","high"):
                values["predicted."+role]=helper["text_exit"](block,native["high"],residual=parts[role]["R"],attention=predicted[role]["A"],mlp=predicted[role]["M"]);counts["projections"]+=1
            values["sham.high"]=post["high"].payload["text"]
            for name,value in values.items():
                nav=NativeCutNavigator(Session.from_cut(adapter,post["high"]),max_steps=13,on_receipt=on_receipt)
                actions=restore(nav,{"text":value},label=name,store=store);finish(name,nav,actions)
            return {"counts":counts,"rows":len(rows)}
        tx.execute("text_writer_futures",execute)
    save(output/"rows.json",rows);save(output/"receipts.json",receipts)
    expected={"loads":1,"encodes":0,"native_transitions":171,"acts":11,"receipts":182,"source_blocks":2,"validation_blocks":2,"projections":6,"images":13}
    if counts!=expected:raise ValueError("text-program census differs: "+str(counts))
    summary={"schema":"mdb-text-program-v1","job_id":os.environ["MRUN_JOB_ID"],"request_sha256":request["request_sha256"],"model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "counts":counts,"seal":seal,"transaction":verify_transaction(tx.root),"worker_seconds":time.monotonic()-started,"output_root":str(Path.cwd()),"mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary);event("complete");save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](output,summary,compression_level=1)


if __name__=="__main__":worker()
