"""Native correction factorial with continuing-program and collateral contrasts."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/"flux_predicted_future_campaign.py"))
    save,sha,identity=[u[k] for k in ("save","sha","identity")]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    if config!=request["config"] or any(sha(p)!=h for p,h in request["source_hashes"].items()):raise ValueError("staged sources differ")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path("interaction-corrections-result");output.mkdir();save(output/"plan.json",config);save(output/"request.json",request)
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
    from mdb.tensor_archive import load_tensor_archive,write_tensor_archive
    restore=runpy.run_path(str(payload/"flux_scene_dependencies.py"))["restore_slots"]
    capture=runpy.run_path(str(payload/"flux_single_contraction.py"))["capture_candidate"]
    algebra=runpy.run_path(str(payload/"flux_interaction_corrections.py"))["correction_fields"]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={"loads":0,"encodes":0,"native_transitions":0,"acts":0,"receipts":0,"source_blocks":0,"validation_blocks":0,"projections":0,"images":0}
    receipts=[];rows=[]
    def event(kind,**extra):
        row={"kind":kind,"seconds":time.monotonic()-started,"counts":dict(counts),**extra}
        with (output/"timeline.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
        print("MDB_INTERACTION "+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        row=r.to_dict();receipts.append(row);counts["receipts"]+=1;counts["native_transitions" if row["operation"]=="continue" else "acts"]+=1
        if counts["native_transitions"]>213 or counts["acts"]>15:raise RuntimeError("finite census exceeded")
        if time.monotonic()-started>config["max_wall_seconds"]:raise RuntimeError("finite wall exceeded")
    def array(name,value):return write_tensor_archive(output/"arrays"/(name+".npy"),value,root=output)
    def cpu(value):return value.detach().float().cpu().numpy()
    model=Path(config["model_path"])
    inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob("*")) if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["checkpoint_sha256"]:raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts["loads"]+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    store=LocalStore(output/"store");inputs={}
    for role,fingerprint in config["input_cuts"].items():
        cut=LocalStore(Path("input-data/store")).load(fingerprint,device="cuda")
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):raise ValueError("numerical profile differs")
        if cut.execution_point["logical_step"]!=3 or cut.execution_point["phase"]!="single" or cut.execution_point["index"]!=7:raise ValueError("source clock differs")
        inputs[role]=cut;store.save(cut)
    base=inputs["base"];entries={"base":base,"additive":inputs["additive"]};entry_actions={"base":[],"additive":[]}
    for role in ("color","shape","joint"):
        nav=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=1,on_receipt=on_receipt)
        actions=restore(nav,{"text":inputs[role].payload["text"]},label="entry."+role,store=store)
        entries[role]=nav.capture();entry_actions[role]=actions;store.save(entries[role])
    for cut in entries.values():
        if describe(cut.payload)!=describe({**base.payload,"text":cut.payload["text"]}):raise ValueError("entries differ outsideT")
    fields={}
    for role,meta in config["prediction_fields"].items():
        if role=="additive":fields[role]=load_tensor_archive(Path("input-data"),meta)
        else:
            file=payload/meta["path"]
            if sha(file)!=meta["sha256"]:raise ValueError("saved equation field differs")
            bits=np.load(file,allow_pickle=False)
            if bits.dtype!=np.uint16 or list(bits.shape)!=meta["shape"]:raise ValueError("BF16 field ABI differs")
            fields[role]=(bits.astype(np.uint32)<<16).view(np.float32)
    candidates,terms=algebra(fields)
    binding={"entries":{k:v.fingerprint for k,v in entries.items()},"entry_actions":entry_actions,
        "field_sources":config["prediction_fields"],"prior_diagnostic":config["prior_diagnostic"],"conditions":config["conditions"]}
    save(output/"binding.json",binding)
    block=pipe.transformer.single_transformer_blocks[7];release={"native":False}
    def guard(module,args):
        if not release["native"]:raise ValueError("block7 entered before correction seal")
    handle=block.register_forward_pre_hook(guard)
    tx=ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
        frontier={"kind":"architectural","binding":binding},execution_scope="New current-write correction and continuing-program factorial; source-assisted equation operands and native M/text/projection/suffix",
        budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    field_records={};native={};post={};intermediate={}
    with tx:
        tx.frontier("binding",binding)
        def forecast(_):
            field_records.update({k:array("candidate."+k,torch.from_numpy(v).to(torch.bfloat16)) for k,v in candidates.items()})
            save(output/"forecasts.json",field_records)
            return {"predictions":{"fields":field_records,"hypotheses":config["hypotheses"],"branches":config["branches"]}}
        tx.predict("correction_panel",forecast);seal=tx.seal();release["native"]=True;handle.remove();event("sealed",seal=seal)
        def execute(_):
            native_records={}
            for role,cut in entries.items():
                native[role]=capture(block,cut,single_index=7);counts["source_blocks"]+=1
                h,t=native[role]["attention_width"],native[role]["text_rows"]
                concat=native[role]["ports"]["original_concat"]
                native_records[role]=array("native."+role+".A",concat[:,t:,:h])
                if not torch.equal(concat[:,t:,h:],native["base"]["ports"]["original_concat"][:,t:,h:]):raise ValueError("imageM changes despite fixedimage")
            save(output/"native-fields.json",native_records)
            fixed=native["additive"];h,t=fixed["attention_width"],fixed["text_rows"]
            def intermediate_record(name,nav):
                values={slot:nav.session.read(slot) for slot in ("image","text")}
                record={"point":nav.point,"slots":{slot:{"descriptor":describe(value),"witness_flat":cpu(value).reshape(-1)[::max(1,value.numel()//64)][:64].tolist()} for slot,value in values.items()}}
                if name.startswith("native."):
                    intermediate[name]={slot:cpu(v) for slot,v in values.items()}
                    for slot,value in values.items():record["slots"][slot]["array"]=array(name+".single11."+slot,value)
                else:
                    record["comparisons"]={ref:{slot:{"l2":float(np.linalg.norm((cpu(v).astype(np.float64)-saved[slot]).reshape(-1))),"signed_mean":float(np.mean(cpu(v).astype(np.float64)-saved[slot]))} for slot,v in values.items()} for ref,saved in intermediate.items()}
                return record
            def finish(name,nav,actions,initial=()):
                path=list(initial);middle=None
                while nav.point["logical_step"]<4:
                    if nav.point["phase"]=="single" and nav.point["index"]==11:middle=intermediate_record(name,nav)
                    path.append(nav.step().to_dict())
                final=nav.capture();store.save(final);png=output/(name+".png");adapter.decode(nav.session,height=512,width=512).save(png);counts["images"]+=1
                if name in config["historical_pngs"] and sha(png)!=config["historical_pngs"][name]:raise ValueError("historical PNG differs: "+name)
                rows.append({"id":name,"actions":actions,"path":path,"final_cut":final.fingerprint,"latent":array(name+".latent",final.payload["latent"]),
                    "png":png.name,"png_sha256":sha(png),"intermediate":middle})
                event("image_complete",branch=name)
            for role in ("base","additive","joint"):
                nav=NativeCutNavigator(Session.from_cut(adapter,entries[role]),max_steps=15,on_receipt=on_receipt)
                first=nav.step().to_dict();post[role]=nav.capture();store.save(post[role])
                if not torch.equal(post[role].payload["image"],native[role]["image"]) or not torch.equal(post[role].payload["text"],native[role]["text"]):raise ValueError("direct/native block mismatch")
                finish("native."+role,nav,[],[first])
            images={}
            for key,value in candidates.items():
                concat=fixed["ports"]["original_concat"].clone();concat[:,t:,:h]=torch.from_numpy(value).to("cuda",torch.bfloat16)
                with torch.inference_mode():images[key]=(fixed["ports"]["residual"]+fixed["ports"]["gate"]*block.attn.to_out(concat))[:,t:]
                counts["projections"]+=1
            validations={}
            for program in ("additive","joint"):
                for field in ("r0c0","r1c1"):
                    concat=fixed["ports"]["original_concat"].clone();concat[:,t:,:h]=torch.from_numpy(candidates[field]).to("cuda",torch.bfloat16)
                    checked=capture(block,entries[program],single_index=7,selection="A",source={"concat":concat,"parent":seal["seal_sha256"],"text_rows":t,"attention_width":h})
                    counts["validation_blocks"]+=1
                    validations[program+"."+field]={"image_exact":torch.equal(checked["image"],images[field]),"text_exact_recipient":torch.equal(checked["text"],native[program]["text"]),
                        "scope":"Actual full-block A-only substitution against cached image projection; source identity is the forecast seal, not a claimed native donor cut."}
            save(output/"projection-validations.json",validations)
            # Only six unique endpoint projections are needed for ten arms.
            for branch in config["branches"]:
                parent=post[branch["program"]];nav=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=14,on_receipt=on_receipt)
                if branch["kind"]=="sham":value=parent.payload["image"]
                else:value=images[branch["field"]]
                actions=restore(nav,{"image":value},label=branch["id"],store=store)
                finish(branch["id"],nav,actions)
            return {"counts":dict(counts),"images":len(rows)}
        tx.execute("native_corrections",execute)
    save(output/"rows.json",rows);save(output/"receipts.json",receipts)
    expected={"loads":1,"encodes":0,"native_transitions":213,"acts":15,"receipts":228,"source_blocks":5,"validation_blocks":4,"projections":6,"images":15}
    if counts!=expected:raise ValueError("census differs: "+str(counts))
    summary={"schema":"mdb-interaction-corrections-v1","job_id":os.environ["MRUN_JOB_ID"],"request_sha256":request["request_sha256"],"counts":counts,
        "model_identity":adapter.model_identity,"execution":dict(adapter.execution),"worker_seconds":time.monotonic()-started,"seal":seal,"transaction":verify_transaction(tx.root),
        "output_root":str(Path.cwd()),"mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary);event("complete");save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](output,summary,compression_level=1)


if __name__=="__main__":worker()
