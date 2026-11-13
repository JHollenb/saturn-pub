"""Separate a current attention write from a continuing contextual eye program."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/"flux_predicted_future_campaign.py"))
    save,sha,identity=(u[k] for k in ("save","sha","identity"))
    config,request=(json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    if config!=request["config"] or any(sha(p)!=h for p,h in request["source_hashes"].items()):raise ValueError("source binding differs")
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path("eye-program-result");output.mkdir(exist_ok=False);save(output/"plan.json",config);save(output/"request.json",request)
    for group,spec in config["input_groups"].items():
        root=Path(spec["root"]).resolve(strict=True)
        for name,meta in spec["files"].items():
            src=root/name;dst=Path("input-data")/group/name
            if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=meta["bytes"] or sha(src)!=meta["sha256"]:raise ValueError("source custody differs")
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open("xb") as stream:stream.write(src.read_bytes())
    save(output/"input-custody.json",config["input_groups"])
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.tensor_archive import write_tensor_archive
    helper=runpy.run_path(str(payload/"flux_scene_dependencies.py"));legacy=runpy.run_path(str(payload/"flux_single_contraction.py"))
    mlp=runpy.run_path(str(payload/"flux_mlp_replacement.py"));attention=runpy.run_path(str(payload/"flux_attention_reference.py"))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={"loads":0,"encodes":0,"tokenizations":0,"native_transitions":0,"acts":0,"receipts":0,"source_blocks":0,"projection_calls":0,"images":0}
    emissions=[];sequence=0
    def event(kind,**fields):
        nonlocal sequence
        row={"kind":kind,"sequence":sequence,"seconds":time.monotonic()-started,"counts":dict(counts),**fields};sequence+=1
        with (output/"timeline.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
        print("MDB_EYE_PROGRAM "+json.dumps(row,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>config["max_wall_seconds"]:raise RuntimeError("finite eye-program budget exhausted")
    def receipt(r):
        row=r.to_dict();counts["receipts"]+=1;counts["native_transitions" if row["operation"]=="continue" else "acts"]+=1;emissions.append(row)
        if counts["native_transitions"]>196 or counts["acts"]>13:raise RuntimeError("eye-program work limit exceeded")
    def array(name,x):return write_tensor_archive(output/"arrays"/(name+".npy"),x,root=output)
    def cpu(x):return x.detach().float().cpu().numpy()
    root=Path(config["model_path"])
    inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["checkpoint_sha256"]:raise ValueError("checkpoint differs")
    event("loading")
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts["loads"]+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu");adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    store=LocalStore(output/"store");cuts={}
    for role,spec in config["input_cuts"].items():
        cuts[role]=LocalStore(Path("input-data")/spec["group"]/"store").load(spec["cut"],device="cuda")
        if cuts[role].model_identity!=adapter.model_identity or dict(cuts[role].execution)!=dict(adapter.execution):raise ValueError("saved numerical profile differs")
        store.save(cuts[role])
    base,program=cuts["post.base"],cuts["post.program"]
    if any(describe(v)!=describe(program.payload[k]) for k,v in base.payload.items() if k not in {"image","text"}):raise ValueError("post7 sources differ outside I/T")
    prepared={};prep_rows={}
    for role in ("color","shape"):
        n=NativeCutNavigator(Session.from_cut(adapter,cuts["pre.base"]),max_steps=1,on_receipt=receipt)
        acts=helper["restore_slots"](n,{"text":cuts["pre."+role].payload["text"]},label="prepare."+role,store=store)
        n.step();prepared[role]=n.capture();store.save(prepared[role]);prep_rows[role]={"acts":acts,"post_cut":prepared[role].fingerprint}
    # Exact native prompt formatting supplies lexical coordinates; no encoder call.
    tokenizer=AutoTokenizer.from_pretrained(str(root/"tokenizer"),local_files_only=True)
    tokens={}
    for role in ("base","joint"):
        text=tokenizer.apply_chat_template([{"role":"user","content":config["prompts"][role]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens[role]=tokenizer(text,return_tensors="pt",padding="max_length",truncation=True,max_length=512);counts["tokenizations"]+=1
    changed=torch.nonzero(tokens["base"]["input_ids"][0]!=tokens["joint"]["input_ids"][0]).flatten().tolist()
    active=torch.nonzero((tokens["base"]["attention_mask"][0]|tokens["joint"]["attention_mask"][0]).bool()).flatten().tolist()
    padding=[j for j in range(512) if j not in set(active)]
    delta=program.payload["text"].float()-base.payload["text"].float()
    top64=torch.argsort(torch.sum(delta*delta,dim=-1)[0],descending=True,stable=True)[:64].tolist()
    additive=(prepared["color"].payload["text"].float()+prepared["shape"].payload["text"].float()-base.payload["text"].float()).to(torch.bfloat16)
    replacements={"native.base":{},"full.program":None,"image.only":{"image":program.payload["image"]},
        "text.only":{"text":program.payload["text"]},"color.only":{"text":prepared["color"].payload["text"]},
        "shape.only":{"text":prepared["shape"].payload["text"]},"additive":{"text":additive}}
    for name,indices in (("lexical",changed),("active",active),("padding",padding),("top64",top64)):
        value=base.payload["text"];value[:,indices]=program.payload["text"][:,indices]
        replacements["text."+name]={"text":value}
    replacements["text.roll"]={"text":torch.roll(program.payload["text"],256,dims=1)}
    save(output/"supports.json",{"changed_rows":changed,"active_rows":active,"padding_rows":padding,"top64_rows":top64,
        "changed_tokens":[{"row":j,"base":tokenizer.convert_ids_to_tokens(int(tokens["base"]["input_ids"][0,j])),"joint":tokenizer.convert_ids_to_tokens(int(tokens["joint"]["input_ids"][0,j]))} for j in changed],
        "scope":"Lexical coordinates derived with exact native chat template; internal working T is contextualized. Energy ranking is not a semantic label."})
    save(output/"preparation.json",prep_rows)
    # Prepare only the additive input before prediction; its Act is reused later.
    add=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=15,on_receipt=receipt)
    add_acts=helper["restore_slots"](add,{"text":additive},label="additive",store=store);addcut=add.capture();store.save(addcut)
    block=pipe.transformer.single_transformer_blocks[7];fused=block.attn.to_qkv_mlp_proj
    qkv_weight=fused.weight[:9216];mlp_weight=fused.weight[9216:]
    qkv_bias=None if fused.bias is None else fused.bias[:9216];mlp_bias=None if fused.bias is None else fused.bias[9216:]
    binding={"parent":addcut.fingerprint,"text":array("add.text",addcut.payload["text"]),"image":array("add.image",addcut.payload["image"]),
        "mod":array("add.mod",addcut.payload["context"]["single_mod"]),"rope_cos":array("rope.cos",addcut.payload["context"]["rope"][0]),"rope_sin":array("rope.sin",addcut.payload["context"]["rope"][1]),
        "qkv_weight":array("qkv.weight",qkv_weight),"mlp_weight":array("mlp.weight",mlp_weight),
        "qkv_bias":None if qkv_bias is None else array("qkv.bias",qkv_bias),"mlp_bias":None if mlp_bias is None else array("mlp.bias",mlp_bias),
        "norm_q":array("norm.q",block.attn.norm_q.weight),"norm_k":array("norm.k",block.attn.norm_k.weight),
        "layernorm_eps":block.norm.eps,"qk_eps":block.attn.norm_q.eps,"heads":block.attn.heads,
        "scope":"New additive contextual program; no joint T used in additive rule, no current-block A/M supplied to predictor. Observed entry state and actual parameters/position operands."}
    save(output/"binding.json",binding)
    release={"native":False}
    def guard(module,args):
        if not release["native"]:raise ValueError("block7 entered before additive forecast seal")
    handle=block.register_forward_pre_hook(guard)
    tx=ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
        frontier={"kind":"architectural","binding":binding},execution_scope="Full image A/M of new additive T forecast before block7; native text/projection/suffix consumers retained",budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    predicted={};rows=[];add_post=None
    with tx:
        tx.frontier("binding",binding)
        def forecast(_):
            raw=torch.cat([addcut.payload["text"],addcut.payload["image"]],dim=1)
            a=attention["predict_attention_field"](cpu(raw),cpu(addcut.payload["context"]["single_mod"]),cpu(qkv_weight),None if qkv_bias is None else cpu(qkv_bias),
                text_rows=512,heads=block.attn.heads,norm_q_weight=cpu(block.attn.norm_q.weight),norm_k_weight=cpu(block.attn.norm_k.weight),layernorm_eps=block.norm.eps,qk_eps=block.attn.norm_q.eps,
                rope_cos=cpu(addcut.payload["context"]["rope"][0]),rope_sin=cpu(addcut.payload["context"]["rope"][1]))
            m=mlp["predict_mlp_field"](cpu(addcut.payload["image"]),cpu(addcut.payload["context"]["single_mod"]),cpu(mlp_weight),None if mlp_bias is None else cpu(mlp_bias),eps=block.norm.eps)
            for key,value in (("A",a["attention"]),("M",m["activation"])):
                predicted[key]=torch.from_numpy(np.array(value,copy=True)).to("cuda",torch.bfloat16)
            save(output/"forecasts.json",{k:array("predicted."+k,v) for k,v in predicted.items()})
            return {"predictions":{"fields":json.loads((output/"forecasts.json").read_text()),"hypotheses":config["hypotheses"]}}
        tx.predict("additive_fields",forecast);seal=tx.seal();release["native"]=True;handle.remove();event("sealed",seal=seal)
        def execute(_):
            nonlocal add_post
            source_native=legacy["capture_candidate"](block,addcut,single_index=7);counts["source_blocks"]+=1
            source_base=legacy["capture_candidate"](block,base,single_index=7);counts["source_blocks"]+=1
            t,h=source_native["text_rows"],source_native["attention_width"]
            native_fields={"A":source_native["ports"]["original_concat"][:,t:,:h],"M":source_native["ports"]["original_concat"][:,t:,h:]}
            save(output/"field-observations.json",{"fields":{k:array("measured."+k,v) for k,v in native_fields.items()},
                "image_M_exact_base_despite_text_change":torch.equal(native_fields["M"],source_base["ports"]["original_concat"][:,t:,h:]),
                "image_A_exact_base":torch.equal(native_fields["A"],source_base["ports"]["original_concat"][:,t:,:h])})
            concat=source_native["ports"]["original_concat"].clone();concat[:,t:,:h]=predicted["A"];concat[:,t:,h:]=predicted["M"]
            with torch.inference_mode():
                projection=block.attn.to_out(concat);pred_image=(source_native["ports"]["residual"]+source_native["ports"]["gate"]*projection)[:,t:]
            counts["projection_calls"]+=1
            def finish(name,nav,acts,initial=()):
                path=list(initial)
                while nav.point["logical_step"]<4:wall();path.append(nav.step().to_dict())
                final=nav.capture();store.save(final);file=output/(name+".png");adapter.decode(nav.session,height=512,width=512).save(file);counts["images"]+=1
                if name in config["historical_pngs"] and sha(file)!=config["historical_pngs"][name]:raise ValueError("historical program PNG differs: "+name)
                row={"id":name,"actions":acts,"path":path,"final_cut":final.fingerprint,"latent":array(name+".latent",final.payload["latent"]),"png":file.name,"png_sha256":sha(file)};rows.append(row);event("image_complete",branch=name)
            for name,replacement in replacements.items():
                if name=="additive":
                    nav=add;acts=add_acts;entry_receipt=nav.step().to_dict();add_post=nav.capture();store.save(add_post)
                    if not torch.equal(add_post.payload["image"],source_native["image"]):raise ValueError("additive direct/native prefix differs")
                else:
                    nav=NativeCutNavigator(Session.from_cut(adapter,program if replacement is None else base),max_steps=15,on_receipt=receipt)
                    acts=helper["restore_slots"](nav,replacement or {},label=name,store=store)
                finish(name,nav,acts,[entry_receipt] if name=="additive" else ())
            n=NativeCutNavigator(Session.from_cut(adapter,add_post),max_steps=14,on_receipt=receipt)
            acts=helper["restore_slots"](n,{"image":pred_image},label="additive.predictedAM",store=store);finish("additive.predictedAM",n,acts)
            return {"counts":dict(counts),"rows":len(rows)}
        tx.execute("native_futures",execute)
    save(output/"rows.json",rows);save(output/"receipts.json",emissions)
    if counts!={"loads":1,"encodes":0,"tokenizations":2,"native_transitions":196,"acts":13,"receipts":209,"source_blocks":2,"projection_calls":1,"images":13}:raise ValueError("eye-program census differs: "+str(counts))
    summary={"schema":"mdb-eye-program-v1","job_id":os.environ["MRUN_JOB_ID"],"request_sha256":request["request_sha256"],"counts":counts,"model_identity":adapter.model_identity,
        "worker_seconds":time.monotonic()-started,"seal":seal,"transaction":verify_transaction(tx.root),"output_root":str(Path.cwd()),"mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary);event("complete");save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](output,summary,compression_level=1)


if __name__=="__main__":worker()
