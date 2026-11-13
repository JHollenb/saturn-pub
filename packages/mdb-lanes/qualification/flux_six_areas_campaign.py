"""Finite installed-runtime six-area study; all CUDA branches share one lease."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def worker():
    payload = Path.cwd()
    config, request = (json.loads(Path(os.environ[key]).read_text()) for key in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if config != request["config"] or any(sha(payload/n) != digest for n, digest in request["source_hashes"].items()):
        raise ValueError("request/config/source binding differs")
    helper = runpy.run_path(str(payload/"flux_six_areas.py"))
    if any(config[k] != v for k, v in helper["plan"]().items()):
        raise ValueError("finite six-area geometry differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    scratch = Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False); os.chdir(scratch)
    out = Path("six-areas-result"); out.mkdir()
    save(out/"plan.json", config); save(out/"request.json", request)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from saturn_pub import Session, Act
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator
    reference = runpy.run_path(str(payload/"flux_reentry_reference.py"))
    install = runpy.run_path(str(payload/"flux_joint_writes.py"))["install_image"]
    terminal = runpy.run_path(str(payload/"flux_scene_dependencies.py"))["terminal_from_forecast"]
    started = time.monotonic(); counts = dict(loads=0, encodes=0, noise_draws=0, native_transitions=0,
        acts=0, receipts=0, candidate_blocks=0, images=0, native_vae=0, fp32_vae=0)
    active = {"row": "setup"}; receipts=[]; timeline=[]
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():
        raise ValueError("admitted CUDA required")
    def cpu(value): return value.detach().float().cpu().numpy()
    def wall():
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise RuntimeError("finite six-area worker wall budget exhausted")
    def event(kind, **fields):
        value=dict(sequence=len(timeline), seconds=time.monotonic()-started, kind=kind, counts=dict(counts), **fields)
        timeline.append(value)
        with (out/"timeline.jsonl").open("a") as stream: stream.write(json.dumps(value, sort_keys=True)+"\n")
        print("MDB_SIX_AREAS "+json.dumps(value, sort_keys=True), flush=True)
    def array(name, value):
        a=cpu(value) if isinstance(value, torch.Tensor) else np.asarray(value)
        path=out/"arrays"/(name+".npy"); path.parent.mkdir(exist_ok=True)
        np.save(path, a, allow_pickle=False)
        return dict(path=path.relative_to(out).as_posix(), sha256=sha(path), shape=list(a.shape), dtype=str(a.dtype))
    def receipt(value):
        wall(); store.receipt(value); body=value.to_dict(); receipts.append(dict(row=active["row"], body=body))
        counts["receipts"]+=1
        counts["native_transitions" if body["operation"]=="continue" else "acts"]+=body.get("steps",1)
        if counts["native_transitions"]>1500 or counts["acts"]>20: raise RuntimeError("finite receipt budget exceeded")
    def nav(cut): return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=112,on_receipt=receipt)
    def capture(n):
        cut=n.session.capture(retain=False); store.save(cut); return cut

    event("inventory")
    root=Path(config["model_path"])
    inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["checkpoint_sha256"]: raise ValueError("checkpoint differs")
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["loads"]+=1; pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("deterministic distilled Klein required")
    if len(pipe.transformer.transformer_blocks)!=5 or len(pipe.transformer.single_transformer_blocks)!=20:
        raise ValueError("architecture differs")
    for c in (pipe.text_encoder,pipe.transformer,pipe.vae): c.eval().requires_grad_(False)
    source=dict(checkpoint_sha256=identity(inventory),checkpoint_files=inventory,
        implementations={inspect.getfile(k):sha(inspect.getfile(k)) for k in (type(pipe),type(pipe.transformer),Flux2KleinAdapter)},
        versions={k:importlib.metadata.version(k) for k in ("torch","diffusers","transformers","saturn-pub","mdb-runtime")})
    # Source-qualify the independent equations rather than silently accepting a
    # changed implementation with the same class names.
    from diffusers.models.transformers import transformer_flux2 as tf
    from diffusers.models import embeddings as emb
    from saturn_pub.adapters import flux2 as af
    for mod,key in ((tf,"transformer_module_sha256"),(emb,"embeddings_module_sha256"),(af,"adapter_module_sha256")):
        if sha(inspect.getfile(mod))!=reference["SOURCE_CONTRACT"][key]: raise ValueError("reference source ABI differs: "+key)
    pipe.text_encoder.to("cuda"); encodings={}; token_records={}
    for spec in config["histories"]:
        wall(); name=spec["id"]; observed={}
        formatted=pipe.tokenizer.apply_chat_template([dict(role="user",content=spec["prompt"])],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        tokens=pipe.tokenizer(formatted,return_tensors="pt",padding="max_length",truncation=True,max_length=512,
            return_offsets_mapping=True)
        def encoder_pre(module,args,kwargs):
            observed["ids"]=kwargs["input_ids"].detach().cpu(); observed["mask"]=kwargs["attention_mask"].detach().cpu()
        def encoder_post(module,args,result):
            observed["selected"]=torch.stack([result.hidden_states[i].detach() for i in (9,18,27)],dim=1).cpu()
        handles=[pipe.text_encoder.register_forward_pre_hook(encoder_pre,with_kwargs=True),
                 pipe.text_encoder.register_forward_hook(encoder_post)]
        try:
            with torch.inference_mode(): c,ids=pipe.encode_prompt(spec["prompt"],device=torch.device("cuda"),max_sequence_length=512)
        finally:
            for h in handles: h.remove()
        counts["encodes"]+=1
        if not torch.equal(tokens["input_ids"],observed["ids"]) or not torch.equal(tokens["attention_mask"],observed["mask"]):
            raise ValueError("actual encoder tokens differ from reported template")
        selected=observed["selected"]; rebuilt=selected.permute(0,2,1,3).reshape(c.shape)
        if not torch.equal(rebuilt,c.cpu()): raise ValueError("9/18/27 conditioning packing differs")
        offsets=tokens["offset_mapping"][0].tolist(); groups={}
        for word in ("red","blue","sphere","cube","left","right"):
            spans=[]; start=0
            while True:
                i=formatted.find(word,start)
                if i<0: break
                spans.append((i,i+len(word))); start=i+len(word)
            groups[word]=[j for j,(a,b) in enumerate(offsets) if b>a and any(a<e and b>s for s,e in spans)]
        groups["padding"]=[i for i,v in enumerate(observed["mask"][0].tolist()) if not v]
        encodings[name]=(c.detach().cpu(),ids.detach().cpu())
        token_records[name]=dict(formatted=formatted,input_ids=observed["ids"][0].tolist(),attention_mask=observed["mask"][0].tolist(),
            tokens=pipe.tokenizer.convert_ids_to_tokens(observed["ids"][0].tolist()),offsets=offsets,groups=groups,
            conditioning=array(name+".conditioning",c),layers=[9,18,27],packing_exact=True)
        event("encoded",history=name)
    save(out/"tokens.json",token_records)
    pipe.text_encoder=None; pipe.tokenizer=None; del observed,selected,rebuilt,c; gc.collect(); torch.cuda.empty_cache()
    pipe.transformer.to("cuda"); pipe.vae.to("cuda"); pipe.vae.bn.to("cpu")
    adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    source.update(model_identity=adapter.model_identity,execution=dict(adapter.execution)); save(out/"source.json",source)
    store=LocalStore(out/"store")
    with torch.inference_mode():
        latent,img_ids=pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,torch.bfloat16,
            torch.device("cuda"),torch.Generator(device="cuda").manual_seed(config["seed"]))
        counts["noise_draws"]+=1
        schedule=None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
        timesteps,steps=retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
    if steps!=4: raise ValueError("schedule differs")
    initial={}; native_navs={}
    for spec in config["histories"]:
        name=spec["id"]; c,ids=encodings[name]
        n=NativeCutNavigator(adapter.session(latent=latent.clone(),conditioning=c.to("cuda"),img_ids=img_ids.clone(),
            txt_ids=ids.to("cuda"),timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone()),max_steps=112,on_receipt=receipt)
        initial[name]=capture(n); native_navs[name]=n
    # Only native parameters and incoming state are accessible to the predictor.
    # Banks are recipe/hash bound, avoiding repeated gigabyte parameter exports.
    parameter_recipes={}
    def weights(owner,linears,norms=()):
        result={}
        for name in linears:
            module=owner.get_submodule(name)
            for field in ("weight","bias"):
                value=getattr(module,field,None)
                if value is not None: result[name+"."+field]=cpu(value)
        for name in norms: result[name]=cpu(owner.get_parameter(name))
        return result
    project_w=weights(pipe.transformer,reference["PROJECT_LINEAR_NAMES"])
    joint_w=weights(pipe.transformer.transformer_blocks[0],reference["JOINT_LINEAR_NAMES"],reference["JOINT_NORM_NAMES"])
    for bank,value in (("project",project_w),("joint0",joint_w)):
        parameter_recipes[bank]={k:reference["descriptor"](v) for k,v in value.items()}
    save(out/"parameter-recipes.json",parameter_recipes)
    guard={"released":False,"blocked":0,"target_calls":0}
    def block_before(module,args):
        guard["target_calls"]+=1
        if not guard["released"]:
            guard["blocked"]+=1; raise ValueError("denoiser target before seal")
    guards=[b.register_forward_pre_hook(block_before) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    tx=ForecastTransaction(out/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),
        source_identity=identity(request["source_hashes"]),frontier=dict(kind="architectural",initial_cuts={k:v.fingerprint for k,v in initial.items()},
            observed="actual native encoder conditioning, initial noise/IDs/schedule and parameters; no denoiser targets or final images"),
        execution_scope=config["hypotheses"]["scope"],budgets=dict(max_native_calls=1,max_wall_ms=config["max_wall_seconds"]*1000))
    predictions={}; prediction_records={}
    def forecast(_):
        for spec in config["histories"]:
            wall(); name=spec["id"]; state=initial[name].payload; tick=time.monotonic()
            p=reference["predict_project"](cpu(state["latent"]),cpu(state["conditioning"]),cpu(state["img_ids"]),
                cpu(state["txt_ids"]),cpu(state["timesteps"][0]),project_w,axes_dims_rope=list(pipe.transformer.config.axes_dims_rope))
            j=reference["predict_joint"](p["image"],p["text"],p["context"],joint_w,
                heads=pipe.transformer.transformer_blocks[0].attn.heads,retain_fields=True)
            predictions[name]=dict(project=p,joint=j)
            prediction_records[name]=dict(seconds=time.monotonic()-tick,
                project={k:array(name+".forecast.project."+k,p[k]) for k in ("image","text")},
                context={k:array(name+".forecast.context."+k,p["context"][k]) for k in ("temb","img_mod","txt_mod","single_mod")},
                rope=[array(name+".forecast.rope"+str(i),a) for i,a in enumerate(p["context"]["rope"])],
                joint={k:array(name+".forecast.joint."+k,j[k]) for k in ("image","text")},
                sampled={role:{k:array(name+".forecast."+role+"."+k,v) for k,v in j["sampled"][role].items() if k.startswith("full_")} for role in ("image","text")})
            event("forecast_complete",history=name,forecast_seconds=prediction_records[name]["seconds"])
        save(out/"forecasts.json",prediction_records)
        return {"predictions":dict(fields=prediction_records,hypotheses=config["hypotheses"],native_targets_before_seal=guard["target_calls"],
            encoder_predicted=False,parameter_recipes_sha256=sha(out/"parameter-recipes.json"))}
    tx.predict("own_state_math",forecast); seal=tx.seal(); event("sealed",seal=seal,guard=dict(guard)); guard["released"]=True
    del project_w,joint_w; gc.collect()
    observed={}; prefixes={}; post0={}; finals={}; rows=[]; trajectories=[]; witness_records=[]; witness_weights={}
    # Reusable sampled MLP inputs/returns and actual named parameter ports.
    # Native operands are observations; the later independent audit predicts
    # selected affine outputs and all sampled activations from those operands.
    def mlp_hooks(name,point):
        single=point["phase"]=="single"; index=point["index"]
        block=(pipe.transformer.single_transformer_blocks if single else pipe.transformer.transformer_blocks)[index]
        stem=("S" if single else "J")+str(index).zfill(2); key=f"{name}.t{point['logical_step']}.{stem}"
        t=512 if single else 0; count=initial[name].payload["latent"].shape[1]
        indices=np.unique(np.linspace(0,count-1,8).astype(int)); indices=(indices+t).tolist(); state={}; handles=[]
        linear_in=block.attn.to_qkv_mlp_proj if single else block.ff.linear_in
        linear_out=block.attn.to_out if single else block.ff.linear_out
        width=block.attn.inner_dim if single else 0
        def first(module,args,result):
            state["input"]=cpu(args[0][:,indices]); state["paired"]=cpu(result[:,indices,3*width:] if single else result[:,indices])
        def last(module,args,result):
            state["consumer_input"]=cpu(args[0][:,indices]); state["output"]=cpu(result[:,indices])
        handles=[linear_in.register_forward_hook(first),linear_out.register_forward_hook(last)]
        if stem not in witness_weights:
            hidden=(linear_in.weight.shape[0]-3*width)//2; neurons=list(range(16))
            selected=[3*width+i for i in neurons]+[3*width+hidden+i for i in neurons]
            witness_weights[stem]=dict(paired_weight=array(stem+".w_in",linear_in.weight[selected]),
                paired_bias=None if linear_in.bias is None else array(stem+".b_in",linear_in.bias[selected]),
                output_weight=array(stem+".w_out",linear_out.weight[:16]),
                output_bias=None if linear_out.bias is None else array(stem+".b_out",linear_out.bias[:16]),
                hidden=hidden,attention_width=width,neurons=neurons,output_channels=list(range(16)),
                full_in_descriptor=tensor_descriptor(linear_in.weight),full_out_descriptor=tensor_descriptor(linear_out.weight))
        def finish():
            for h in handles:h.remove()
            if set(state)!={"input","paired","consumer_input","output"}: raise ValueError("MLP witness incomplete")
            witness_records.append(dict(id=key,history=name,point=point,law=stem,indices=indices,
                fields={k:array(key+"."+k,v) for k,v in state.items()}))
        return finish

    def measured_route(name,candidate):
        raw=candidate["qkv"]["observed"]
        record={k:array(name+".qkv."+k,v) for k,v in raw.items()}
        route=helper["route_witness"]({k:cpu(v) for k,v in raw.items()},512,token_records[name]["groups"])
        return dict(qkv=record,route={k:array(name+".route."+k,v) for k,v in route.items()},
            dispatch=candidate["qkv"]["dispatch"])

    def finish_image(n,name,kind,**fields):
        final=capture(n); finals[name]=final
        path=out/(name+".png"); adapter.decode(n.session,height=512,width=512).save(path)
        counts["native_vae"]+=1; counts["images"]+=1
        row=dict(id=name,kind=kind,final=final.fingerprint,latent=array(name+".final",final.payload["latent"]),
            png=path.name,png_sha256=sha(path),**fields); rows.append(row); event("image_complete",row=name)
        return row

    def trace(n,name,*,natural=False):
        while n.point["logical_step"]<4:
            wall(); point=n.point; fields={}
            # Eight fixed spatial rows track receiving state through every block;
            # complete selected cuts preserve replayable closure.
            image=n.session.read("image") if point["phase"] in {"joint","single","readout"} else None
            if image is not None:
                indices=np.unique(np.linspace(0,image.shape[1]-1,8).astype(int)).tolist()
                fields.update(image=array(f"{name}.trace.{len(trajectories)}",image[:,indices]),
                    image_rms=float(torch.sqrt(torch.mean(image.float()**2)).item()))
            selected=natural and ((point["logical_step"]==0 and point["phase"]=="joint" and point["index"]==0)
                or (point["logical_step"]==1 and point["phase"]=="joint" and point["index"]==4)
                or (point["logical_step"]==3 and point["phase"]=="single" and point["index"]==19))
            closer=mlp_hooks(name,point) if selected else None
            try: n.step()
            finally:
                if closer:closer()
            trajectories.append(dict(history=name,point=point,**fields))
            if natural and point["phase"]=="project" and point["logical_step"]==0:
                cut=capture(n); observed.setdefault(name,{})["project"]=cut.fingerprint
                p=predictions[name]["project"]
                observed[name]["project_errors"]={k:helper["error"](p[k],cpu(cut.payload[k])) for k in ("image","text")}
                observed[name]["context_errors"]={k:helper["error"](p["context"][k],cpu(cut.payload["context"][k])) for k in ("temb","img_mod","txt_mod","single_mod")}
            if natural and point["phase"]=="joint" and point["index"]==0 and point["logical_step"]==0:
                cut=capture(n); post0[name]=cut; observed[name]["joint0"]=cut.fingerprint
                observed[name]["joint_errors"]={k:helper["error"](predictions[name]["joint"][k],cpu(cut.payload[k])) for k in ("image","text")}
            if natural and n.point["logical_step"]==1 and n.point["phase"]=="joint" and n.point["index"]==4:
                prefixes[name]=capture(n)
                # End this prefix before the selected block. Its counted direct
                # native observation is checked against the next adapter step.
                candidate=helper["port_candidate"](pipe.transformer.transformer_blocks[4],prefixes[name])
                counts["candidate_blocks"]+=1; candidates[name]=candidate
                observed[name]["joint4_entry"]=prefixes[name].fingerprint
                observed[name]["joint4_sources"]=measured_route(name,candidate)
                observed[name]["joint4_ports"]={k:array(name+".port."+k,v) for k,v in candidate["ports"].items()}
            if natural and point["logical_step"]==1 and point["phase"]=="joint" and point["index"]==4:
                cut=capture(n)
                if not torch.equal(cut.payload["image"],candidates[name]["image"]) or not torch.equal(cut.payload["text"],candidates[name]["text"]):
                    raise ValueError("direct QKV instrument differs from native block")
                observed[name]["joint4_post"]=cut.fingerprint
            if natural and point["phase"]=="scheduler":
                cut=capture(n); observed[name].setdefault("step_latents",[]).append(dict(step=n.point["logical_step"],cut=cut.fingerprint,
                    latent=array(name+".z"+str(n.point["logical_step"]),cut.payload["latent"])))
        return n

    candidates={}
    with tx:
        def execute(_):
            for spec in config["histories"]:
                name=spec["id"]; active["row"]=name
                trace(native_navs[name],name,natural=True); finish_image(native_navs[name],name,"natural")
            # Predicted complete I/T exit is a coherent numerical bridge; context
            # is the native receiving scaffold, whose forecast error is separate.
            active["row"]="math.bridge"; n=nav(post0["b0r0"]); actions=[]
            for slot in ("image","text"):
                value=torch.from_numpy(np.array(predictions["b0r0"]["joint"][slot],copy=True)).to("cuda",torch.bfloat16)
                before=capture(n); act=Act.replace(slot,value,name="math.bridge."+slot); r=n.apply(act); after=capture(n)
                if describe(after.payload)!=describe({**before.payload,slot:value}): raise ValueError("bridge undeclared closure")
                actions.append(dict(before=before.fingerprint,after=after.fingerprint,act=act.manifest(),receipt=r.to_dict()))
            trace(n,"math.bridge"); finish_image(n,"math.bridge","predicted-joint0-I/T-native-scaffold",actions=actions)
            for spec in config["interventions"]:
                name=spec["id"]; active["row"]=name; n=nav(prefixes["b0r0"]); donor=candidates[spec["source"]]
                value=helper["port_candidate"](pipe.transformer.transformer_blocks[4],prefixes["b0r0"],
                    donor=donor["qkv"]["observed"],fields=spec.get("fields",()),
                    attention=donor["ports"]["attention"] if spec.get("attention") else None,
                    mlp=donor["ports"]["mlp"] if spec.get("mlp") else None)
                counts["candidate_blocks"]+=1; n.step()
                action=install(n,value["image"],name=name,store=store)
                port_refs={k:array(name+".ports."+k,v) for k,v in value["ports"].items()}
                trace(n,name); finish_image(n,name,"image-only-causal-port",spec=spec,action=action,ports=port_refs,
                    text_and_future_conditioning="base recipient unchanged",equation_exact=value["equation_exact"])
            return dict(rows=rows,observed=observed,counts=dict(counts))
        tx.execute("native",execute)
    for h in guards:h.remove()
    save(out/"observed.json",observed); save(out/"witnesses.json",dict(weights=witness_weights,records=witness_records))
    save(out/"trajectories.json",trajectories); save(out/"receipts.json",receipts)
    # A second transaction starts from two observed native final latents.
    # These are response probes, never forecasts of the original final latent.
    z0=cpu(finals["b0r0"].payload["latent"]); z1=cpu(finals["b1r0"].payload["latent"])
    decoder_inputs={}; decoder_records={}; raw_pixels={}
    for alpha in config["decoder_alphas"]:
        key=f"{alpha:+.2f}"; continuous=z0+np.float32(alpha)*(z1-z0); rounded=reference["bf16"](continuous)
        decoder_inputs[key]=(continuous,rounded)
        decoder_records[key]=dict(alpha=alpha,continuous=array("decode."+key+".continuous",continuous),
            rounded=array("decode."+key+".rounded",rounded),rounding=helper["error"](rounded,continuous))
    dtx=ForecastTransaction(out/"decoder-transaction",model_identity=adapter.model_identity,input_identity=identity(decoder_records),
        source_identity=identity(request["source_hashes"]),frontier=dict(kind="architectural",observed="two native final latents; original RGB available; new doses/FP32 decoder unexecuted"),
        budgets=dict(max_native_calls=1,max_wall_ms=120000))
    dtx.predict("doses",lambda _:dict(predictions=dict(inputs=decoder_records,hypothesis=config["hypotheses"]["decoder"])))
    dseal=dtx.seal(); event("decoder_sealed",seal=dseal)
    original_decode=pipe.vae.decode
    def decode_observe(*args,**kwargs):
        result=original_decode(*args,**kwargs); value=result[0] if isinstance(result,tuple) else result.sample
        raw_pixels[active["row"]]=array(active["row"]+".raw",value)
        return result
    pipe.vae.decode=decode_observe
    with dtx:
        def decode_panel(_):
            for key,(_,rounded) in decoder_inputs.items():
                wall(); name="decode.native."+key; active["row"]=name
                session=terminal(adapter,finals["b0r0"],torch.from_numpy(rounded.copy()).to("cuda",torch.bfloat16),parent=finals["b0r0"].fingerprint)
                derived=session.capture(retain=False); store.save(derived)
                path=out/(name+".png"); adapter.decode(session,height=512,width=512).save(path)
                counts["native_vae"]+=1; counts["images"]+=1
                rows.append(dict(id=name,kind="derived-terminal-native-decoder",terminal=derived.fingerprint,
                    lineage="explicit non-native terminal initialization from dose state",png=path.name,png_sha256=sha(path),alpha=float(key)))
            # Widen the same BF16 parameter values, then hold this FP32 consumer
            # fixed while changing only input rounding. This is a separate
            # arithmetic lane, not an independent mathematical VAE predictor.
            pipe.vae.decode=original_decode; pipe.vae.to(device="cuda",dtype=torch.float32)
            for key,(continuous,rounded) in decoder_inputs.items():
                for mode,z in (("continuous",continuous),("rounded",rounded)):
                    wall(); name="decode.fp32."+key+"."+mode; active["row"]=name
                    with torch.inference_mode():
                        packed=torch.from_numpy(z.copy()).to("cuda",torch.float32)
                        unpacked=pipe._unpack_latents_with_ids(packed,img_ids,512//(pipe.vae_scale_factor*2),512//(pipe.vae_scale_factor*2))
                        mean=pipe.vae.bn.running_mean.view(1,-1,1,1)
                        std=(pipe.vae.bn.running_var.view(1,-1,1,1)+pipe.vae.config.batch_norm_eps).sqrt()
                        pixels=pipe.vae.decode(pipe._unpatchify_latents(unpacked*std+mean),return_dict=False)[0]
                    raw_pixels[name]=array(name+".raw",pixels)
                    path=out/(name+".png"); pipe.image_processor.postprocess(pixels,output_type="pil")[0].save(path)
                    counts["fp32_vae"]+=1; counts["images"]+=1
                    rows.append(dict(id=name,kind="FP32-consumer-rounding-diagnostic",alpha=float(key),mode=mode,png=path.name,png_sha256=sha(path)))
            return dict(raw_pixels=raw_pixels,rows=rows,counts=dict(counts))
        dtx.execute("decode",decode_panel)
    save(out/"decoder-inputs.json",decoder_records); save(out/"raw-pixels.json",raw_pixels); save(out/"rows.json",rows)
    if next(r["png_sha256"] for r in rows if r["id"]=="route.sham")!=next(r["png_sha256"] for r in rows if r["id"]=="b0r0"):
        raise ValueError("same-parent QKV sham image differs")
    if next(r["png_sha256"] for r in rows if r["id"]=="decode.native.+0.00")!=next(r["png_sha256"] for r in rows if r["id"]=="b0r0"):
        raise ValueError("native terminal decode zero-dose control differs")
    summary=dict(schema="mdb-flux-six-areas-v1",job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],
        model_identity=adapter.model_identity,counts=counts,guard=guard,seal=seal,decoder_seal=dseal,
        transaction=verify_transaction(tx.root),decoder_transaction=verify_transaction(dtx.root),
        worker_seconds=time.monotonic()-started,output_root=str(scratch),mechanics_status="pending-independent-audit",terminal_status="unassessed",
        sham_exact=True,decoder_zero_exact=True,scope=config["hypotheses"]["scope"])
    save(out/"summary.json",summary); event("complete")
    save(out/"files.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload/"flux_family_custody.py"))["emit"](out,summary,compression_level=1)


if __name__=="__main__":
    worker_root=Path.cwd()
    try:
        worker()
    except Exception as exc:
        output=Path("six-areas-result")
        if output.is_dir() and not (output/"files.json").exists():
            request=json.loads((output/"request.json").read_text())
            failure=dict(schema="mdb-six-areas-failure-v1",job_id=os.environ.get("MRUN_JOB_ID"),
                request_sha256=request["request_sha256"],error=f"{type(exc).__name__}: {exc}",mechanics_status="errored")
            save(output/"failure.json",failure)
            save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
            runpy.run_path(str(worker_root/"flux_family_custody.py"))["emit"](output,failure,compression_level=1)
        raise
