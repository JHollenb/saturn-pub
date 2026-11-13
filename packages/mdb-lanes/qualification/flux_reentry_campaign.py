"""Predict a remaining denoising step, full next projection and joint0 reentry.

The complete forecast is sealed before any selected current-step or next-step
native target. Native clock scaffolds never replace the predicted latent,
context or working streams: the opt-in public closure Act admits all four.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import shutil
import time
from functools import lru_cache
from pathlib import Path

CONTEXT = {"id":"p1-s97","prompt_id":"p1","seed":97,
    "prompt":"A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BLOCK_INDICES = list(range(6,20))
BRANCHES = [{"id":"native","mode":"native"},{"id":"sham","mode":"sham"},
    {"id":"latent.predicted","mode":"latent"},{"id":"project.predicted","mode":"project"},
    {"id":"joint.predicted","mode":"joint"}]
ORDER = [row["id"] for row in BRANCHES]
COUNTS = {"model_loads":1,"prompt_encodings":1,"noise_draws":1,"adapter_steps":219,"acts":4,
    "candidate_calls":0,"receipt_emissions":223,"images":5,"vae_calls":5,"physical_batch":1}
SAMPLE_INDICES = [0,31,511,512,543,767,1023,1535]
IMAGE_SAMPLE_INDICES = [0,31,32,255,511,512,767,1023]
STEP_SOURCE_SHA256 = "e3f0eac65d4d26bf49e0f562f6b70df38a61ceb0deb1a2001deb77b7986dde66"
OBSERVATION_PLAN = {"reference_rows":["native"],"boundaries_per_branch":27,
    "slots":["image","text","latent","noise_prediction"],"witness_count":64,
    "reference_storage":"lossless-bf16-bits","candidate_storage":"descriptor-and-fixed-coordinate-witness",
    "scope":"Actual common step3/joint1 plus26 native exits; native reference arrays retained. Candidate intermediate full norms are worker observations with64 fixed witnesses, not independently auditable full arrays."}
PROJECT_PREFIXES = ("x_embedder.","context_embedder.","time_guidance_embed.timestep_embedder.",
    "double_stream_modulation_img.","double_stream_modulation_txt.","single_stream_modulation.")
PROJECT_TARGETS = ("x_embedder","context_embedder","time_guidance_embed","double_stream_modulation_img",
    "double_stream_modulation_txt","single_stream_modulation")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _step():
    path = Path(__file__).with_name("flux_step_boundary_campaign.py")
    if _sha(path) != STEP_SOURCE_SHA256:
        raise ValueError("qualified prior step-boundary source bytes differ")
    return runpy.run_path(str(path))


def _util():
    return _step()["_util"]()


def contract(config=None):
    if config is not None and config.get("block_indices",BLOCK_INDICES) != BLOCK_INDICES:
        raise ValueError("reentry supports only the declared14-block remaining step")
    return json.loads(json.dumps({"context":CONTEXT,"block_indices":BLOCK_INDICES,"branches":BRANCHES,
        "order":ORDER,"counts":COUNTS,"prefix_steps":68,"reference_steps":18,"bridge_steps":3,"suffix_steps":26,
        "panel_counts":{"adapter_steps":151,"acts":4,"receipt_emissions":155,"candidate_calls":0,"images":5,"vae_calls":5},
        "target_calls":{**{str(index):6 for index in BLOCK_INDICES},"norm_out":6,"proj_out":6,
            **{name:2 for name in PROJECT_TARGETS},"pos_embed":4,"joint0":3},
        "observation_plan":OBSERVATION_PLAN}))


def validate_config(config):
    fixed = {"context":CONTEXT,"block_indices":BLOCK_INDICES,"branches":BRANCHES,"branch_execution_order":ORDER,
        "selector":{"logical_step":2,"phase":"single","index":6},"height":512,"width":512,"steps":4,
        "sample_indices":SAMPLE_INDICES,"image_sample_indices":IMAGE_SAMPLE_INDICES,"query_chunk":64,
        "projection_chunk":256,"neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits",
        "archive_integer_ids":"native-int64","cpu_threads":4,"transaction_max_bytes":768<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":8<<30,"max_member_bytes":1<<30},"observation_plan":OBSERVATION_PLAN}
    if any(config.get(key) != value for key,value in fixed.items()):
        raise ValueError("reentry frozen scientific/closure geometry differs")
    for key in ("adapter_steps","acts","candidate_calls","images","vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("reentry finite physical census differs")
    if (type(config.get("cpu_threads")) is not int or type(config.get("max_wall_seconds")) is not int
            or not 1 <= config["max_wall_seconds"] <= 900):
        raise ValueError("reentry execution budget differs")
    closure_sha = config.get("expected_closure_module_sha256")
    if not isinstance(closure_sha,str) or len(closure_sha) != 64 or any(char not in "0123456789abcdef" for char in closure_sha):
        raise ValueError("reentry requires an explicit public closure module source pin")


def verify_source_abi(*,boundary,reentry,closure_type,expected_closure_module_sha256):
    from diffusers.models import embeddings
    from diffusers.models.normalization import AdaLayerNormContinuous
    from diffusers.models.transformers import transformer_flux2
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    base = _step()["verify_native_source_contract"](boundary=boundary,norm_type=AdaLayerNormContinuous,
        scheduler_type=FlowMatchEulerDiscreteScheduler,adapter_type=Flux2KleinAdapter)
    pins = reentry["SOURCE_CONTRACT"]
    actual = {"transformer_module_sha256":_sha(inspect.getfile(transformer_flux2)),
        "embeddings_module_sha256":_sha(inspect.getfile(embeddings)),
        "adapter_module_sha256":_sha(inspect.getfile(Flux2KleinAdapter))}
    classes = {name:getattr(transformer_flux2,name) for name in ("Flux2TransformerBlock","Flux2AttnProcessor","Flux2PosEmbed")}
    classes.update({name:getattr(embeddings,name) for name in ("get_timestep_embedding","get_1d_rotary_pos_embed","apply_rotary_emb")})
    for name,value in classes.items():
        actual[name+".source_sha256"] = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
    if (any(pins[key] != value for key,value in actual.items())
            or not issubclass(closure_type,Flux2KleinAdapter)
            or _sha(inspect.getfile(closure_type)) != expected_closure_module_sha256
            or closure_type.advance is not Flux2KleinAdapter.advance):
        raise ValueError("reentry native/base/closure source ABI differs")
    return {"schema":"mdb-flux-reentry-native-source-abi-v1","verified":True,"base_step":base,
        "reentry_expected":{key:pins[key] for key in actual},"reentry_actual":actual,
        "closure_module_sha256":expected_closure_module_sha256,
        "closure_type":closure_type.__module__+"."+closure_type.__qualname__,"native_advance_inherited":True,
        "scope":"Exact base/project/joint Python source pins plus separately identified opt-in closure surface. Native advancement is inherited unchanged; no CUDA kernel independence claim."}


def _receipt(output,store,counts,branch,value):
    body = value.to_dict()
    counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps",1)
    counts["receipt_emissions"] += 1
    if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
        raise RuntimeError("reentry finite physical budget exhausted")
    store.receipt(value)
    with (output/"receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps({"sequence":counts["receipt_emissions"]-1,"branch":branch,
            "receipt":value.fingerprint},sort_keys=True)+"\n")


def run_prefix(*,adapter,session,output,store,counts,wall):
    from mdb_lanes.native_navigation import NativeCutNavigator

    initial = session.capture()
    if initial.execution_point["logical_step"] != 0 or initial.execution_point["phase"] != "project":
        raise ValueError("reentry requires a clean newly encoded/seeded native initial state")
    store.save(initial)
    nav = NativeCutNavigator(session,max_steps=68,
        on_receipt=lambda value:_receipt(output,store,counts,"prefix",value))
    path = []
    for _ in range(68):
        wall()
        path.append(nav.step().to_dict())
    pre6 = nav.capture()
    if any(pre6.execution_point[key] != value for key,value in
            {"logical_step":2,"phase":"single","index":6,"local_clock":6}.items()):
        raise ValueError("reentry prefix did not reach step2/pre6")
    store.save(pre6)
    record = {"schema":"mdb-flux-reentry-prefix-v1","context":CONTEXT,"initial":initial.fingerprint,
        "pre6":pre6.fingerprint,"point":dict(pre6.execution_point),"receipts":path,
        "counts":{"adapter_steps":68,"acts":0,"receipt_emissions":68},
        "scope":"One clean native seed97 history; no imported or engineered low/high source."}
    _util()["save"](output/"prefix.json",record)
    return pre6,record


def bind_inputs(*,model,pre6,array,counts,prefix_ref,config,production_profile):
    import torch

    from mdb_lanes.forward_recording import tensor_descriptor

    previous = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))
    binding = _step()["bind_inputs"](model=model,cut=pre6,array=array,counts=counts,prefix_ref=prefix_ref,
        config=config,profile=previous["attention_profile"],production_profile=production_profile)
    binding["schema"] = "mdb-flux-reentry-binding-v1"
    binding["target_entries_before_seal"] = {key:0 for key in contract()["target_calls"]}
    def archive(parameters):
        result = {}
        for name,value in parameters.items():
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError("reentry learned source must be finite native BF16")
            result[name] = array(value)
        return result
    project = {name:value for name,value in model.named_parameters() if name.startswith(PROJECT_PREFIXES)}
    if not project or getattr(model.time_guidance_embed,"guidance_embedder",None) is not None:
        raise ValueError("reentry requires deterministic unguided full native project profile")
    binding["project"] = {"parameters":archive(project),
        "native_parameters":{name:tensor_descriptor(value) for name,value in project.items()},
        "conditioning":array(pre6.payload["conditioning"]),"img_ids":array(pre6.payload["img_ids"]),
        "txt_ids":array(pre6.payload["txt_ids"]),"timestep":array(pre6.payload["timesteps"][3]),
        "axes_dims_rope":list(model.pos_embed.axes_dim),"rope_theta":model.pos_embed.theta,
        "timestep_channels":model.time_guidance_embed.time_proj.num_channels}
    joint = model.transformer_blocks[0]
    attn = joint.attn
    if (any(getattr(joint,name).elementwise_affine or getattr(joint,name).eps != 1e-6
            for name in ("norm1","norm1_context","norm2","norm2_context"))
            or len({attn.norm_q.eps,attn.norm_k.eps,attn.norm_added_q.eps,attn.norm_added_k.eps}) != 1):
        raise ValueError("reentry joint0 normalization profile differs")
    joint_parameters = dict(joint.named_parameters())
    binding["joint0"] = {"parameters":archive(joint_parameters),
        "native_parameters":{name:tensor_descriptor(value) for name,value in joint_parameters.items()},
        "heads":attn.heads,"head_dim":attn.inner_dim//attn.heads,"layernorm_eps":joint.norm1.eps,
        "qk_eps":attn.norm_q.eps,"processor_type":type(attn.processor).__module__+"."+type(attn.processor).__qualname__,
        "scope":"Separate image/text attention, gated attention residual, second LayerNorm/modulation, then serial stream-specific MLP and residual."}
    return binding


def predict_reentry(binding,*,archive_root,transition,boundary,reentry,config,event):
    from mdb_lanes.tensor_archive import load_tensor_archive

    load = lambda record:load_tensor_archive(archive_root,record)
    remaining = _step()["predict_step"](binding,archive_root=archive_root,transition=transition,
        boundary=boundary,config=config,event=event)
    spec = binding["project"]
    parameters = {name:load(value) for name,value in spec["parameters"].items()}
    project = reentry["predict_project"](remaining["commit"]["latent"],load(spec["conditioning"]),
        load(spec["img_ids"]),load(spec["txt_ids"]),load(spec["timestep"]),parameters,
        axes_dims_rope=spec["axes_dims_rope"],rope_theta=spec["rope_theta"],timestep_channels=spec["timestep_channels"],
        projection_chunk=config["projection_chunk"])
    del parameters
    gc.collect()
    event("project_forecast_complete")
    spec = binding["joint0"]
    parameters = {name:load(value) for name,value in spec["parameters"].items()}
    joint = reentry["predict_joint"](project["image"],project["text"],project["context"],parameters,
        heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
        query_chunk=config["query_chunk"],projection_chunk=config["projection_chunk"],retain_fields=False)
    del parameters
    gc.collect()
    event("joint_forecast_complete")
    return {"remaining_step":remaining,"project":project,"joint0":joint,
        "scope":"Whole14block/readout/commit/project/joint0 prediction before native targets; each stage consumes own predicted complete state, never native future context/RoPE/latent/I/T."}


def validate_prediction(pre6,prediction):
    import numpy as np

    _step()["validate_prediction"](pre6,prediction["remaining_step"])
    normal = _util()["json_value"]
    descriptor = _step()["_descriptor"]
    project,joint = prediction["project"],prediction["joint0"]
    if (normal(project["input_latent_descriptor"]) != normal(prediction["remaining_step"]["commit"]["latent_descriptor"])
            or normal(project["conditioning_descriptor"]) != descriptor(pre6.payload["conditioning"].float().cpu().numpy())
            or normal(project["image_descriptor"]) != descriptor(project["image"])
            or normal(project["text_descriptor"]) != descriptor(project["text"])
            or normal(joint["input_image_descriptor"]) != normal(project["image_descriptor"])
            or normal(joint["input_text_descriptor"]) != normal(project["text_descriptor"])
            or normal(joint["image_descriptor"]) != descriptor(joint["image"])
            or normal(joint["text_descriptor"]) != descriptor(joint["text"])):
        raise ValueError("reentry own-state complete stage/descriptor join differs")
    for name,value in project["context"].items():
        if name == "rope":
            if len(value) != 2 or any(item.dtype != np.float32 or not np.isfinite(item).all() for item in value):
                raise ValueError("sealed reentry RoPE profile differs")
        else:
            descriptor(value)


def run_panel(*,adapter,pre6,output,config,store,array,counts,wall,event,prediction):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb_lanes.flux_forecast_closure import closure_act
    from mdb_lanes.native_navigation import NativeCutNavigator

    util = _util()
    def navigator(cut,branch,limit):
        return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=limit,
            on_receipt=lambda value:_receipt(output,store,counts,branch,value))
    def archive_tree(value):
        if isinstance(value,torch.Tensor):
            return array(value)
        if isinstance(value,dict):
            return {key:archive_tree(child) for key,child in value.items()}
        if isinstance(value,(tuple,list)):
            return [archive_tree(child) for child in value]
        return value
    def boundary(cut):
        return {"cut":cut.fingerprint,"point":dict(cut.execution_point),"closure":{
            name:archive_tree(cut.payload[name]) for name in ("latent","context","image","text","noise_prediction")},
            "clock":{name:cut.payload[name] for name in ("step","phase","cursor")}}
    nav = navigator(pre6,"reference",18)
    reference,receipts,cuts = [],[],[]
    for position in range(18):
        wall()
        entry = nav.step()
        cut = nav.capture()
        store.save(cut)
        cuts.append(cut)
        receipts.append(entry.to_dict())
        reference.append({"index":position,"receipt_id":entry.fingerprint,**boundary(cut)})
        event("reference_transition_complete",index=position,point=dict(cut.execution_point))
    native_z3,native_project,native_joint = cuts[15:18]
    if (native_z3.execution_point["logical_step"] != 3 or native_z3.execution_point["phase"] != "project"
            or any(native_project.execution_point[key] != value for key,value in {"logical_step":3,"phase":"joint","index":0}.items())
            or any(native_joint.execution_point[key] != value for key,value in {"logical_step":3,"phase":"joint","index":1}.items())):
        raise ValueError("reentry actual native clock scaffolds differ")
    util["save"](output/"native-segment.json",{"input_cut":pre6.fingerprint,"boundaries":reference,"receipts":receipts,
        "native_Z3":boundary(native_z3),"native_project":boundary(native_project),"common_post_joint0":boundary(native_joint),
        "counts":{"adapter_steps":18,"acts":0,"receipt_emissions":18}})
    validate_prediction(pre6,prediction)
    del cuts
    # The old observer is source-pinned; only phase labels/final flags differ.
    observer_class = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))["ContinuationObserver"]
    observer = observer_class(array=array,plan=config["observation_plan"])
    rows = []
    def tensor(value,reference):
        if not isinstance(value,np.ndarray) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("reentry payload must contain complete finite FP32 native widenings")
        if reference.dtype == torch.bfloat16:
            _step()["_descriptor"](value)
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError("reentry payload geometry differs")
        converted = torch.from_numpy(np.array(value,copy=True)).to(device=reference.device,dtype=reference.dtype)
        result = torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
        result.copy_(converted)
        if not np.array_equal(result.float().cpu().numpy().view(np.uint32),value.view(np.uint32)):
            raise ValueError("reentry payload cast would silently change sealed values")
        return result
    def tree(value,reference):
        if isinstance(reference,torch.Tensor):
            return tensor(value,reference)
        if isinstance(reference,dict) and set(value) == set(reference):
            return {key:tree(value[key],reference[key]) for key in reference}
        if isinstance(reference,(tuple,list)) and len(value) == len(reference):
            return tuple(tree(left,right) for left,right in zip(value,reference,strict=True))
        raise ValueError("reentry nested complete context profile differs")
    def install_closure(nav,name,mode):
        before = nav.capture()
        if mode == "sham":
            payload = {key:before.payload[key] for key in ("latent","image","text","context")}
        else:
            stage = prediction["project"] if mode == "project" else prediction["joint0"]
            payload = {"latent":tensor(prediction["remaining_step"]["commit"]["latent"],before.payload["latent"]),
                "context":tree(prediction["project"]["context"],before.payload["context"]),
                "image":tensor(stage["image"],before.payload["image"]),"text":tensor(stage["text"],before.payload["text"])}
        store.save(before)
        act = closure_act(before,**payload,name=name+".complete-closure")
        receipt = nav.apply(act)
        after = nav.capture()
        store.save(after)
        if describe(after.payload) != describe({**before.payload,**payload}):
            raise ValueError("reentry closure Act changed undeclared state or sealed values")
        return {"slots":["latent","context","image","text"],"parent":before.fingerprint,"post_act":after.fingerprint,
            "act":act.manifest(),"receipt":receipt.to_dict(),"receipt_id":receipt.fingerprint,"undeclared_slots_exact":True}
    for spec in config["branches"]:
        name,mode = spec["id"],spec["mode"]
        parent = native_z3 if mode == "latent" else native_project if mode == "project" else native_joint
        nav = navigator(parent,name,28)
        row = {**spec,"parent":parent.fingerprint,"acts":[],"bridge_receipts":[],"bridge_boundaries":[]}
        if mode == "latent":
            value = tensor(prediction["remaining_step"]["commit"]["latent"],parent.payload["latent"])
            row["acts"].append(util["replace_slot"](nav,"latent",value,name=name+".Z3",store=store))
        elif mode != "native":
            row["acts"].append(install_closure(nav,name,mode))
        if row["acts"]:
            row["installed"] = boundary(nav.capture())
        for _ in range(2 if mode == "latent" else 1 if mode == "project" else 0):
            wall()
            receipt = nav.step()
            bridge = nav.capture()
            store.save(bridge)
            row["bridge_receipts"].append(receipt.to_dict())
            row["bridge_boundaries"].append(boundary(bridge))
        common = nav.capture()
        if any(common.execution_point[key] != value for key,value in
                {"logical_step":3,"phase":"joint","index":1,"local_clock":1}.items()):
            raise ValueError("all reentry futures must join actual common post-joint0")
        store.save(common)
        row["common_post_joint0"] = boundary(common)
        incoming = (row["bridge_receipts"][-1]["fingerprint"] if row["bridge_receipts"] else
            row["acts"][-1]["receipt_id"] if row["acts"] else receipts[-1]["fingerprint"])
        def observe(current,index,incoming):
            entry = observer.observe(name,current,index=index,incoming_receipt=incoming)
            entry["complete_cut_retained"] = index in {0,26}
            for key,value in entry["slots"].items():
                if key in {"image","text"}:
                    active = current.execution_point["phase"] in {"joint","single","readout"}
                    value["meaning"] = {"phase":"working-joint-or-single-stream" if active else "absent-or-stale-working-stream",
                        "meaningful_propagation_observable":active and value["present"],"stale":not active}
            return entry
        observe(common,0,incoming)
        path = []
        for position in range(26):
            wall()
            receipt = nav.step()
            path.append(receipt.to_dict())
            observe(nav.capture(),position+1,receipt.fingerprint)
        final = nav.capture()
        if final.execution_point["logical_step"] != 4 or final.execution_point["phase"] != "project":
            raise ValueError("reentry native future did not commit Z4")
        store.save(final)
        counts["vae_calls"] += 1
        image = adapter.decode(nav.session,height=512,width=512)
        image_path = output/(name+".png")
        image.save(image_path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint,final_latent=array(final.payload["latent"]),
            image={"path":image_path.name,"sha256":_sha(image_path)},continuation_receipts=path)
        util["save"](output/"rows"/(name+".json"),row)
        rows.append(row)
        event("branch_complete",row=name)
    if set(observer.rows) != set(ORDER) or any(len(value) != 27 for value in observer.rows.values()):
        raise ValueError("reentry continuation observation census differs")
    util["save"](output/"continuation-observations.json",{"schema":"mdb-flux-reentry-continuation-v1",
        "plan":config["observation_plan"],"rows":observer.rows,"branch_execution_order":ORDER,"model_calls_added":0})
    controls = {"sham_final_latent_exact_native":rows[0]["final_latent"] == rows[1]["final_latent"],
        "sham_png_exact_native":rows[0]["image"]["sha256"] == rows[1]["image"]["sha256"]}
    return {"rows":rows,"counts":dict(counts),"control_observations":controls,
        "native_segment_sha256":_sha(output/"native-segment.json"),
        "continuation_observations_sha256":_sha(output/"continuation-observations.json")}


def run(*,model,adapter,session,output,config,request,source,counts,array=None,guard_components=None,
        wall=lambda:None,event=lambda *args,**kwargs:None,started=None,production_profile=True):
    from saturn_pub.store import LocalStore

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    started = time.monotonic() if started is None else started
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"plan.json").exists():
        raise ValueError("immutable reentry output already contains a plan")
    util = _util()
    save,identity = util["save"],util["identity"]
    array = array or _step()["make_array_writer"](output,wall)
    save(output/"plan.json",config)
    save(output/"plan-frozen.json",{"plan_sha256":_sha(output/"plan.json"),"job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_identity":identity(request["source_hashes"]),
        "counts":dict(counts),"scope":"Frozen before native68sourceprefix; full prediction sealed before selected targets."})
    boundary = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    abi = verify_source_abi(boundary=boundary,reentry=reentry,closure_type=type(adapter),
        expected_closure_module_sha256=config["expected_closure_module_sha256"])
    if adapter.execution.get("adapter") != "mdb-flux-forecast-closure-v1":
        raise ValueError("reentry requires the separately identified typed closure execution contract")
    save(output/"native-source-abi.json",abi)
    store = LocalStore(output/"typed-store")
    pre6,prefix = run_prefix(adapter=adapter,session=session,output=output,store=store,counts=counts,wall=wall)
    prefix_ref = {"path":"prefix.json","sha256":_sha(output/"prefix.json")}
    source = {**source,"prefix":prefix_ref,"native_source_abi":{"path":"native-source-abi.json",
        "sha256":_sha(output/"native-source-abi.json")}}
    save(output/"source.json",source)
    transition_path = Path(__file__).with_name("flux_transition_reference.py")
    if _sha(transition_path) != _step()["TRANSITION_REFERENCE_SHA256"]:
        raise ValueError("frozen complete single-block reference source changed")
    transition = runpy.run_path(str(transition_path))
    release = {"native":False,"blocked":0,"targets":{key:0 for key in contract()["target_calls"]}}
    targets = {model.single_transformer_blocks[index]:str(index) for index in BLOCK_INDICES}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out",model.pos_embed:"pos_embed",
        model.transformer_blocks[0]:"joint0",**{getattr(model,name):name for name in PROJECT_TARGETS}})
    def guard(module,args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("native selected/current or next-step component entered before full reentry seal")
        if module in targets:
            release["targets"][targets[module]] += 1
    handles = [module.register_forward_pre_hook(guard) for component in (guard_components or (model,)) for module in component.modules()]
    try:
        binding = bind_inputs(model=model,pre6=pre6,array=array,counts=counts,prefix_ref=prefix_ref,
            config=config,production_profile=production_profile)
        save(output/"binding.json",binding)
        tx = ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,
            input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
            frontier={"kind":"architectural","binding":binding},execution_scope=config["scope"],
            budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000,"max_bytes":config["transaction_max_bytes"]})
        with tx:
            tx.frontier("source_binding",binding)
            event("prediction_start")
            tx.predict("reentry",lambda _:{"predictions":predict_reentry(binding,archive_root=output,
                transition=transition,boundary=boundary,reentry=reentry,config=config,event=event)})
            tx.predict("hypotheses",lambda _:{"predictions":config["hypotheses"]})
            seal = tx.seal()
            event("reentry_sealed",seal=seal,target_entries_before_seal=dict(release["targets"]),blocked_entries=release["blocked"])
            release["native"] = True
            tx.execute("native",lambda context:run_panel(adapter=adapter,pre6=pre6,output=output,config=config,
                store=store,array=array,counts=counts,wall=wall,event=event,prediction=context.predictions["reentry"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output/"transaction-report.json",tx.report())
    if (any(counts[name] != COUNTS[name] for name in ("adapter_steps","acts","receipt_emissions","images","vae_calls","candidate_calls"))
            or release["blocked"] != 0 or release["targets"] != contract()["target_calls"]):
        raise ValueError("reentry completed physical/target census differs")
    summary = {"schema":"mdb-flux-reentry-result-v1","job_id":request["job_id"],"request_sha256":request["request_sha256"],
        "counts":dict(counts),"prefix_counts":prefix["counts"],"panel_counts":contract()["panel_counts"],
        "target_calls":release["targets"],"blocked_entries":release["blocked"],"seal":seal,
        "model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "transaction_verification":verify_transaction(tx.root),"control_observations":tx.observation("native")["control_observations"],
        "worker_seconds":time.monotonic()-started,"mechanics_status":"pending-independent-audit",
        "terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary)
    return util["json_value"](summary)


def worker():
    """One admitted CUDA process; SSH custody is performed by the launcher."""
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90,repeat=True)
    util = _util()
    save,sha,identity,json_value = (util[name] for name in ("save","sha","identity","json_value"))
    config,request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if (request["config"] != config or any(sha(path) != digest for path,digest in request["source_hashes"].items())
            or config.get("transport_method") != "planned-authenticated-ssh-manifest"):
        raise ValueError("reentry request/source/transport identity differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path("flux-reentry-result")
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_FLUX_REENTRY_BOOTSTRAP "+json.dumps({"job_id":request["job_id"],"seconds":time.monotonic()-started}),flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb_lanes.flux_forecast_closure import FluxForecastClosureAdapter

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("reentry requires admitted CUDA")
    reference = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    source_abi = verify_source_abi(boundary=reference,reentry=reentry,closure_type=FluxForecastClosureAdapter,
        expected_closure_module_sha256=config["expected_closure_module_sha256"])
    save(output / "pre-model-source-abi.json",source_abi)
    scalar_qualification = reference["qualify_torch_scalar_promotion"](torch,device="cuda")
    save(output / "cuda-scalar-qualification.json",scalar_qualification)
    disk = shutil.disk_usage(output)
    if disk.free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("reentry output filesystem lacks declared8GiB free disk")
    save(output / "disk-admission.json",{"output_filesystem":str(output.resolve()),"free_bytes":disk.free,
        "total_bytes":disk.total,"max_output_bytes":config["transport_limits"]["max_expanded_bytes"],
        "scope":"Pre-model free-space observation and finite output bound; no quota or guaranteed future free space."})
    counts = {key:0 for key in COUNTS}
    counts["physical_batch"] = 1
    heartbeat = {"last":time.monotonic(),"sequence":0}
    def event(kind,**fields):
        body = json_value({"sequence":heartbeat["sequence"],"kind":kind,"seconds":time.monotonic()-started,
            "counts":dict(counts),**fields})
        heartbeat["sequence"] += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body,sort_keys=True,allow_nan=False)+"\n")
        print("MDB_FLUX_REENTRY_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        now = time.monotonic()
        if now-started > config["max_wall_seconds"]:
            raise RuntimeError("finite reentry wall budget exhausted")
        if now-heartbeat["last"] >= 15:
            heartbeat["last"] = now
            event("heartbeat")
    array = _step()["make_array_writer"](output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("reentry checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("reentry requires deterministic distilled five-joint/twenty-single Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind):sha(inspect.getfile(kind)) for kind in (type(pipe),type(pipe.text_encoder),
        type(pipe.transformer),type(pipe.transformer.single_transformer_blocks[6]),
        type(pipe.transformer.single_transformer_blocks[6].attn.processor),type(pipe.transformer.norm_out),
        type(pipe.vae),type(pipe.scheduler),Flux2KleinAdapter,FluxForecastClosureAdapter,
        type(pipe.transformer.transformer_blocks[0]),type(pipe.transformer.pos_embed),
        type(pipe.transformer.time_guidance_embed.timestep_embedder))}
    pipe.text_encoder.to("cuda")
    with torch.inference_mode():
        conditioning,txt_ids = pipe.encode_prompt(CONTEXT["prompt"],device=torch.device("cuda"),
            num_images_per_prompt=1,max_sequence_length=512)
    counts["prompt_encodings"] += 1
    encoding = {"conditioning":array(conditioning),"txt_ids":array(txt_ids)}
    conditioning,txt_ids = conditioning.cpu(),txt_ids.cpu()
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    event("encoder_released")
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = FluxForecastClosureAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    with torch.inference_mode():
        conditioning,txt_ids = conditioning.to("cuda"),txt_ids.to("cuda")
        latent,img_ids = pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,conditioning.dtype,
            torch.device("cuda"),torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
        timesteps,steps = retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
        if steps != 4:
            raise ValueError("reentry native schedule differs")
        session = adapter.session(latent=latent,conditioning=conditioning,img_ids=img_ids,txt_ids=txt_ids,
            timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
    source = {"schema":"mdb-flux-reentry-source-v1","job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_hashes":request["source_hashes"],"context":CONTEXT,
        "encoding":encoding,"encodings":encoding,"model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "checkpoint_sha256":identity(inventory),"checkpoint_files":inventory,"implementation_sources":implementations,
        "versions":{name:importlib.metadata.version(name) for name in ("torch","numpy","diffusers","transformers","saturn-pub","mdb-runtime")},
        "counts_before_prefix":dict(counts),"cuda_scalar_qualification":{"path":"cuda-scalar-qualification.json",
            "sha256":sha(output / "cuda-scalar-qualification.json")},
        "pre_model_source_abi":{"path":"pre-model-source-abi.json","sha256":sha(output / "pre-model-source-abi.json")},
        "scope":config["scope"]}
    summary = run(model=pipe.transformer,adapter=adapter,session=session,output=output,config=config,request=request,
        source=source,counts=counts,array=array,guard_components=(pipe.transformer,pipe.vae),wall=wall,event=event,started=started)
    if counts != COUNTS:
        raise ValueError("reentry global load/encoding/noise census differs")
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    manifest = _step()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"])
    print("MDB_FLUX_REENTRY_COMPLETE "+json.dumps(json_value({"summary":summary,"manifest":manifest}),sort_keys=True),flush=True)
    return output,summary


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-reentry-result")
        if output.is_dir() and not (output/"remote-manifest.json").exists():
            _util()["save"](output/"failure.json",{"schema":"mdb-flux-reentry-failure-v1",
                "job_id":os.environ.get("MRUN_JOB_ID"),"error":f"{type(error).__name__}: {error}",
                "mechanics_status":"errored","scope":"Completed evidence preserved; no model retry, archive or fabricated report."})
        raise
