"""Seal own-state 56-stage complete two-pass prediction, then native futures.

Only byte-pinned pure predecessor functions are reused. This module owns its
finite execution, guards and atomic live and terminal admission; predecessor worker
functions/globals are never changed or invoked as a different experiment.
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

CONTEXT = {"id":"p1-s109","prompt_id":"p1","seed":109,
    "prompt":"A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BLOCK_INDICES = list(range(20))
LOGICAL_STEPS = [2,3]
BRANCHES = [{"id":"native","mode":"native"},{"id":"sham","mode":"sham"},
    {"id":"bridge.predicted","mode":"bridge"},{"id":"full.predicted","mode":"full"}]
ORDER = [row["id"] for row in BRANCHES]
EXECUTION_ORDER = ["full.predicted","native","sham","bridge.predicted"]
COUNTS = {"model_loads":1,"prompt_encodings":1,"noise_draws":1,"adapter_steps":134,"acts":3,
    "candidate_calls":0,"receipt_emissions":137,"images":4,"vae_calls":4,"physical_batch":1}
SAMPLE_INDICES = [0,31,511,512,543,767,1023,1535]
IMAGE_SAMPLE_INDICES = [0,31,32,255,511,512,767,1023]
PROJECT_TARGETS = ("x_embedder","context_embedder","time_guidance_embed","double_stream_modulation_img",
    "double_stream_modulation_txt","single_stream_modulation")
OBSERVATION_PLAN = {"reference_rows":["native","bridge.predicted"],"boundaries_per_branch":23,
    "slots":["image","text","latent","noise_prediction"],"storage":"lossless-bf16-bits-and-complete-typed-cuts",
    "scope":"Common step3/single0 and22 native exits for reference and bridge. All present full arrays/cuts retained; working slots resetNone at terminal commit."}
OLD_CLOSURE_SHA256 = "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"
PRIOR_JOINT_CLOSURE_SHA256 = "bb352ae4859335abdd751d33ccc2774fa4d37c53f5167c5012183c693b0d709c"
PRIOR_TAIL_CLOSURE_SHA256 = "2788c984045098532009e105219e7c9cfcc38732a54d892bdc296149a979b586"
PRIOR_FINAL_CLOSURE_SHA256 = "b70f3d68f7fb0b1c0e2892bb7ab4efb8828a906ef80d9acb6f794608d673e579"
LEGACY_SOURCES = {
    "flux_final_latent_campaign.py":"7f1a6370c993af0808857ace14ab17aaf3b802ea65db47eb2adb2f5cbc891d4d",
    "flux_joint_tail_campaign.py":"7815f92400b531ed218112acdab5b626365f5b23379b2be5711cdffafa180340",
    "flux_joint_chain_campaign.py":"fe5a8fd7bc4dc67a183956dc050a3c0330ae6d7456e9019c4f0965357dff18a7",
    "flux_reentry_campaign.py":"7033626038e1858408415799bd4ef0e31a727993f8bca7eb7819d67f184a9fe5",
    "flux_reentry_reference.py":"e7ba7e9b55d775386767a15d87af4f011d64831322a38ab7ee4c59537b31e108",
    "flux_step_boundary_campaign.py":"e3f0eac65d4d26bf49e0f562f6b70df38a61ceb0deb1a2001deb77b7986dde66",
    "flux_step_boundary_reference.py":"f890a62248f0ce8a1f30d29fe3db421326a196640c0fd578a8be7960dd315768",
    "flux_transition_reference.py":"f531e059635fe9940d55c38eaf5cc1d7ab9dd3a4cd1adeee5225ba9342f3bf83",
    "flux_mlp_replacement.py":"5ee9ba9d67c44e947bb6983a8bdec7b1e9335aaf28f0ffdd8ce71ab559767f8a",
    "flux_single_contexts.py":"c14b0e297f41eae91c9138e7608c0637f41eecd24ca73d0fa7f9a9e1eb6c0168",
    "flux_attention_replacement_campaign.py":"9b47ed44799f0098518b5ae6826da30995c427ed77856f9f705c6a34c70ddbf1"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def prior():
    for name,digest in LEGACY_SOURCES.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("immutable predecessor source differs: "+name)
    return runpy.run_path(str(Path(__file__).with_name("flux_final_latent_campaign.py")))


def step():
    return prior()["step"]()


def util():
    return prior()["util"]()


def contract(config=None):
    if config is not None and (config.get("block_indices",BLOCK_INDICES) != BLOCK_INDICES
            or config.get("logical_steps",LOGICAL_STEPS) != LOGICAL_STEPS):
        raise ValueError("two-pass requires two complete declared passes")
    return json.loads(json.dumps({"context":CONTEXT,"block_indices":BLOCK_INDICES,"logical_steps":LOGICAL_STEPS,"branches":BRANCHES,
        "order":ORDER,"execution_order":EXECUTION_ORDER,"counts":COUNTS,"prefix_steps":56,"reference_steps":56,"bridge_steps":22,"suffix_steps":0,
        "panel_counts":{"adapter_steps":78,"acts":3,"receipt_emissions":81,"candidate_calls":0,"images":4,"vae_calls":4},
        "target_calls":{**{str(index):3 for index in BLOCK_INDICES},"norm_out":3,"proj_out":3,
            **{name:2 for name in PROJECT_TARGETS},"pos_embed":4,**{"joint"+str(index):2 for index in range(5)}},"observation_plan":OBSERVATION_PLAN}))


def validate_config(config):
    fixed = {"context":CONTEXT,"block_indices":BLOCK_INDICES,"logical_steps":LOGICAL_STEPS,"branches":BRANCHES,"branch_execution_order":EXECUTION_ORDER,
        "selector":{"logical_step":2,"phase":"project","index":None},"height":512,"width":512,"steps":4,
        "sample_indices":SAMPLE_INDICES,"image_sample_indices":IMAGE_SAMPLE_INDICES,"query_chunk":64,
        "projection_chunk":256,"neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits",
        "archive_integer_ids":"native-int64","cpu_threads":4,"transaction_max_bytes":1536<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":14<<30,"max_member_bytes":1536<<20},"observation_plan":OBSERVATION_PLAN,
        "expected_closure_module_sha256":OLD_CLOSURE_SHA256,"expected_joint_closure_module_sha256":PRIOR_JOINT_CLOSURE_SHA256,
        "expected_tail_closure_module_sha256":PRIOR_TAIL_CLOSURE_SHA256,
        "expected_final_closure_module_sha256":PRIOR_FINAL_CLOSURE_SHA256}
    if any(config.get(key) != value for key,value in fixed.items()):
        raise ValueError("two-pass frozen scientific/closure geometry differs")
    for key in ("adapter_steps","acts","candidate_calls","images","vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("two-pass finite physical census differs")
    digest = config.get("expected_derived_terminal_module_sha256")
    if not isinstance(digest,str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("two-pass requires an explicit derived terminal module pin")
    if type(config.get("cpu_threads")) is not int or type(config.get("max_wall_seconds")) is not int or not 1<=config["max_wall_seconds"]<=1200:
        raise ValueError("two-pass finite execution budget differs")


def verify_source_abi(*,boundary,reentry,closure_type,expected_derived_terminal_module_sha256):
    from mdb_lanes import flux_derived_terminal

    old = prior()["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=closure_type,
        expected_final_closure_module_sha256=PRIOR_FINAL_CLOSURE_SHA256)
    if sha(inspect.getfile(flux_derived_terminal)) != expected_derived_terminal_module_sha256:
        raise ValueError("two-pass derived terminal source ABI differs")
    return {"schema":"mdb-flux-two-pass-native-source-abi-v1","verified":True,"prior_final_latent":old,
        "derived_terminal_module_sha256":expected_derived_terminal_module_sha256,"native_advance_inherited":True,
        "scope":"All prior mathematical and native source pins unchanged; separately pinned explicit non-native prefix-derived terminal scaffold."}


def receipt(output,store,counts,branch,value):
    body = value.to_dict()
    counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps",1)
    counts["receipt_emissions"] += 1
    if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
        raise RuntimeError("two-pass finite physical budget exhausted")
    store.receipt(value)
    with (output/"receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps({"sequence":counts["receipt_emissions"]-1,"branch":branch,"receipt":value.fingerprint},sort_keys=True)+"\n")


def run_prefix(*,adapter,session,output,store,counts,wall):
    from mdb_lanes.native_navigation import NativeCutNavigator

    initial = session.capture()
    if initial.execution_point["logical_step"] != 0 or initial.execution_point["phase"] != "project":
        raise ValueError("two-pass requires a clean newly encoded/seeded native initial state")
    store.save(initial)
    nav = NativeCutNavigator(session,max_steps=56,on_receipt=lambda value:receipt(output,store,counts,"prefix",value))
    path = []
    for _ in range(56):
        wall()
        path.append(nav.step().to_dict())
    pre_project = nav.capture()
    if any(pre_project.execution_point[key] != value for key,value in {"logical_step":2,"phase":"project","index":None,"local_clock":0}.items()):
        raise ValueError("two-pass prefix did not reach step2/pre_project")
    store.save(pre_project)
    record = {"schema":"mdb-flux-two-pass-prefix-v1","context":CONTEXT,"initial":initial.fingerprint,
        "pre_project":pre_project.fingerprint,"point":dict(pre_project.execution_point),"receipts":path,
        "counts":{"adapter_steps":56,"acts":0,"receipt_emissions":56},"scope":"One clean seed109 history; no imported/engineered state."}
    util()["save"](output/"prefix.json",record)
    return pre_project,record




def bind_inputs(*,model,pre_project,array,counts,prefix_ref,config,production_profile):
    import torch
    from saturn_pub.values import describe

    from mdb_lanes.forward_recording import tensor_descriptor

    p = pre_project.payload
    if (p["step"] != 2 or p["phase"] != "project" or p["cursor"] != 0
            or any(p[key] is not None for key in ("image","text","context","noise_prediction"))):
        raise ValueError("two-pass source must be the genuine pre-project frontier")
    def parameter(value):
        if value is None:
            return None
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("two-pass learned parameters must be finite native BF16")
        return array(value)
    def bank(parameters):
        return {name:parameter(value) for name,value in parameters.items()}
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_campaign.py")))
    profile = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))["attention_profile"]
    binding = {"schema":"mdb-flux-two-pass-binding-v1","input_cut":pre_project.fingerprint,
        "point":dict(pre_project.execution_point),"prefix":prefix_ref,"block_indices":BLOCK_INDICES,"logical_steps":LOGICAL_STEPS,
        "counts_before_seal":dict(counts),"target_entries_before_seal":{key:0 for key in contract()["target_calls"]},
        "inputs":{key:array(p[key]) for key in ("latent","conditioning","img_ids","txt_ids","timesteps","sigmas")},
        "text_rows":p["conditioning"].shape[1],"final_blocks":{},
        "carried_state":describe({key:value for key,value in p.items()
            if key not in {"image","text","latent","context","noise_prediction","step","phase","cursor"}}),
        "clock_state":{key:p[key] for key in ("step","phase","cursor")}}
    project = {name:value for name,value in model.named_parameters() if name.startswith(reentry["PROJECT_PREFIXES"])}
    if not project or getattr(model.time_guidance_embed,"guidance_embedder",None) is not None:
        raise ValueError("two-pass requires unguided qualified project")
    binding["project"] = {"parameters":bank(project),
        "native_parameters":{name:tensor_descriptor(value) for name,value in project.items()},
        "axes_dims_rope":list(model.pos_embed.axes_dim),"rope_theta":model.pos_embed.theta,
        "timestep_channels":model.time_guidance_embed.time_proj.num_channels}
    for index,joint in enumerate(model.transformer_blocks):
        attn = joint.attn
        if (any(getattr(joint,name).elementwise_affine or getattr(joint,name).eps != 1e-6
                for name in ("norm1","norm1_context","norm2","norm2_context"))
                or len({attn.norm_q.eps,attn.norm_k.eps,attn.norm_added_q.eps,attn.norm_added_k.eps}) != 1):
            raise ValueError("two-pass joint normalization profile differs")
        parameters = dict(joint.named_parameters())
        binding["joint"+str(index)] = {"parameters":bank(parameters),
            "native_parameters":{name:tensor_descriptor(value) for name,value in parameters.items()},
            "native_parameter_prefix":f"transformer_blocks.{index}.","block_index":index,
            "heads":attn.heads,"head_dim":attn.inner_dim//attn.heads,
            "layernorm_eps":joint.norm1.eps,"qk_eps":attn.norm_q.eps}
    for index,block in enumerate(model.single_transformer_blocks):
        attn = block.attn
        h,m = attn.inner_dim,attn.mlp_hidden_dim
        if (block.norm.elementwise_affine or block.norm.eps != 1e-6 or attn.norm_q.eps != attn.norm_k.eps
                or not attn.norm_q.elementwise_affine or not attn.norm_k.elementwise_affine
                or attn.norm_q.eps != 1e-6 or h != attn.heads*attn.head_dim
                or tuple(attn.to_qkv_mlp_proj.weight.shape) != (3*h+2*m,h)
                or tuple(attn.to_out.weight.shape) != (h,h+m)):
            raise ValueError("two-pass single profile differs")
        if production_profile and (h,m,attn.heads,attn.head_dim) != (3072,9216,24,128):
            raise ValueError("two-pass qualified production dimensions differ")
        parameters = {"qkv_mlp_weight":attn.to_qkv_mlp_proj.weight,"qkv_mlp_bias":attn.to_qkv_mlp_proj.bias,
            "to_out_weight":attn.to_out.weight,"to_out_bias":attn.to_out.bias,
            "norm_q_weight":attn.norm_q.weight,"norm_k_weight":attn.norm_k.weight}
        binding["final_blocks"][str(index)] = {"parameters":bank(parameters),
            "native_parameters":{name:tensor_descriptor(value) for name,value in block.named_parameters()},
            "native_parameter_prefix":f"single_transformer_blocks.{index}.","block_index":index,
            "heads":attn.heads,"hidden_width":h,"mlp_width":m,"head_dim":attn.head_dim,
            "layernorm_eps":block.norm.eps,"qk_eps":attn.norm_q.eps,"attention_profile":profile(block)}
    if len(model.transformer_blocks) != 5 or len(model.single_transformer_blocks) != 20:
        raise ValueError("two-pass requires exact five joint and twenty single banks")
    norm = model.norm_out
    if (not isinstance(norm.norm,torch.nn.LayerNorm) or norm.norm.elementwise_affine
            or norm.norm.eps != 1e-6 or not isinstance(norm.silu,torch.nn.SiLU)):
        raise ValueError("two-pass readout profile differs")
    binding["readout"] = {"parameters":bank({"modulation_weight":norm.linear.weight,"modulation_bias":norm.linear.bias,
        "projection_weight":model.proj_out.weight,"projection_bias":model.proj_out.bias}),
        "native_parameters":{name:tensor_descriptor(value) for name,value in norm.named_parameters()},
        "projection_native_parameters":{name:tensor_descriptor(value) for name,value in model.proj_out.named_parameters()},
        "eps":norm.norm.eps,"elementwise_affine":False}
    binding["pass_schedules"] = {str(index):{"timestep":array(p["timesteps"][index]),"sigma":array(p["sigmas"][index]),
        "sigma_next":array(p["sigmas"][index+1]),"next_clock":{"step":index+1,"phase":"project","cursor":0},
        "reset":dict.fromkeys(("image","text","context","noise_prediction"))} for index in LOGICAL_STEPS}
    if production_profile and (tuple(p["latent"].shape) != (1,1024,128)
            or tuple(p["conditioning"].shape) != (1,512,7680)
            or tuple(p["img_ids"].shape[-2:]) != (1024,4) or tuple(p["txt_ids"].shape[-2:]) != (512,4)):
        raise ValueError("two-pass qualified source geometry differs")
    return binding


def context_descriptors(value):
    return prior()["context_descriptors"](value)


def array_storage_profile(value):
    return prior()["array_storage_profile"](value)


def predict_two_pass(binding,*,archive_root,transition,boundary,reentry,config,event):
    import numpy as np

    from mdb_lanes.tensor_archive import load_tensor_archive

    def load(record):
        return None if record is None else load_tensor_archive(archive_root,record)
    latent = load(binding["inputs"]["latent"])
    conditioning,img_ids,txt_ids = (load(binding["inputs"][key]) for key in ("conditioning","img_ids","txt_ids"))
    passes = []
    for logical in LOGICAL_STEPS:
        schedule = binding["pass_schedules"][str(logical)]
        spec = binding["project"]
        parameters = {name:load(value) for name,value in spec["parameters"].items()}
        project = reentry["predict_project"](latent,conditioning,img_ids,txt_ids,load(schedule["timestep"]),parameters,
            axes_dims_rope=spec["axes_dims_rope"],rope_theta=spec["rope_theta"],
            timestep_channels=spec["timestep_channels"],projection_chunk=config["projection_chunk"])
        del parameters
        event("project_forecast_complete",logical_step=logical)
        ctx,current_image,current_text = project["context"],project["image"],project["text"]
        joints = {}
        for index in range(5):
            spec = binding["joint"+str(index)]
            parameters = {name:load(value) for name,value in spec["parameters"].items()}
            prediction = reentry["predict_joint"](current_image,current_text,ctx,parameters,
                heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
                query_chunk=config["query_chunk"],projection_chunk=config["projection_chunk"],retain_fields=False)
            del parameters
            joints[str(index)] = prediction
            current_image,current_text = prediction["image"],prediction["text"]
            event("joint_forecast_complete",logical_step=logical,block_index=index)
            gc.collect()
        current = np.concatenate((current_text,current_image),axis=1)
        current.setflags(write=False)
        singles = []
        for index in BLOCK_INDICES:
            spec = binding["final_blocks"][str(index)]
            parameters = {name:load(value) for name,value in spec["parameters"].items()}
            prediction = transition["predict_transition"](current,ctx["single_mod"],**parameters,
                heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
                rope_cos=ctx["rope"][0],rope_sin=ctx["rope"][1],query_chunk=config["query_chunk"],
                projection_chunk=config["projection_chunk"],neuron_chunk=config["neuron_chunk"],
                retain_fields=False,text_rows=None,sample_indices=config["sample_indices"])
            del parameters
            singles.append({"step":index,"block_index":index,
                "input_source":"own complete predicted joint4 I/T" if index == 0 else "previous complete predicted output",
                "input_descriptor":prediction["input_descriptor"],"prediction":prediction})
            current = prediction["output"]
            event("single_forecast_complete",logical_step=logical,block_index=index)
            gc.collect()
        readout = boundary["predict_readout"](current[:,binding["text_rows"]:],ctx["temb"],
            **{name:load(value) for name,value in binding["readout"]["parameters"].items()},
            eps=binding["readout"]["eps"],neuron_chunk=config["projection_chunk"],
            sample_indices=config["image_sample_indices"],retain_fields=False)
        commit = boundary["predict_flow_commit"](latent,readout["noise"],load(schedule["sigma"]),load(schedule["sigma_next"]))
        event("readout_commit_forecast_complete",logical_step=logical)
        passes.append({"logical_step":logical,"input_source":"observed prefix latent" if logical == 2 else "own previous pass commit latent",
            "input_latent_descriptor":step()["_descriptor"](latent),"project":project,"joints":joints,
            "single_steps":singles,"readout":readout,"commit":commit,"chain_length":28,
            "context_source":"own predicted project context","context_input_descriptors":context_descriptors(ctx),
            "next_state_contract":{"clock":schedule["next_clock"],"reset":schedule["reset"],"carried_state":binding["carried_state"]}})
        latent = commit["latent"]
    result = {"passes":passes,"chain_length":56,
        "scope":"Two complete own-state passes; every project regenerates context from its own incoming latent and observed conditioning/IDs/schedule. No native future input."}
    event("two_pass_forecast_storage",**array_storage_profile(result))
    return result


def validate_prediction(pre_project,prediction):
    import numpy as np

    normal,descriptor = util()["json_value"],step()["_descriptor"]
    if prediction["chain_length"] != 56 or len(prediction["passes"]) != 2:
        raise ValueError("two-pass forecast length differs")
    latent = pre_project.payload["latent"].float().cpu().numpy()
    for logical,row in zip(LOGICAL_STEPS,prediction["passes"],strict=True):
        project,ctx = row["project"],row["project"]["context"]
        if (row["logical_step"] != logical or row["chain_length"] != 28
                or row["input_source"] != ("observed prefix latent" if logical == 2 else "own previous pass commit latent")
                or normal(row["input_latent_descriptor"]) != descriptor(latent)
                or normal(project["input_latent_descriptor"]) != descriptor(latent)
                or normal(row["context_input_descriptors"]) != context_descriptors(ctx)):
            raise ValueError("two-pass own latent/context input differs")
        image,text = project["image"],project["text"]
        for index in range(5):
            joint = row["joints"][str(index)]
            if (normal(joint["input_image_descriptor"]) != descriptor(image)
                    or normal(joint["input_text_descriptor"]) != descriptor(text)
                    or normal(joint["image_descriptor"]) != descriptor(joint["image"])
                    or normal(joint["text_descriptor"]) != descriptor(joint["text"])):
                raise ValueError("two-pass joint own-state join differs")
            image,text = joint["image"],joint["text"]
        current = np.concatenate((text,image),axis=1)
        if len(row["single_steps"]) != 20:
            raise ValueError("two-pass single census differs")
        for index,entry in enumerate(row["single_steps"]):
            value = entry["prediction"]
            expected = descriptor(current)
            if (entry["block_index"] != index or entry["step"] != index
                    or normal(entry["input_descriptor"]) != expected or normal(value["input_descriptor"]) != expected
                    or normal(value["output_descriptor"]) != descriptor(value["output"])
                    or value["output"].shape != current.shape or "image" in value or "text" in value):
                raise ValueError("two-pass single own-state join differs")
            current = value["output"]
        readout,commit = row["readout"],row["commit"]
        if (normal(readout["input_descriptor"]) != descriptor(current[:,text.shape[1]:])
                or normal(readout["noise_descriptor"]) != descriptor(readout["noise"])
                or normal(commit["input_descriptor"]) != descriptor(latent)
                or normal(commit["noise_descriptor"]) != descriptor(readout["noise"])
                or normal(commit["latent_descriptor"]) != descriptor(commit["latent"])
                or normal(row["next_state_contract"]["clock"]) != {"step":logical+1,"phase":"project","cursor":0}
                or normal(row["next_state_contract"]["reset"]) != dict.fromkeys(("image","text","context","noise_prediction"))):
            raise ValueError("two-pass readout/commit/reset join differs")
        latent = commit["latent"]


def parameter_census(components):
    return util()["identity"]({str(index):{name:{"id":id(value),"version":value._version,
        "shape":list(value.shape),"dtype":str(value.dtype),"device":str(value.device)}
        for name,value in component.named_parameters()} for index,component in enumerate(components)})


def run_panel(*,adapter,pre_project,output,config,store,array,counts,wall,event,prediction,phase_control):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb_lanes.flux_derived_terminal import derive_terminal
    from mdb_lanes.flux_final_latent_closure import closure_act
    from mdb_lanes.native_navigation import NativeCutNavigator

    helpers = util()
    def navigator(cut,branch,limit):
        return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=limit,
            on_receipt=lambda value:receipt(output,store,counts,branch,value))
    def archived(value):
        if isinstance(value,torch.Tensor):
            return array(value)
        if isinstance(value,dict):
            return {key:archived(child) for key,child in value.items()}
        if isinstance(value,(tuple,list)):
            return [archived(child) for child in value]
        return value
    def boundary(cut):
        return {"cut":cut.fingerprint,"point":dict(cut.execution_point),
            "closure":{key:archived(cut.payload[key]) for key in ("latent","context","image","text","noise_prediction")},
            "clock":{key:cut.payload[key] for key in ("step","phase","cursor")}}
    def terminal(cut):
        if (cut.payload["step"] != 4 or cut.payload["phase"] != "project" or cut.payload["cursor"] != 0
                or any(cut.payload[key] is not None for key in ("context","image","text","noise_prediction"))):
            raise ValueError("two-pass terminal reset differs")
    def tensor(value,reference):
        if not isinstance(value,np.ndarray) or value.dtype != np.float32 or tuple(value.shape) != tuple(reference.shape) or not np.isfinite(value).all():
            raise ValueError("two-pass sealed payload geometry differs")
        if reference.dtype == torch.bfloat16:
            step()["_descriptor"](value)
        result = torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
        result.copy_(torch.from_numpy(np.array(value,copy=True)).to(device=reference.device,dtype=reference.dtype))
        if not np.array_equal(result.float().cpu().numpy().view(np.uint32),value.view(np.uint32)):
            raise ValueError("two-pass admission casts sealed values")
        return result
    def context(value,reference):
        if isinstance(reference,torch.Tensor):
            return tensor(value,reference)
        if isinstance(reference,dict) and set(value) == set(reference):
            return {key:context(value[key],reference[key]) for key in reference}
        if isinstance(reference,(tuple,list)) and len(value) == len(reference):
            return tuple(context(left,right) for left,right in zip(value,reference,strict=True))
        raise ValueError("two-pass context geometry differs")
    def act_record(nav,act,payload):
        before = nav.capture()
        entry = nav.apply(act)
        installed = nav.capture()
        store.save(installed)
        if describe(installed.payload) != describe({**before.payload,**payload}):
            raise ValueError("two-pass atomic admission changed undeclared state")
        return {"slots":["latent","context","image","text"],"parent":before.fingerprint,
            "post_act":installed.fingerprint,"receipt":entry.to_dict(),"receipt_id":entry.fingerprint,
            "act":act.manifest(),"undeclared_slots_exact":True},installed
    def finish(row,nav):
        final = nav.capture()
        terminal(final)
        store.save(final)
        row["terminal_final"] = boundary(final)
        counts["vae_calls"] += 1
        image = adapter.decode(nav.session,height=512,width=512)
        path = output/(row["id"]+".png")
        image.save(path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint,final_latent=array(final.payload["latent"]),
            image={"path":path.name,"sha256":sha(path)})
        helpers["save"](output/"rows"/(row["id"]+".json"),row)
        event("branch_complete",row=row["id"])
        return row
    validate_prediction(pre_project,prediction)
    # This non-native derivation creates no continuation or application receipt.
    predicted = tensor(prediction["passes"][1]["commit"]["latent"],pre_project.payload["latent"])
    derived = derive_terminal(adapter,pre_project,predicted_latent=predicted,name="full.predicted.complete-closure")
    store.save(derived.scaffold)
    helpers["save"](output/"derived-terminal.json",derived.derivation)
    # Navigator requires a positive step allowance; this terminal arm calls no step.
    nav = navigator(derived.scaffold,"full.predicted",1)
    payload = {"latent":predicted,"context":None,"image":None,"text":None}
    action,installed = act_record(nav,derived.action,payload)
    row = {"id":"full.predicted","mode":"full","parent":derived.scaffold.fingerprint,
        "entry":boundary(derived.scaffold),"derivation":{"path":"derived-terminal.json","sha256":sha(output/"derived-terminal.json")},
        "acts":[action],"installed":boundary(installed),"bridge_receipts":[],"bridge_records":[],
        "bridge_boundaries":[],"continuation_receipts":[]}
    phase_control["phase"] = "predicted_vae"
    full = finish(row,nav)
    phase_control["verify_first_image"]()
    event("predicted_image_complete_before_native_reference",counts=dict(counts),
        target_entries=dict(phase_control["targets"]),image=full["image"])
    helpers["save"](output/"first-predicted-image.json",{"counts":dict(counts),"target_entries":dict(phase_control["targets"]),
        "image":full["image"],"source_cut":pre_project.fingerprint,"source_unchanged":True,"parameters_unchanged":True,
        "derivation_sha256":sha(output/"derived-terminal.json"),"scope":"Actual native VAE PNG saved before any selected native transformer target."})
    phase_control["phase"] = "native"
    nav = navigator(pre_project,"reference",56)
    references,paths,cuts = [],[],[]
    for position in range(56):
        wall()
        entry = nav.step()
        current = nav.capture()
        store.save(current)
        references.append({"index":position,"receipt_id":entry.fingerprint,**boundary(current)})
        paths.append(entry.to_dict())
        cuts.append(current)
        event("reference_transition_complete",index=position,point=dict(current.execution_point))
    common,native_final = cuts[33],cuts[55]
    terminal(native_final)
    if any(common.execution_point[key] != value for key,value in {"logical_step":3,"phase":"single","index":0,"local_clock":0}.items()):
        raise ValueError("two-pass common post-joint4 clock differs")
    helpers["save"](output/"native-segment.json",{"input_cut":pre_project.fingerprint,"boundaries":references,"receipts":paths,
        "passes":[{"logical_step":logical,"project":references[offset],
            "joints":{str(index):references[offset+1+index] for index in range(5)},
            "single_steps":references[offset+6:offset+26],"readout":references[offset+26],"commit":references[offset+27]}
            for logical,offset in ((2,0),(3,28))],
        "common_post_joint4":boundary(common),"native_final":boundary(native_final),
        "counts":{"adapter_steps":56,"acts":0,"receipt_emissions":56}})
    del cuts
    def observed(record,index,incoming):
        return {**record,"index":index,"incoming_receipt":incoming,"complete_cut_retained":True,
            "scope":"Complete live arrays and typed cut; stored latent constant until actual commit, working fields resetNone at terminal."}
    trajectories = {"native":[observed(value,index,paths[33+index]["fingerprint"])
        for index,value in enumerate(references[33:56])]}
    rows = {"full.predicted":full}
    for name in ORDER[:-1]:
        mode = next(spec["mode"] for spec in BRANCHES if spec["id"] == name)
        parent = common if mode == "bridge" else native_final
        nav = navigator(parent,name,22)
        row = {"id":name,"mode":mode,"parent":parent.fingerprint,"entry":boundary(parent),
            "acts":[],"bridge_receipts":[],"bridge_records":[],"bridge_boundaries":[],"continuation_receipts":[]}
        if mode != "native":
            before = nav.capture()
            if mode == "sham":
                payload = {key:before.payload[key] for key in ("latent","context","image","text")}
            else:
                second = prediction["passes"][1]
                payload = {"latent":tensor(prediction["passes"][0]["commit"]["latent"],before.payload["latent"]),
                    "context":context(second["project"]["context"],before.payload["context"]),
                    **{key:tensor(second["joints"]["4"][key],before.payload[key]) for key in ("image","text")}}
            action,installed = act_record(nav,closure_act(before,**payload,name=name+".complete-closure"),payload)
            row["acts"].append(action)
            row["installed"] = boundary(installed)
        if mode == "bridge":
            row["common_post_joint4"] = boundary(nav.capture())
            trajectory = [observed(row["common_post_joint4"],0,row["acts"][-1]["receipt_id"])]
            for index in range(22):
                wall()
                before = nav.capture()
                entry = nav.step()
                current = nav.capture()
                store.save(current)
                after_record = boundary(current)
                row["bridge_receipts"].append(entry.to_dict())
                row["bridge_boundaries"].append(after_record)
                row["bridge_records"].append({"index":index,"before":boundary(before),"after":after_record,
                    "receipt_id":entry.fingerprint,"receipt":entry.to_dict()})
                trajectory.append(observed(after_record,index+1,entry.fingerprint))
            trajectories[name] = trajectory
        rows[name] = finish(row,nav)
    helpers["save"](output/"continuation-observations.json",{"schema":"mdb-flux-two-pass-continuation-v1",
        "plan":config["observation_plan"],"rows":trajectories,"branch_execution_order":["native","bridge.predicted"],"model_calls_added":0})
    controls = {"sham_final_latent_exact_native":rows["native"]["final_latent"] == rows["sham"]["final_latent"],
        "sham_png_exact_native":rows["native"]["image"]["sha256"] == rows["sham"]["image"]["sha256"]}
    return {"rows":[rows[name] for name in ORDER],"counts":dict(counts),"control_observations":controls,
        "branch_execution_order":EXECUTION_ORDER,"first_predicted_image_sha256":sha(output/"first-predicted-image.json"),
        "derivation_sha256":sha(output/"derived-terminal.json"),"native_segment_sha256":sha(output/"native-segment.json"),
        "continuation_observations_sha256":sha(output/"continuation-observations.json")}


def run(*,model,adapter,session,output,config,request,source,counts,array=None,guard_components=None,
        wall=lambda:None,event=lambda *args,**kwargs:None,started=None,production_profile=True):
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    started = time.monotonic() if started is None else started
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    helpers = util()
    save,identity = helpers["save"],helpers["identity"]
    if (output/"plan.json").exists():
        raise ValueError("immutable two-pass output already contains a plan")
    array = array or step()["make_array_writer"](output,wall)
    save(output/"plan.json",config)
    save(output/"plan-frozen.json",{"plan_sha256":sha(output/"plan.json"),"job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_identity":identity(request["source_hashes"]),"counts":dict(counts),
        "scope":"Frozen before clean56transitionprefix; all56 own-state stages seal before native selected targets."})
    boundary = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    abi = verify_source_abi(boundary=boundary,reentry=reentry,closure_type=type(adapter),
        expected_derived_terminal_module_sha256=config["expected_derived_terminal_module_sha256"])
    if adapter.execution.get("adapter") != "mdb-flux-final-latent-closure-v1":
        raise ValueError("two-pass requires its distinct live single0 and terminal reset closure contract")
    save(output/"native-source-abi.json",abi)
    store = LocalStore(output/"typed-store")
    pre_project,prefix = run_prefix(adapter=adapter,session=session,output=output,store=store,counts=counts,wall=wall)
    prefix_ref = {"path":"prefix.json","sha256":sha(output/"prefix.json")}
    source = {**source,"prefix":prefix_ref,"native_source_abi":{"path":"native-source-abi.json","sha256":sha(output/"native-source-abi.json")}}
    save(output/"source.json",source)
    transition = runpy.run_path(str(Path(__file__).with_name("flux_transition_reference.py")))
    components = tuple(guard_components or (model,))
    source_fingerprint = identity(describe(pre_project.payload))
    original_parameters = parameter_census(components)
    released = {"phase":"forecast","blocked":0,"targets":{key:0 for key in contract()["target_calls"]}}
    vae_modules = {id(module) for component in components if component is not model for module in component.modules()}
    targets = {model.single_transformer_blocks[index]:str(index) for index in BLOCK_INDICES}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out",model.pos_embed:"pos_embed",
        **{model.transformer_blocks[index]:"joint"+str(index) for index in range(5)},
        **{getattr(model,name):name for name in PROJECT_TARGETS}})
    def guard(module,args):
        if released["phase"] != "native" and not (released["phase"] == "predicted_vae" and id(module) in vae_modules):
            released["blocked"] += 1
            raise ValueError("native component entered before whole two-pass forecast seal")
        if module in targets:
            released["targets"][targets[module]] += 1
    def verify_first_image():
        if (identity(describe(pre_project.payload)) != source_fingerprint or parameter_census(components) != original_parameters
                or any(released["targets"].values()) or released["blocked"] != 0
                or any(counts[key] != value for key,value in {"adapter_steps":56,"acts":1,"images":1,"vae_calls":1,"receipt_emissions":57}.items())):
            raise ValueError("two-pass first predicted image source/parameter/zero-target contract differs")
    released["verify_first_image"] = verify_first_image
    handles = [module.register_forward_pre_hook(guard) for component in components for module in component.modules()]
    try:
        binding = bind_inputs(model=model,pre_project=pre_project,array=array,counts=counts,prefix_ref=prefix_ref,
            config=config,production_profile=production_profile)
        save(output/"binding.json",binding)
        tx = ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,
            input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
            frontier={"kind":"architectural","binding":binding},execution_scope=config["scope"],
            budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000,"max_bytes":config["transaction_max_bytes"]})
        with tx:
            tx.frontier("source_binding",binding)
            event("prediction_start")
            tx.predict("two_pass",lambda _:{"predictions":predict_two_pass(binding,archive_root=output,
                transition=transition,boundary=boundary,reentry=reentry,config=config,event=event)})
            tx.predict("hypotheses",lambda _:{"predictions":config["hypotheses"]})
            seal = tx.seal()
            event("two_pass_sealed",seal=seal,target_entries_before_seal=dict(released["targets"]),blocked_entries=released["blocked"])
            tx.execute("native",lambda context:run_panel(adapter=adapter,pre_project=pre_project,output=output,config=config,
                store=store,array=array,counts=counts,wall=wall,event=event,prediction=context.predictions["two_pass"],phase_control=released))
    finally:
        for handle in handles:
            handle.remove()
    save(output/"transaction-report.json",tx.report())
    if (any(counts[name] != COUNTS[name] for name in ("adapter_steps","acts","receipt_emissions","images","vae_calls","candidate_calls"))
            or released["blocked"] != 0 or released["targets"] != contract()["target_calls"]):
        raise ValueError("two-pass completed physical/target census differs")
    summary = {"schema":"mdb-flux-two-pass-result-v1","job_id":request["job_id"],"request_sha256":request["request_sha256"],
        "counts":dict(counts),"prefix_counts":prefix["counts"],"panel_counts":contract()["panel_counts"],
        "target_calls":released["targets"],"blocked_entries":released["blocked"],"seal":seal,
        "model_identity":adapter.model_identity,"execution":dict(adapter.execution),"transaction_verification":verify_transaction(tx.root),
        "control_observations":tx.observation("native")["control_observations"],"worker_seconds":time.monotonic()-started,
        "mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary)
    return helpers["json_value"](summary)


def worker():
    """One admitted CUDA process; SSH custody is performed by the launcher."""
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90,repeat=True)
    helpers = util()
    save,sha,identity,json_value = (helpers[name] for name in ("save","sha","identity","json_value"))
    config,request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if (request["config"] != config or any(sha(path) != digest for path,digest in request["source_hashes"].items())
            or config.get("transport_method") != "planned-authenticated-ssh-manifest"):
        raise ValueError("two-pass request/source/transport identity differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path("flux-two-pass-result")
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_FLUX_TWO_PASS_BOOTSTRAP "+json.dumps({"job_id":request["job_id"],"seconds":time.monotonic()-started}),flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb_lanes import flux_derived_terminal
    from mdb_lanes.flux_final_latent_closure import FluxFinalLatentClosureAdapter
    from mdb_lanes.flux_forecast_closure import FluxForecastClosureAdapter
    from mdb_lanes.flux_joint_forecast_closure import FluxJointForecastClosureAdapter
    from mdb_lanes.flux_joint_tail_closure import FluxJointTailClosureAdapter

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("two-pass requires admitted CUDA")
    reference = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    source_abi = verify_source_abi(boundary=reference,reentry=reentry,closure_type=FluxFinalLatentClosureAdapter,
        expected_derived_terminal_module_sha256=config["expected_derived_terminal_module_sha256"])
    save(output / "pre-model-source-abi.json",source_abi)
    scalar_qualification = reference["qualify_torch_scalar_promotion"](torch,device="cuda")
    save(output / "cuda-scalar-qualification.json",scalar_qualification)
    disk = shutil.disk_usage(output)
    if disk.free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("two-pass output filesystem lacks declared14GiB free disk")
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
        print("MDB_FLUX_TWO_PASS_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        now = time.monotonic()
        if now-started > config["max_wall_seconds"]:
            raise RuntimeError("finite two-pass wall budget exhausted")
        if now-heartbeat["last"] >= 15:
            heartbeat["last"] = now
            event("heartbeat")
    array = step()["make_array_writer"](output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("two-pass checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("two-pass requires deterministic distilled five-joint/twenty-single Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind):sha(inspect.getfile(kind)) for kind in (type(pipe),type(pipe.text_encoder),
        type(pipe.transformer),type(pipe.transformer.single_transformer_blocks[6]),
        type(pipe.transformer.single_transformer_blocks[6].attn.processor),type(pipe.transformer.norm_out),
        type(pipe.vae),type(pipe.scheduler),Flux2KleinAdapter,FluxForecastClosureAdapter,FluxJointForecastClosureAdapter,FluxJointTailClosureAdapter,FluxFinalLatentClosureAdapter,
        type(pipe.transformer.transformer_blocks[0]),type(pipe.transformer.pos_embed),
        type(pipe.transformer.time_guidance_embed.timestep_embedder))}
    implementations[inspect.getfile(flux_derived_terminal)] = sha(inspect.getfile(flux_derived_terminal))
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
    adapter = FluxFinalLatentClosureAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    with torch.inference_mode():
        conditioning,txt_ids = conditioning.to("cuda"),txt_ids.to("cuda")
        latent,img_ids = pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,conditioning.dtype,
            torch.device("cuda"),torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
        timesteps,steps = retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
        if steps != 4:
            raise ValueError("two-pass native schedule differs")
        session = adapter.session(latent=latent,conditioning=conditioning,img_ids=img_ids,txt_ids=txt_ids,
            timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
    source = {"schema":"mdb-flux-two-pass-source-v1","job_id":request["job_id"],
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
        raise ValueError("two-pass global load/encoding/noise census differs")
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    manifest = step()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"],
        max_expanded_bytes=config["transport_limits"]["max_expanded_bytes"],
        max_member_bytes=config["transport_limits"]["max_member_bytes"])
    print("MDB_FLUX_TWO_PASS_COMPLETE "+json.dumps(json_value({"summary":summary,"manifest":manifest}),sort_keys=True),flush=True)
    return output,summary


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-two-pass-result")
        if output.is_dir() and not (output/"remote-manifest.json").exists():
            util()["save"](output/"failure.json",{"schema":"mdb-flux-two-pass-failure-v1",
                "job_id":os.environ.get("MRUN_JOB_ID"),"error":f"{type(error).__name__}: {error}",
                "mechanics_status":"errored","scope":"Completed evidence preserved; no model retry, archive or fabricated report."})
        raise
