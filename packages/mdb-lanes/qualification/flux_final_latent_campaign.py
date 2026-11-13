"""Seal own-state 44-stage full final-latent prediction, then native futures.

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

CONTEXT = {"id":"p1-s107","prompt_id":"p1","seed":107,
    "prompt":"A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BLOCK_INDICES = list(range(6,20))
FINAL_BLOCK_INDICES = list(range(20))
BRANCHES = [{"id":"native","mode":"native"},{"id":"sham","mode":"sham"},
    {"id":"bridge.predicted","mode":"bridge"},{"id":"full.predicted","mode":"full"}]
ORDER = [row["id"] for row in BRANCHES]
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
LEGACY_SOURCES = {
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
    return runpy.run_path(str(Path(__file__).with_name("flux_joint_tail_campaign.py")))


def step():
    return prior()["step"]()


def util():
    return prior()["util"]()


def contract(config=None):
    if config is not None and config.get("block_indices",BLOCK_INDICES) != BLOCK_INDICES:
        raise ValueError("final-latent requires the declared14-block remaining step")
    return json.loads(json.dumps({"context":CONTEXT,"block_indices":BLOCK_INDICES,"final_block_indices":FINAL_BLOCK_INDICES,"branches":BRANCHES,
        "order":ORDER,"counts":COUNTS,"prefix_steps":68,"reference_steps":44,"bridge_steps":22,"suffix_steps":0,
        "panel_counts":{"adapter_steps":66,"acts":3,"receipt_emissions":69,"candidate_calls":0,"images":4,"vae_calls":4},
        "target_calls":{**{str(index):(2 if index < 6 else 3) for index in FINAL_BLOCK_INDICES},"norm_out":3,"proj_out":3,
            **{name:1 for name in PROJECT_TARGETS},"pos_embed":2,"joint0":1,"joint1":1,"joint2":1,"joint3":1,"joint4":1},"observation_plan":OBSERVATION_PLAN}))


def validate_config(config):
    fixed = {"context":CONTEXT,"block_indices":BLOCK_INDICES,"final_block_indices":FINAL_BLOCK_INDICES,"branches":BRANCHES,"branch_execution_order":ORDER,
        "selector":{"logical_step":2,"phase":"single","index":6},"height":512,"width":512,"steps":4,
        "sample_indices":SAMPLE_INDICES,"image_sample_indices":IMAGE_SAMPLE_INDICES,"query_chunk":64,
        "projection_chunk":256,"neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits",
        "archive_integer_ids":"native-int64","cpu_threads":4,"transaction_max_bytes":1024<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":12<<30,"max_member_bytes":1<<30},"observation_plan":OBSERVATION_PLAN,
        "expected_closure_module_sha256":OLD_CLOSURE_SHA256,"expected_joint_closure_module_sha256":PRIOR_JOINT_CLOSURE_SHA256,
        "expected_tail_closure_module_sha256":PRIOR_TAIL_CLOSURE_SHA256}
    if any(config.get(key) != value for key,value in fixed.items()):
        raise ValueError("final-latent frozen scientific/closure geometry differs")
    for key in ("adapter_steps","acts","candidate_calls","images","vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("final-latent finite physical census differs")
    digest = config.get("expected_final_closure_module_sha256")
    if not isinstance(digest,str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("final-latent requires an explicit new joint closure module pin")
    if type(config.get("cpu_threads")) is not int or type(config.get("max_wall_seconds")) is not int or not 1<=config["max_wall_seconds"]<=1050:
        raise ValueError("final-latent finite execution budget differs")


def verify_source_abi(*,boundary,reentry,closure_type,expected_final_closure_module_sha256):
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb_lanes.flux_joint_tail_closure import FluxJointTailClosureAdapter

    old = prior()["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=FluxJointTailClosureAdapter,
        expected_tail_closure_module_sha256=PRIOR_TAIL_CLOSURE_SHA256)
    if (not issubclass(closure_type,Flux2KleinAdapter) or closure_type.advance is not Flux2KleinAdapter.advance
            or sha(inspect.getfile(closure_type)) != expected_final_closure_module_sha256):
        raise ValueError("final-latent native/base/new closure source ABI differs")
    return {"schema":"mdb-flux-final-latent-native-source-abi-v1","verified":True,"prior_joint_tail":old,
        "final_closure_module_sha256":expected_final_closure_module_sha256,
        "final_closure_type":closure_type.__module__+"."+closure_type.__qualname__,"native_advance_inherited":True,
        "scope":"Pinned base2004/old45dd/jointbb352/tail2788 and separately identified final latent closure surface; native advancement and numerical reference sources unchanged."}


def receipt(output,store,counts,branch,value):
    body = value.to_dict()
    counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps",1)
    counts["receipt_emissions"] += 1
    if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
        raise RuntimeError("final-latent finite physical budget exhausted")
    store.receipt(value)
    with (output/"receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps({"sequence":counts["receipt_emissions"]-1,"branch":branch,"receipt":value.fingerprint},sort_keys=True)+"\n")


def run_prefix(*,adapter,session,output,store,counts,wall):
    from mdb_lanes.native_navigation import NativeCutNavigator

    initial = session.capture()
    if initial.execution_point["logical_step"] != 0 or initial.execution_point["phase"] != "project":
        raise ValueError("final-latent requires a clean newly encoded/seeded native initial state")
    store.save(initial)
    nav = NativeCutNavigator(session,max_steps=68,on_receipt=lambda value:receipt(output,store,counts,"prefix",value))
    path = []
    for _ in range(68):
        wall()
        path.append(nav.step().to_dict())
    pre6 = nav.capture()
    if any(pre6.execution_point[key] != value for key,value in {"logical_step":2,"phase":"single","index":6,"local_clock":6}.items()):
        raise ValueError("final-latent prefix did not reach step2/pre6")
    store.save(pre6)
    record = {"schema":"mdb-flux-final-latent-prefix-v1","context":CONTEXT,"initial":initial.fingerprint,
        "pre6":pre6.fingerprint,"point":dict(pre6.execution_point),"receipts":path,
        "counts":{"adapter_steps":68,"acts":0,"receipt_emissions":68},"scope":"One clean seed107 history; no imported/engineered state."}
    util()["save"](output/"prefix.json",record)
    return pre6,record



def bind_inputs(*,model,pre6,array,counts,prefix_ref,config,production_profile):
    import torch

    from mdb_lanes.forward_recording import tensor_descriptor

    binding = prior()["bind_inputs"](model=model,pre6=pre6,array=array,counts=counts,prefix_ref=prefix_ref,
        config=config,production_profile=production_profile)
    binding["schema"] = "mdb-flux-final-latent-binding-v1"
    binding["target_entries_before_seal"] = {key:0 for key in contract()["target_calls"]}
    binding["final_blocks"] = {key:{**value,"block_index":int(key),
        "native_parameter_prefix":f"single_transformer_blocks.{key}."} for key,value in binding["blocks"].items()}
    profile = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))["attention_profile"]
    def archive(value):
        if value is None:
            return None
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("final-step parameters must be finite native BF16")
        return array(value)
    for index in range(6):
        block = model.single_transformer_blocks[index]
        attn = block.attn
        h,m = attn.inner_dim,attn.mlp_hidden_dim
        if (block.norm.elementwise_affine or block.norm.eps != 1e-6 or attn.norm_q.eps != attn.norm_k.eps
                or not attn.norm_q.elementwise_affine or not attn.norm_k.elementwise_affine
                or attn.norm_q.eps != 1e-6 or h != attn.heads*attn.head_dim
                or tuple(attn.to_qkv_mlp_proj.weight.shape) != (3*h+2*m,h)
                or tuple(attn.to_out.weight.shape) != (h,h+m)):
            raise ValueError("final-step complete block profile differs")
        if production_profile and (h,m,attn.heads,attn.head_dim) != (3072,9216,24,128):
            raise ValueError("qualified final-step source geometry differs")
        parameters = {"qkv_mlp_weight":attn.to_qkv_mlp_proj.weight,"qkv_mlp_bias":attn.to_qkv_mlp_proj.bias,
            "to_out_weight":attn.to_out.weight,"to_out_bias":attn.to_out.bias,
            "norm_q_weight":attn.norm_q.weight,"norm_k_weight":attn.norm_k.weight}
        binding["final_blocks"][str(index)] = {"parameters":{name:archive(value) for name,value in parameters.items()},
            "native_parameters":{name:tensor_descriptor(value) for name,value in block.named_parameters()},
            "block_index":index,"native_parameter_prefix":f"single_transformer_blocks.{index}.",
            "heads":attn.heads,"hidden_width":h,"mlp_width":m,"head_dim":attn.head_dim,
            "layernorm_eps":block.norm.eps,"qk_eps":attn.norm_q.eps,"attention_profile":profile(block)}
    binding["final_commit"] = {"step_index":3,"sigma":array(pre6.payload["sigmas"][3]),
        "sigma_next":array(pre6.payload["sigmas"][4]),"latent_input_source":"own predicted step2 commit Z3",
        "reset":{"image":None,"text":None,"context":None,"noise_prediction":None},
        "next_clock":{"step":4,"phase":"project","cursor":0},
        "scope":"Terminal latent predicted from own Z3 and own final readout; all working slots resetNone, other carried state preserved."}
    return binding


def context_descriptors(value):
    """Exact full project-context identity, including legitimate FP32 RoPE."""
    from collections.abc import Mapping

    import numpy as np

    if isinstance(value,np.ndarray):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("predicted project context must be finite FP32")
        return {"shape":list(value.shape),"dtype":"float32",
            "sha256":hashlib.sha256(np.ascontiguousarray(value).astype("<f4",copy=False).tobytes()).hexdigest()}
    if isinstance(value,Mapping):
        return {key:context_descriptors(child) for key,child in value.items()}
    if isinstance(value,(tuple,list)):
        return [context_descriptors(child) for child in value]
    raise ValueError("predicted project context structure differs")


def array_storage_profile(value):
    """Measure deduplicated numerical NPY geometry without building large buffers."""
    import io

    import numpy as np

    seen = set()
    profile = {"array_occurrences":0,"unique_arrays":0,"payload_bytes":0,"npy_bytes":0}
    def visit(node):
        if isinstance(node,np.ndarray):
            profile["array_occurrences"] += 1
            digest = hashlib.sha256(np.ascontiguousarray(node).tobytes()).hexdigest()
            key = node.dtype.str,node.shape,digest
            if key not in seen:
                seen.add(key)
                header = io.BytesIO()
                np.lib.format.write_array_header_1_0(header,{"descr":np.lib.format.dtype_to_descr(node.dtype),
                    "fortran_order":False,"shape":node.shape})
                profile["unique_arrays"] += 1
                profile["payload_bytes"] += node.nbytes
                profile["npy_bytes"] += node.nbytes+header.tell()
        elif isinstance(node,dict):
            for child in node.values():
                visit(child)
        elif isinstance(node,(tuple,list)):
            for child in node:
                visit(child)
    visit(value)
    return {**profile,"scope":"Exact unique-array NPY geometry under dtype/shape/value deduplication; excludes bundle magic and JSON. Transaction independently enforces total retained bytes."}


def predict_final_latent(binding,*,archive_root,transition,boundary,reentry,config,event):
    import numpy as np

    from mdb_lanes.tensor_archive import load_tensor_archive

    prefix = prior()["predict_tail"](binding,archive_root=archive_root,transition=transition,
        boundary=boundary,reentry=reentry,config=config,event=event)
    project = prefix["prefix"]["prefix"]["project"]
    ctx = project["context"]
    current = np.concatenate((prefix["joints"]["4"]["text"],prefix["joints"]["4"]["image"]),axis=1)
    current.setflags(write=False)
    def load(record):
        return None if record is None else load_tensor_archive(archive_root,record)
    steps = []
    for index in FINAL_BLOCK_INDICES:
        spec = binding["final_blocks"][str(index)]
        parameters = {name:load(value) for name,value in spec["parameters"].items()}
        prediction = transition["predict_transition"](current,ctx["single_mod"],**parameters,
            heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
            rope_cos=ctx["rope"][0],rope_sin=ctx["rope"][1],query_chunk=config["query_chunk"],
            projection_chunk=config["projection_chunk"],neuron_chunk=config["neuron_chunk"],
            retain_fields=False,text_rows=None,sample_indices=config["sample_indices"])
        del parameters
        steps.append({"step":index,"block_index":index,
            "input_source":"own complete predicted joint4 I/T" if index == 0 else "previous complete predicted output",
            "input_descriptor":prediction["input_descriptor"],"prediction":prediction})
        current = prediction["output"]
        event("final_single_forecast_complete",block_index=index)
        gc.collect()
    readout = boundary["predict_readout"](current[:,binding["text_rows"]:],ctx["temb"],
        **{name:load(value) for name,value in binding["readout"]["parameters"].items()},
        eps=binding["readout"]["eps"],neuron_chunk=config["projection_chunk"],
        sample_indices=config["image_sample_indices"],retain_fields=False)
    latent = prefix["prefix"]["prefix"]["remaining_step"]["commit"]["latent"]
    commit = boundary["predict_flow_commit"](latent,readout["noise"],
        load(binding["final_commit"]["sigma"]),load(binding["final_commit"]["sigma_next"]))
    event("final_readout_commit_forecast_complete")
    result = {"prefix":prefix,"final_step":{"steps":steps,"chain_length":20,
        "input_descriptor":steps[0]["input_descriptor"],"output_descriptor":steps[-1]["prediction"]["output_descriptor"],
        "context_source":"own predicted project context","context_input_descriptors":context_descriptors(ctx),
        "readout":readout,"commit":commit,"next_state_contract":{"clock":binding["final_commit"]["next_clock"],
            "reset":binding["final_commit"]["reset"],"carried_state":binding["carried_state"]},
        "scope":"Twenty final singles then readout/Euler consume own predicted joint4 fields, project context and Z3; no native future refresh."}}
    event("final_forecast_storage",**array_storage_profile(result))
    return result


def validate_prediction(pre6,prediction):
    import numpy as np

    prior()["validate_prediction"](pre6,prediction["prefix"])
    normal,descriptor = util()["json_value"],step()["_descriptor"]
    prefix,final = prediction["prefix"],prediction["final_step"]
    current = np.concatenate((prefix["joints"]["4"]["text"],prefix["joints"]["4"]["image"]),axis=1)
    expected = descriptor(current)
    ctx = prefix["prefix"]["prefix"]["project"]["context"]
    if (final["chain_length"] != 20 or len(final["steps"]) != 20 or normal(final["input_descriptor"]) != expected
            or final["context_source"] != "own predicted project context"
            or normal(final["context_input_descriptors"]) != context_descriptors(ctx)):
        raise ValueError("final-step complete own-state/context source differs")
    for index,entry in enumerate(final["steps"]):
        value = entry["prediction"]
        actual = descriptor(value["output"])
        if (entry["step"] != index or entry["block_index"] != index
                or entry["input_source"] != ("own complete predicted joint4 I/T" if index == 0 else "previous complete predicted output")
                or normal(entry["input_descriptor"]) != expected or normal(value["input_descriptor"]) != expected
                or normal(value["output_descriptor"]) != actual or value["output"].shape != current.shape
                or "image" in value or "text" in value):
            raise ValueError("final-step complete own-state recurrence differs")
        expected = actual
    readout,commit = final["readout"],final["commit"]
    latent = prefix["prefix"]["prefix"]["remaining_step"]["commit"]["latent"]
    if (normal(final["output_descriptor"]) != expected
            or normal(readout["input_descriptor"]) != descriptor(final["steps"][-1]["prediction"]["output"][:,pre6.payload["text"].shape[1]:])
            or normal(readout["noise_descriptor"]) != descriptor(readout["noise"])
            or normal(commit["noise_descriptor"]) != descriptor(readout["noise"])
            or normal(commit["input_descriptor"]) != descriptor(latent)
            or normal(commit["latent_descriptor"]) != descriptor(commit["latent"])
            or commit["latent"].shape != tuple(pre6.payload["latent"].shape)
            or normal(final["next_state_contract"]["clock"]) != {"step":4,"phase":"project","cursor":0}
            or normal(final["next_state_contract"]["reset"]) != {"image":None,"text":None,"context":None,"noise_prediction":None}):
        raise ValueError("final-step readout/terminal commit own-state join differs")


def run_panel(*,adapter,pre6,output,config,store,array,counts,wall,event,prediction):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

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
            raise ValueError("final-latent terminal commit/reset boundary differs")
    def observed(record,index,incoming):
        return {**record,"index":index,"incoming_receipt":incoming,"complete_cut_retained":True,
            "scope":"Full current arrays and typed cut retained; stored Z3 stays constant until commit; working arrays resetNone at terminal."}
    nav = navigator(pre6,"reference",44)
    references,paths,cuts = [],[],[]
    for position in range(44):
        wall()
        entry = nav.step()
        current = nav.capture()
        store.save(current)
        references.append({"index":position,"receipt_id":entry.fingerprint,**boundary(current)})
        paths.append(entry.to_dict())
        cuts.append(current)
        event("reference_transition_complete",index=position,point=dict(current.execution_point))
    common,native_final = cuts[21],cuts[43]
    terminal(native_final)
    if any(common.execution_point[key] != value for key,value in {"logical_step":3,"phase":"single","index":0,"local_clock":0}.items()):
        raise ValueError("final-latent common joint-tail clock differs")
    helpers["save"](output/"native-segment.json",{"input_cut":pre6.fingerprint,"boundaries":references,"receipts":paths,
        "native_Z3":references[15],"native_project":references[16],
        **{"native_post_joint"+str(index):references[17+index] for index in range(5)},
        "common_post_joint4":boundary(common),"native_final":boundary(native_final),
        "counts":{"adapter_steps":44,"acts":0,"receipt_emissions":44}})
    validate_prediction(pre6,prediction)
    del cuts
    trajectories = {"native":[observed(value,index,paths[21+index]["fingerprint"])
        for index,value in enumerate(references[21:44])]}
    def tensor(value,reference):
        if not isinstance(value,np.ndarray) or value.dtype != np.float32 or tuple(value.shape) != tuple(reference.shape) or not np.isfinite(value).all():
            raise ValueError("final-latent sealed payload geometry/type differs")
        if reference.dtype == torch.bfloat16:
            step()["_descriptor"](value)
        result = torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
        result.copy_(torch.from_numpy(np.array(value,copy=True)).to(device=reference.device,dtype=reference.dtype))
        if not np.array_equal(result.float().cpu().numpy().view(np.uint32),value.view(np.uint32)):
            raise ValueError("final-latent admission would cast sealed values")
        return result
    def context(value,reference):
        if isinstance(reference,torch.Tensor):
            return tensor(value,reference)
        if isinstance(reference,dict) and set(value) == set(reference):
            return {key:context(value[key],reference[key]) for key in reference}
        if isinstance(reference,(tuple,list)) and len(value) == len(reference):
            return tuple(context(left,right) for left,right in zip(value,reference,strict=True))
        raise ValueError("final-latent complete live context structure differs")
    rows = []
    for spec in config["branches"]:
        name,mode = spec["id"],spec["mode"]
        parent = common if mode == "bridge" else native_final
        nav = navigator(parent,name,22)
        row = {**spec,"parent":parent.fingerprint,"entry":boundary(parent),"acts":[],
            "bridge_receipts":[],"bridge_records":[],"bridge_boundaries":[],"continuation_receipts":[]}
        if mode != "native":
            before = nav.capture()
            if mode == "sham":
                payload = {key:before.payload[key] for key in ("latent","context","image","text")}
            elif mode == "bridge":
                prefix = prediction["prefix"]
                reentry = prefix["prefix"]["prefix"]
                payload = {"latent":tensor(reentry["remaining_step"]["commit"]["latent"],before.payload["latent"]),
                    "context":context(reentry["project"]["context"],before.payload["context"]),
                    **{key:tensor(prefix["joints"]["4"][key],before.payload[key]) for key in ("image","text")}}
            else:
                payload = {"latent":tensor(prediction["final_step"]["commit"]["latent"],before.payload["latent"]),
                    "context":None,"image":None,"text":None}
            act = closure_act(before,**payload,name=name+".complete-closure")
            entry = nav.apply(act)
            installed = nav.capture()
            store.save(installed)
            if describe(installed.payload) != describe({**before.payload,**payload}):
                raise ValueError("final-latent atomic admission changed undeclared state")
            row["acts"].append({"slots":["latent","context","image","text"],"parent":before.fingerprint,
                "post_act":installed.fingerprint,"receipt":entry.to_dict(),"receipt_id":entry.fingerprint,
                "act":act.manifest(),"undeclared_slots_exact":True})
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
        final = nav.capture()
        terminal(final)
        store.save(final)
        row["terminal_final"] = boundary(final)
        counts["vae_calls"] += 1
        image = adapter.decode(nav.session,height=512,width=512)
        image_path = output/(name+".png")
        image.save(image_path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint,final_latent=array(final.payload["latent"]),
            image={"path":image_path.name,"sha256":sha(image_path)})
        helpers["save"](output/"rows"/(name+".json"),row)
        rows.append(row)
        event("branch_complete",row=name)
    if set(trajectories) != {"native","bridge.predicted"} or any(len(value) != 23 for value in trajectories.values()):
        raise ValueError("final-latent full trajectory census differs")
    helpers["save"](output/"continuation-observations.json",{"schema":"mdb-flux-final-latent-continuation-v1",
        "plan":config["observation_plan"],"rows":trajectories,"branch_execution_order":["native","bridge.predicted"],"model_calls_added":0})
    controls = {"sham_final_latent_exact_native":rows[0]["final_latent"] == rows[1]["final_latent"],
        "sham_png_exact_native":rows[0]["image"]["sha256"] == rows[1]["image"]["sha256"]}
    return {"rows":rows,"counts":dict(counts),"control_observations":controls,
        "native_segment_sha256":sha(output/"native-segment.json"),
        "continuation_observations_sha256":sha(output/"continuation-observations.json")}


def run(*,model,adapter,session,output,config,request,source,counts,array=None,guard_components=None,
        wall=lambda:None,event=lambda *args,**kwargs:None,started=None,production_profile=True):
    from saturn_pub.store import LocalStore

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    started = time.monotonic() if started is None else started
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    helpers = util()
    save,identity = helpers["save"],helpers["identity"]
    if (output/"plan.json").exists():
        raise ValueError("immutable final-latent output already contains a plan")
    array = array or step()["make_array_writer"](output,wall)
    save(output/"plan.json",config)
    save(output/"plan-frozen.json",{"plan_sha256":sha(output/"plan.json"),"job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_identity":identity(request["source_hashes"]),"counts":dict(counts),
        "scope":"Frozen before clean68transitionprefix; all44 own-state stages seal before native selected targets."})
    boundary = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    abi = verify_source_abi(boundary=boundary,reentry=reentry,closure_type=type(adapter),
        expected_final_closure_module_sha256=config["expected_final_closure_module_sha256"])
    if adapter.execution.get("adapter") != "mdb-flux-final-latent-closure-v1":
        raise ValueError("final-latent requires its distinct live single0 and terminal reset closure contract")
    save(output/"native-source-abi.json",abi)
    store = LocalStore(output/"typed-store")
    pre6,prefix = run_prefix(adapter=adapter,session=session,output=output,store=store,counts=counts,wall=wall)
    prefix_ref = {"path":"prefix.json","sha256":sha(output/"prefix.json")}
    source = {**source,"prefix":prefix_ref,"native_source_abi":{"path":"native-source-abi.json","sha256":sha(output/"native-source-abi.json")}}
    save(output/"source.json",source)
    transition = runpy.run_path(str(Path(__file__).with_name("flux_transition_reference.py")))
    released = {"native":False,"blocked":0,"targets":{key:0 for key in contract()["target_calls"]}}
    targets = {model.single_transformer_blocks[index]:str(index) for index in FINAL_BLOCK_INDICES}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out",model.pos_embed:"pos_embed",
        **{model.transformer_blocks[index]:"joint"+str(index) for index in range(5)},
        **{getattr(model,name):name for name in PROJECT_TARGETS}})
    def guard(module,args):
        if not released["native"]:
            released["blocked"] += 1
            raise ValueError("native component entered before whole final-latent forecast seal")
        if module in targets:
            released["targets"][targets[module]] += 1
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
            tx.predict("final_latent",lambda _:{"predictions":predict_final_latent(binding,archive_root=output,
                transition=transition,boundary=boundary,reentry=reentry,config=config,event=event)})
            tx.predict("hypotheses",lambda _:{"predictions":config["hypotheses"]})
            seal = tx.seal()
            event("final_latent_sealed",seal=seal,target_entries_before_seal=dict(released["targets"]),blocked_entries=released["blocked"])
            released["native"] = True
            tx.execute("native",lambda context:run_panel(adapter=adapter,pre6=pre6,output=output,config=config,
                store=store,array=array,counts=counts,wall=wall,event=event,prediction=context.predictions["final_latent"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output/"transaction-report.json",tx.report())
    if (any(counts[name] != COUNTS[name] for name in ("adapter_steps","acts","receipt_emissions","images","vae_calls","candidate_calls"))
            or released["blocked"] != 0 or released["targets"] != contract()["target_calls"]):
        raise ValueError("final-latent completed physical/target census differs")
    summary = {"schema":"mdb-flux-final-latent-result-v1","job_id":request["job_id"],"request_sha256":request["request_sha256"],
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
        raise ValueError("final-latent request/source/transport identity differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path("flux-final-latent-result")
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_FLUX_FINAL_LATENT_BOOTSTRAP "+json.dumps({"job_id":request["job_id"],"seconds":time.monotonic()-started}),flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

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
        raise ValueError("final-latent requires admitted CUDA")
    reference = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    source_abi = verify_source_abi(boundary=reference,reentry=reentry,closure_type=FluxFinalLatentClosureAdapter,
        expected_final_closure_module_sha256=config["expected_final_closure_module_sha256"])
    save(output / "pre-model-source-abi.json",source_abi)
    scalar_qualification = reference["qualify_torch_scalar_promotion"](torch,device="cuda")
    save(output / "cuda-scalar-qualification.json",scalar_qualification)
    disk = shutil.disk_usage(output)
    if disk.free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("final-latent output filesystem lacks declared12GiB free disk")
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
        print("MDB_FLUX_FINAL_LATENT_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        now = time.monotonic()
        if now-started > config["max_wall_seconds"]:
            raise RuntimeError("finite final-latent wall budget exhausted")
        if now-heartbeat["last"] >= 15:
            heartbeat["last"] = now
            event("heartbeat")
    array = step()["make_array_writer"](output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("final-latent checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("final-latent requires deterministic distilled five-joint/twenty-single Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind):sha(inspect.getfile(kind)) for kind in (type(pipe),type(pipe.text_encoder),
        type(pipe.transformer),type(pipe.transformer.single_transformer_blocks[6]),
        type(pipe.transformer.single_transformer_blocks[6].attn.processor),type(pipe.transformer.norm_out),
        type(pipe.vae),type(pipe.scheduler),Flux2KleinAdapter,FluxForecastClosureAdapter,FluxJointForecastClosureAdapter,FluxJointTailClosureAdapter,FluxFinalLatentClosureAdapter,
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
            raise ValueError("final-latent native schedule differs")
        session = adapter.session(latent=latent,conditioning=conditioning,img_ids=img_ids,txt_ids=txt_ids,
            timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
    source = {"schema":"mdb-flux-final-latent-source-v1","job_id":request["job_id"],
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
        raise ValueError("final-latent global load/encoding/noise census differs")
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    manifest = step()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"],
        max_expanded_bytes=config["transport_limits"]["max_expanded_bytes"],
        max_member_bytes=config["transport_limits"]["max_member_bytes"])
    print("MDB_FLUX_FINAL_LATENT_COMPLETE "+json.dumps(json_value({"summary":summary,"manifest":manifest}),sort_keys=True),flush=True)
    return output,summary


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-final-latent-result")
        if output.is_dir() and not (output/"remote-manifest.json").exists():
            util()["save"](output/"failure.json",{"schema":"mdb-flux-final-latent-failure-v1",
                "job_id":os.environ.get("MRUN_JOB_ID"),"error":f"{type(error).__name__}: {error}",
                "mechanics_status":"errored","scope":"Completed evidence preserved; no model retry, archive or fabricated report."})
        raise
