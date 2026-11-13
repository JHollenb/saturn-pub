"""Two sealed own-state forecasts from actual Z0 and matched native Z2.

Only immutable pure predecessor numerical helpers are reused. Checkpoint-backed
parameter records deliberately have no corresponding producer-side weight files.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import runpy
import shutil
import struct
import time
from functools import lru_cache
from pathlib import Path

CONTEXT = {"id":"p1-s113","prompt_id":"p1","seed":113,
    "prompt":"A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BLOCK_INDICES = list(range(20))
LOGICAL_STEPS = [0,1,2,3]
MATCHED_LOGICAL_STEPS = [2,3]
BRANCHES = [{"id":"full.predicted","mode":"forecast","origin":0},
    {"id":"matched.predicted","mode":"forecast","origin":2},{"id":"native","mode":"native"},{"id":"sham","mode":"sham"}]
ORDER = [row["id"] for row in BRANCHES]
COUNTS = {"model_loads":1,"prompt_encodings":1,"noise_draws":1,"adapter_steps":112,"acts":3,
    "candidate_calls":0,"receipt_emissions":115,"images":4,"vae_calls":4,"physical_batch":1}
SAMPLE_INDICES = [0,31,511,512,543,767,1023,1535]
IMAGE_SAMPLE_INDICES = [0,31,32,255,511,512,767,1023]
PROJECT_TARGETS = ("x_embedder","context_embedder","time_guidance_embed","double_stream_modulation_img",
    "double_stream_modulation_txt","single_stream_modulation")
PARAMETER_MODULE_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
OLD_CLOSURE_SHA256 = "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"
PRIOR_JOINT_CLOSURE_SHA256 = "bb352ae4859335abdd751d33ccc2774fa4d37c53f5167c5012183c693b0d709c"
PRIOR_TAIL_CLOSURE_SHA256 = "2788c984045098532009e105219e7c9cfcc38732a54d892bdc296149a979b586"
PRIOR_FINAL_CLOSURE_SHA256 = "b70f3d68f7fb0b1c0e2892bb7ab4efb8828a906ef80d9acb6f794608d673e579"
DERIVED_TERMINAL_SHA256 = "7291e89a5b09823c5f595b3263626b23fb44951f46ef9fa5fd7c59002b8866c8"
OBSERVATION_PLAN = {"native_boundaries":112,"forecast_transitions":{"full":112,"matched":56},
    "storage":"complete-predicted-fields-and-native-lossless-BF16-arrays-and-typed-cuts",
    "scope":"Actual Z0/conditioning/IDs/schedule bound; full PNG before any native denoiser, matched PNG after56native before logicalstep2/3 targets."}
LEGACY_SOURCES = {
    "flux_two_pass_campaign.py":"f6f631343820046bdc1e5eb1ef05f2c991be6e174c44bebd0abd0aaa02686e9e",
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
        for block in iter(lambda:stream.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def prior():
    for name,digest in LEGACY_SOURCES.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("immutable predecessor source differs: "+name)
    return runpy.run_path(str(Path(__file__).with_name("flux_two_pass_campaign.py")))


def step():
    return prior()["step"]()


def util():
    return prior()["util"]()


def target_counts(multiplier=1):
    return {**{str(index):multiplier for index in BLOCK_INDICES},"norm_out":multiplier,"proj_out":multiplier,
        **{name:multiplier for name in PROJECT_TARGETS},"pos_embed":2*multiplier,
        **{"joint"+str(index):multiplier for index in range(5)}}


def contract(config=None):
    if config is not None and (config.get("logical_steps",LOGICAL_STEPS) != LOGICAL_STEPS
            or config.get("block_indices",BLOCK_INDICES) != BLOCK_INDICES):
        raise ValueError("initial-noise requires the full declared four-pass profile")
    return json.loads(json.dumps({"context":CONTEXT,"block_indices":BLOCK_INDICES,"logical_steps":LOGICAL_STEPS,
        "matched_logical_steps":MATCHED_LOGICAL_STEPS,"branches":BRANCHES,"order":ORDER,"counts":COUNTS,
        "prefix_steps":0,"reference_steps":112,"matched_origin_after_native_steps":56,"mathematical_steps":168,
        "target_calls":target_counts(4),"target_calls_by_step":{str(index):target_counts() for index in LOGICAL_STEPS},
        "observation_plan":OBSERVATION_PLAN}))


def validate_config(config):
    fixed = {"context":CONTEXT,"block_indices":BLOCK_INDICES,"logical_steps":LOGICAL_STEPS,
        "matched_logical_steps":MATCHED_LOGICAL_STEPS,"branches":BRANCHES,"branch_execution_order":ORDER,
        "selector":{"logical_step":0,"phase":"project","index":None},
        "matched_selector":{"logical_step":2,"phase":"project","index":None},"height":512,"width":512,"steps":4,
        "sample_indices":SAMPLE_INDICES,"image_sample_indices":IMAGE_SAMPLE_INDICES,"query_chunk":64,
        "projection_chunk":256,"neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits",
        "archive_integer_ids":"native-int64","parameter_storage":"checkpoint-recipe-v1",
        "parameter_checkpoint_file":"transformer/diffusion_pytorch_model.safetensors","cpu_threads":4,
        "full_transaction_max_bytes":3<<30,"matched_transaction_max_bytes":1536<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":10<<30,"max_member_bytes":3<<30},"observation_plan":OBSERVATION_PLAN,
        "expected_closure_module_sha256":OLD_CLOSURE_SHA256,"expected_joint_closure_module_sha256":PRIOR_JOINT_CLOSURE_SHA256,
        "expected_tail_closure_module_sha256":PRIOR_TAIL_CLOSURE_SHA256,"expected_final_closure_module_sha256":PRIOR_FINAL_CLOSURE_SHA256,
        "expected_derived_terminal_module_sha256":DERIVED_TERMINAL_SHA256,
        "expected_parameter_artifacts_module_sha256":PARAMETER_MODULE_SHA256}
    if any(config.get(key) != value for key,value in fixed.items()):
        raise ValueError("initial-noise frozen scientific/source geometry differs")
    for key in ("adapter_steps","acts","candidate_calls","images","vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("initial-noise physical census differs")
    if type(config.get("cpu_threads")) is not int or type(config.get("max_wall_seconds")) is not int or not 1<=config["max_wall_seconds"]<=1800:
        raise ValueError("initial-noise finite wall/thread budget differs")


def verify_source_abi(*,boundary,reentry,closure_type,config):
    from mdb_lanes import parameter_artifacts

    old = prior()["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=closure_type,
        expected_derived_terminal_module_sha256=config["expected_derived_terminal_module_sha256"])
    actual = sha(inspect.getfile(parameter_artifacts))
    if actual != config["expected_parameter_artifacts_module_sha256"]:
        raise ValueError("parameter retrieval module source ABI differs")
    return {"schema":"mdb-flux-initial-noise-native-source-abi-v1","verified":True,"prior_two_pass":old,
        "parameter_artifacts_module_sha256":actual,"native_advance_inherited":True,
        "scope":"Qualified native advancement and immutable pure equations unchanged; parameter recipes separately source-bound."}

class CheckpointParameters:
    """Bound native parameter exports and ephemeral FP32 banks, with no weight files."""
    def __init__(self,root,filename,file_sha256,model):
        self.path = Path(root)/filename
        if self.path.is_symlink() or not self.path.is_file() or sha(self.path) != file_sha256:
            raise ValueError("parameter checkpoint complete file identity differs")
        self.filename,self.file_sha256 = filename,file_sha256
        self.signature = self._signature()
        with self.path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("parameter checkpoint header truncated")
            length = struct.unpack("<Q",raw)[0]
            if not 0 < length <= 16<<20:
                raise ValueError("parameter checkpoint header exceeds bound")
            self.header = json.loads(stream.read(length))
            self.data_start = 8+length
        self.names = {id(value):name for name,value in model.named_parameters()}
        self.descriptors,self.records,self.entries = {},{},{}
        self.loads = 0

    def _signature(self):
        stat = self.path.stat()
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def _check(self):
        if self._signature() != self.signature:
            raise ValueError("parameter checkpoint changed during forecast")

    def descriptor(self,value):
        from mdb_lanes.forward_recording import tensor_descriptor

        key = id(value)
        if key not in self.descriptors:
            self.descriptors[key] = tensor_descriptor(value)
        return self.descriptors[key]

    def record(self,value):
        import numpy as np
        import torch

        if value is None:
            return None
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("learned source must be finite native BF16")
        name = self.names[id(value)]
        if name in self.records:
            return self.records[name]
        self._check()
        native = self.descriptor(value)
        spec = self.header.get(name)
        if (not isinstance(spec,dict) or spec.get("dtype") != "BF16" or spec.get("shape") != native["shape"]
                or len(spec.get("data_offsets",())) != 2):
            raise ValueError("native parameter/checkpoint tensor geometry differs")
        start,end = spec["data_offsets"]
        expected = 2
        for dimension in native["shape"]:
            expected *= dimension
        if type(start) is not int or type(end) is not int or not 0 <= start <= end or end-start != expected or self.data_start+end > self.signature[2]:
            raise ValueError("parameter checkpoint offsets differ")
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(buffer,{"descr":"<u2","fortran_order":False,"shape":tuple(native["shape"])})
        prefix = buffer.getvalue()
        payload,archive = hashlib.sha256(),hashlib.sha256(prefix)
        with self.path.open("rb") as stream:
            stream.seek(self.data_start+start)
            remaining = end-start
            while remaining:
                block = stream.read(min(8<<20,remaining))
                if not block:
                    raise ValueError("parameter checkpoint payload truncated")
                payload.update(block)
                archive.update(block)
                remaining -= len(block)
        if payload.hexdigest() != native["sha256"]:
            raise ValueError("native parameter differs from pinned checkpoint payload")
        self._check()
        identifier = archive.hexdigest()
        path = "parameters/"+identifier+".npy"
        entry = {"path":path,"bytes":len(prefix)+end-start,"sha256":identifier,
            "npy_prefix_b64":base64.b64encode(prefix).decode(),"source":{"file":self.filename,
                "file_sha256":self.file_sha256,"tensor":name,"dtype":"BF16","shape":native["shape"],
                "data_offsets":[start,end],"payload_sha256":payload.hexdigest()}}
        record = {"path":path,"sha256":identifier,"encoding":"bf16-bits-v1","native_byteorder":"little",
            "descriptor":native,"stored_dtype":"uint16","stored_shape":native["shape"],
            "native_stride":list(value.stride()),"storage":"checkpoint-recipe-v1","parameter_tensor":name}
        if path in self.entries:
            aliases = self.entries[path].setdefault("checkpoint_tensor_aliases",[self.entries[path]["source"]["tensor"]])
            if name not in aliases:
                aliases.append(name)
        else:
            self.entries[path] = entry
        self.records[name] = record
        return record

    def load(self,record):
        import numpy as np

        if record is None:
            return None
        self._check()
        name = record.get("parameter_tensor")
        if name not in self.records or record != self.records[name]:
            raise ValueError("parameter record outside bound recipe allowlist")
        entry = self.entries[record["path"]]
        start,end = entry["source"]["data_offsets"]
        with self.path.open("rb") as stream:
            stream.seek(self.data_start+start)
            raw = stream.read(end-start)
        if len(raw) != end-start or hashlib.sha256(raw).hexdigest() != entry["source"]["payload_sha256"]:
            raise ValueError("parameter checkpoint slice identity differs")
        words = np.frombuffer(raw,dtype="<u2").astype(np.uint32)
        words <<= 16
        value = words.view(np.float32).reshape(record["stored_shape"])
        value.setflags(write=False)
        self._check()
        self.loads += 1
        return value

    def emit(self,output,*,job_id,request_sha256):
        from mdb_lanes import parameter_artifacts

        module = Path(inspect.getfile(parameter_artifacts))
        if sha(module) != PARAMETER_MODULE_SHA256:
            raise ValueError("parameter recipe standalone source differs")
        manifest = {"schema":"mdb-parameter-artifacts-v1","artifacts":list(self.entries.values()),
            "defaults":{"artifact_root":"parameter-cache","checkpoint_root":str(self.path.parent.parent),
                "ssh_host":"beast","python":"python3"},
            "provenance":{"job_id":job_id,"request_sha256":request_sha256,"producer_source_sha256":sha(Path(__file__)),
                "retrieval_module_sha256":sha(module),"checkpoint_file_sha256":self.file_sha256,
                "scope":"Recipes only; parameter archive paths intentionally absent from producer inventory. Read pinned checkpoint banks ephemerally, or hydrate into a shared external cache with --root before audit. No per-run persistent weight bank."}}
        manifest = parameter_artifacts.validate_manifest(manifest)
        util()["save"](Path(output)/"parameter-artifacts.json",manifest)
        with (Path(output)/"retrieve_weights.py").open("xb") as stream:
            stream.write(module.read_bytes())
        return {"path":"parameter-artifacts.json","sha256":sha(Path(output)/"parameter-artifacts.json"),
            "script":"retrieve_weights.py","script_sha256":sha(module),"artifacts":len(manifest["artifacts"]),
            "parameter_files_written":0,"checkpoint_file_sha256":self.file_sha256}


def origin_record(cut,array):
    p = cut.payload
    if (p["phase"] != "project" or p["cursor"] != 0 or p["step"] not in (0,2)
            or any(p[key] is not None for key in ("image","text","context","noise_prediction"))):
        raise ValueError("forecast origin must be an actual clean project frontier")
    return {"cut":cut.fingerprint,"point":dict(cut.execution_point),"logical_step":p["step"],
        "inputs":{key:array(p[key]) for key in ("latent","conditioning","img_ids","txt_ids","timesteps","sigmas")}}


def bind_inputs(*,model,initial,array,counts,config,production_profile,checkpoint):
    import torch
    from saturn_pub.values import describe

    

    p = initial.payload
    if (p["step"] != 0 or p["phase"] != "project" or p["cursor"] != 0
            or any(p[key] is not None for key in ("image","text","context","noise_prediction"))):
        raise ValueError("initial-noise source must be the genuine pre-project frontier")
    def parameter(value):
        return checkpoint.record(value)
    def bank(parameters):
        return {name:parameter(value) for name,value in parameters.items()}
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_campaign.py")))
    profile = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))["attention_profile"]
    binding = {"schema":"mdb-flux-initial-noise-shared-binding-v1","input_cut":initial.fingerprint,
        "point":dict(initial.execution_point),"block_indices":BLOCK_INDICES,"logical_steps":LOGICAL_STEPS,
        "counts_before_parameter_binding":dict(counts),"target_entries_before_parameter_binding":{key:0 for key in contract()["target_calls"]},
        "origins":{"full":origin_record(initial,array)},
        "text_rows":p["conditioning"].shape[1],"final_blocks":{},
        "carried_state":describe({key:value for key,value in p.items()
            if key not in {"image","text","latent","context","noise_prediction","step","phase","cursor"}}),
        "clock_state":{key:p[key] for key in ("step","phase","cursor")}}
    project = {name:value for name,value in model.named_parameters() if name.startswith(reentry["PROJECT_PREFIXES"])}
    if not project or getattr(model.time_guidance_embed,"guidance_embedder",None) is not None:
        raise ValueError("initial-noise requires unguided qualified project")
    binding["project"] = {"parameters":bank(project),
        "native_parameters":{name:checkpoint.descriptor(value) for name,value in project.items()},
        "axes_dims_rope":list(model.pos_embed.axes_dim),"rope_theta":model.pos_embed.theta,
        "timestep_channels":model.time_guidance_embed.time_proj.num_channels}
    for index,joint in enumerate(model.transformer_blocks):
        attn = joint.attn
        if (any(getattr(joint,name).elementwise_affine or getattr(joint,name).eps != 1e-6
                for name in ("norm1","norm1_context","norm2","norm2_context"))
                or len({attn.norm_q.eps,attn.norm_k.eps,attn.norm_added_q.eps,attn.norm_added_k.eps}) != 1):
            raise ValueError("initial-noise joint normalization profile differs")
        parameters = dict(joint.named_parameters())
        binding["joint"+str(index)] = {"parameters":bank(parameters),
            "native_parameters":{name:checkpoint.descriptor(value) for name,value in parameters.items()},
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
            raise ValueError("initial-noise single profile differs")
        if production_profile and (h,m,attn.heads,attn.head_dim) != (3072,9216,24,128):
            raise ValueError("initial-noise qualified production dimensions differ")
        parameters = {"qkv_mlp_weight":attn.to_qkv_mlp_proj.weight,"qkv_mlp_bias":attn.to_qkv_mlp_proj.bias,
            "to_out_weight":attn.to_out.weight,"to_out_bias":attn.to_out.bias,
            "norm_q_weight":attn.norm_q.weight,"norm_k_weight":attn.norm_k.weight}
        binding["final_blocks"][str(index)] = {"parameters":bank(parameters),
            "native_parameters":{name:checkpoint.descriptor(value) for name,value in block.named_parameters()},
            "native_parameter_prefix":f"single_transformer_blocks.{index}.","block_index":index,
            "heads":attn.heads,"hidden_width":h,"mlp_width":m,"head_dim":attn.head_dim,
            "layernorm_eps":block.norm.eps,"qk_eps":attn.norm_q.eps,"attention_profile":profile(block)}
    if len(model.transformer_blocks) != 5 or len(model.single_transformer_blocks) != 20:
        raise ValueError("initial-noise requires exact five joint and twenty single banks")
    norm = model.norm_out
    if (not isinstance(norm.norm,torch.nn.LayerNorm) or norm.norm.elementwise_affine
            or norm.norm.eps != 1e-6 or not isinstance(norm.silu,torch.nn.SiLU)):
        raise ValueError("initial-noise readout profile differs")
    binding["readout"] = {"parameters":bank({"modulation_weight":norm.linear.weight,"modulation_bias":norm.linear.bias,
        "projection_weight":model.proj_out.weight,"projection_bias":model.proj_out.bias}),
        "native_parameters":{name:checkpoint.descriptor(value) for name,value in norm.named_parameters()},
        "projection_native_parameters":{name:checkpoint.descriptor(value) for name,value in model.proj_out.named_parameters()},
        "eps":norm.norm.eps,"elementwise_affine":False}
    binding["pass_schedules"] = {str(index):{"timestep":array(p["timesteps"][index]),"sigma":array(p["sigmas"][index]),
        "sigma_next":array(p["sigmas"][index+1]),"next_clock":{"step":index+1,"phase":"project","cursor":0},
        "reset":dict.fromkeys(("image","text","context","noise_prediction"))} for index in LOGICAL_STEPS}
    if production_profile and (tuple(p["latent"].shape) != (1,1024,128)
            or tuple(p["conditioning"].shape) != (1,512,7680)
            or tuple(p["img_ids"].shape[-2:]) != (1024,4) or tuple(p["txt_ids"].shape[-2:]) != (512,4)):
        raise ValueError("initial-noise qualified source geometry differs")
    return binding


def predict_passes(binding,*,archive_root,checkpoint,transition,boundary,reentry,config,event):
    import numpy as np

    from mdb_lanes.tensor_archive import load_tensor_archive

    def load(record):
        return None if record is None else load_tensor_archive(archive_root,record)
    inputs = binding["origin"]["inputs"]
    latent = load(inputs["latent"])
    conditioning,img_ids,txt_ids = (load(inputs[key]) for key in ("conditioning","img_ids","txt_ids"))
    passes = []
    for logical in binding["logical_steps"]:
        schedule = binding["pass_schedules"][str(logical)]
        spec = binding["project"]
        parameters = {name:checkpoint.load(value) for name,value in spec["parameters"].items()}
        project = reentry["predict_project"](latent,conditioning,img_ids,txt_ids,load(schedule["timestep"]),parameters,
            axes_dims_rope=spec["axes_dims_rope"],rope_theta=spec["rope_theta"],
            timestep_channels=spec["timestep_channels"],projection_chunk=config["projection_chunk"])
        del parameters
        event("project_forecast_complete",history=binding["history"],logical_step=logical)
        ctx,current_image,current_text = project["context"],project["image"],project["text"]
        joints = {}
        for index in range(5):
            spec = binding["joint"+str(index)]
            parameters = {name:checkpoint.load(value) for name,value in spec["parameters"].items()}
            prediction = reentry["predict_joint"](current_image,current_text,ctx,parameters,
                heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
                query_chunk=config["query_chunk"],projection_chunk=config["projection_chunk"],retain_fields=False)
            del parameters
            joints[str(index)] = prediction
            current_image,current_text = prediction["image"],prediction["text"]
            event("joint_forecast_complete",history=binding["history"],logical_step=logical,block_index=index)
            gc.collect()
        current = np.concatenate((current_text,current_image),axis=1)
        current.setflags(write=False)
        singles = []
        for index in BLOCK_INDICES:
            spec = binding["final_blocks"][str(index)]
            parameters = {name:checkpoint.load(value) for name,value in spec["parameters"].items()}
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
            event("single_forecast_complete",history=binding["history"],logical_step=logical,block_index=index)
            gc.collect()
        readout = boundary["predict_readout"](current[:,binding["text_rows"]:],ctx["temb"],
            **{name:checkpoint.load(value) for name,value in binding["readout"]["parameters"].items()},
            eps=binding["readout"]["eps"],neuron_chunk=config["projection_chunk"],
            sample_indices=config["image_sample_indices"],retain_fields=False)
        commit = boundary["predict_flow_commit"](latent,readout["noise"],load(schedule["sigma"]),load(schedule["sigma_next"]))
        event("readout_commit_forecast_complete",history=binding["history"],logical_step=logical)
        passes.append({"logical_step":logical,"input_source":"observed origin latent" if logical == binding["logical_steps"][0] else "own previous pass commit latent",
            "input_latent_descriptor":step()["_descriptor"](latent),"project":project,"joints":joints,
            "single_steps":singles,"readout":readout,"commit":commit,"chain_length":28,
            "context_source":"own predicted project context","context_input_descriptors":prior()["context_descriptors"](ctx),
            "next_state_contract":{"clock":schedule["next_clock"],"reset":schedule["reset"],"carried_state":binding["carried_state"]}})
        latent = commit["latent"]
    result = {"passes":passes,"chain_length":28*len(passes),"origin_logical_step":binding["origin"]["logical_step"],
        "scope":"Complete declared own-state passes; every project regenerates context from its own incoming latent and observed conditioning/IDs/schedule. No native future input."}
    event("forecast_storage",history=binding["history"],**prior()["array_storage_profile"](result))
    return result


def validate_prediction(origin,prediction):
    logical_steps = list(range(origin.payload["step"],4))
    import numpy as np

    normal,descriptor = util()["json_value"],step()["_descriptor"]
    if (prediction["chain_length"] != 28*len(logical_steps) or len(prediction["passes"]) != len(logical_steps)
            or prediction["origin_logical_step"] != origin.payload["step"]):
        raise ValueError("initial-noise forecast length differs")
    latent = origin.payload["latent"].float().cpu().numpy()
    for logical,row in zip(logical_steps,prediction["passes"],strict=True):
        project,ctx = row["project"],row["project"]["context"]
        if (row["logical_step"] != logical or row["chain_length"] != 28
                or row["input_source"] != ("observed origin latent" if logical == logical_steps[0] else "own previous pass commit latent")
                or normal(row["input_latent_descriptor"]) != descriptor(latent)
                or normal(project["input_latent_descriptor"]) != descriptor(latent)
                or normal(row["context_input_descriptors"]) != prior()["context_descriptors"](ctx)):
            raise ValueError("initial-noise own latent/context input differs")
        image,text = project["image"],project["text"]
        for index in range(5):
            joint = row["joints"][str(index)]
            if (normal(joint["input_image_descriptor"]) != descriptor(image)
                    or normal(joint["input_text_descriptor"]) != descriptor(text)
                    or normal(joint["image_descriptor"]) != descriptor(joint["image"])
                    or normal(joint["text_descriptor"]) != descriptor(joint["text"])):
                raise ValueError("initial-noise joint own-state join differs")
            image,text = joint["image"],joint["text"]
        current = np.concatenate((text,image),axis=1)
        if len(row["single_steps"]) != 20:
            raise ValueError("initial-noise single census differs")
        for index,entry in enumerate(row["single_steps"]):
            value = entry["prediction"]
            expected = descriptor(current)
            if (entry["block_index"] != index or entry["step"] != index
                    or normal(entry["input_descriptor"]) != expected or normal(value["input_descriptor"]) != expected
                    or normal(value["output_descriptor"]) != descriptor(value["output"])
                    or value["output"].shape != current.shape or "image" in value or "text" in value):
                raise ValueError("initial-noise single own-state join differs")
            current = value["output"]
        readout,commit = row["readout"],row["commit"]
        if (normal(readout["input_descriptor"]) != descriptor(current[:,text.shape[1]:])
                or normal(readout["noise_descriptor"]) != descriptor(readout["noise"])
                or normal(commit["input_descriptor"]) != descriptor(latent)
                or normal(commit["noise_descriptor"]) != descriptor(readout["noise"])
                or normal(commit["latent_descriptor"]) != descriptor(commit["latent"])
                or normal(row["next_state_contract"]["clock"]) != {"step":logical+1,"phase":"project","cursor":0}
                or normal(row["next_state_contract"]["reset"]) != dict.fromkeys(("image","text","context","noise_prediction"))):
            raise ValueError("initial-noise readout/commit/reset join differs")
        latent = commit["latent"]


def receipt(output,store,counts,branch,value):
    body = value.to_dict()
    counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps",1)
    counts["receipt_emissions"] += 1
    if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
        raise RuntimeError("initial-noise finite physical budget exhausted")
    store.receipt(value)
    with (output/"receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps({"sequence":counts["receipt_emissions"]-1,"branch":branch,"receipt":value.fingerprint},sort_keys=True)+"\n")


def verify_history_transaction(root,config,history,seal):
    """Use the already declared finite history budget, not the legacy1GiB default."""
    from mdb_lanes.forecast_transaction import verify_transaction

    if history not in ("full","matched"):
        raise ValueError("unknown transaction history")
    return verify_transaction(root,expected_seal_sha256=seal["seal_sha256"],
        max_bytes=config[history+"_transaction_max_bytes"])


def run(*,model,adapter,session,output,config,request,source,counts,checkpoint,array=None,guard_components=None,
        wall=lambda:None,event=lambda *args,**kwargs:None,started=None,production_profile=True):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.flux_derived_terminal import derive_terminal
    from mdb_lanes.flux_final_latent_closure import closure_act
    from mdb_lanes.forecast_transaction import ForecastTransaction
    from mdb_lanes.native_navigation import NativeCutNavigator

    started = time.monotonic() if started is None else started
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    helpers = util()
    save,identity,normal = helpers["save"],helpers["identity"],helpers["json_value"]
    if (output/"plan.json").exists():
        raise ValueError("immutable initial-noise output already contains a plan")
    if adapter.execution.get("adapter") != "mdb-flux-final-latent-closure-v1":
        raise ValueError("initial-noise requires qualified inherited final closure surface")
    array = array or step()["make_array_writer"](output,wall)
    save(output/"plan.json",config)
    save(output/"plan-frozen.json",{"plan_sha256":sha(output/"plan.json"),"job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_identity":identity(request["source_hashes"]),"counts":dict(counts),
        "scope":"Frozen at actual initialZ0 before any native denoiser; no prefix is executed."})
    store = LocalStore(output/"typed-store")
    initial = session.capture()
    if any(initial.execution_point[key] != value for key,value in {"logical_step":0,"phase":"project","index":None,"local_clock":0}.items()):
        raise ValueError("initial-noise requires genuine initial pre-project Z0")
    if counts["adapter_steps"] != 0 or counts["acts"] != 0 or counts["images"] != 0:
        raise ValueError("initial-noise history already executed a selected consumer")
    store.save(initial)
    boundary_math = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    transition = runpy.run_path(str(Path(__file__).with_name("flux_transition_reference.py")))
    abi = verify_source_abi(boundary=boundary_math,reentry=reentry,closure_type=type(adapter),config=config)
    save(output/"native-source-abi.json",abi)
    components = tuple(guard_components or (model,))
    parameters_before = prior()["parameter_census"](components)
    state = {"phase":"forecast","history":"full","blocked":0,"point":dict(initial.execution_point),
        "target_calls":target_counts(0),"target_calls_by_step":{str(index):target_counts(0) for index in LOGICAL_STEPS}}
    targets = {model.single_transformer_blocks[index]:str(index) for index in BLOCK_INDICES}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out",model.pos_embed:"pos_embed",
        **{model.transformer_blocks[index]:"joint"+str(index) for index in range(5)},
        **{getattr(model,name):name for name in PROJECT_TARGETS}})
    vae_modules = {id(module) for component in components if component is not model for module in component.modules()}
    def guard(module,args):
        if state["phase"] != "native" and not (state["phase"] in ("predicted_vae","terminal_vae") and id(module) in vae_modules):
            state["blocked"] += 1
            raise ValueError("native component entered a forbidden forecast/render window")
        if module in targets:
            logical = str(state["point"]["logical_step"])
            if logical not in state["target_calls_by_step"]:
                raise ValueError("native target entered outside qualified live schedule")
            key = targets[module]
            state["target_calls"][key] += 1
            state["target_calls_by_step"][logical][key] += 1
    handles = [module.register_forward_pre_hook(guard) for component in components for module in component.modules()]
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
    def navigator(cut,branch,limit):
        return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=limit,
            on_receipt=lambda value:receipt(output,store,counts,branch,value))
    def terminal(cut):
        if (cut.payload["step"] != 4 or cut.payload["phase"] != "project" or cut.payload["cursor"] != 0
                or any(cut.payload[key] is not None for key in ("context","image","text","noise_prediction"))):
            raise ValueError("initial-noise terminal clock/reset differs")
    def payload(value,reference):
        step()["_descriptor"](value)
        if value.dtype != np.float32 or tuple(value.shape) != tuple(reference.shape):
            raise ValueError("sealed terminal latent geometry differs")
        result = torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
        result.copy_(torch.from_numpy(np.array(value,copy=True)).to(device=reference.device,dtype=reference.dtype))
        if not np.array_equal(result.float().cpu().numpy().view(np.uint32),value.view(np.uint32)):
            raise ValueError("terminal admission would cast sealed values")
        return result
    def apply(nav,action,expected):
        before = nav.capture()
        entry = nav.apply(action)
        installed = nav.capture()
        store.save(installed)
        if describe(installed.payload) != describe({**before.payload,**expected}):
            raise ValueError("atomic terminal admission changed undeclared state")
        return {"slots":["latent","context","image","text"],"parent":before.fingerprint,"post_act":installed.fingerprint,
            "receipt":entry.to_dict(),"receipt_id":entry.fingerprint,"act":action.manifest(),"undeclared_slots_exact":True},installed
    def render(row,nav):
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
        save(output/"rows"/(row["id"]+".json"),row)
        event("branch_complete",row=row["id"],counts=dict(counts))
        return row
    def guard_snapshot(origin_step):
        return {"target_calls":normal(state["target_calls"]),"target_calls_by_step":normal(state["target_calls_by_step"]),
            "clock_qualified_targets_remaining":{str(index):normal(state["target_calls_by_step"][str(index)])
                for index in range(origin_step,4)}}
    def integrity(origin,source_before,native_before):
        checkpoint._check()
        if (identity(describe(origin.payload)) != source_before
                or prior()["parameter_census"](components) != parameters_before or state["blocked"] != 0
                or counts["adapter_steps"] != native_before
                or any(any(row.values()) for index,row in state["target_calls_by_step"].items() if int(index) >= origin.payload["step"])):
            raise ValueError("forecast origin/parameters/clock-qualified future census differs")
    rows,seals,reports,verifications,first_images = {},{},{},{},{}
    native_boundaries,native_receipts = [],[]
    try:
        binding = bind_inputs(model=model,initial=initial,array=array,counts=counts,config=config,
            production_profile=production_profile,checkpoint=checkpoint)
        binding["origins"]["full"]["counts_at_observation"] = dict(counts)
        recipes = checkpoint.emit(output,job_id=request["job_id"],request_sha256=request["request_sha256"])
        if production_profile and recipes["artifacts"] != 169:
            raise ValueError("qualified complete parameter recipe inventory differs")
        binding["parameter_artifacts"] = recipes
        save(output/"binding-initial.json",binding)
        source = {**source,"native_source_abi":{"path":"native-source-abi.json","sha256":sha(output/"native-source-abi.json")},
            "binding_initial":{"path":"binding-initial.json","sha256":sha(output/"binding-initial.json")},
            "parameter_artifacts":recipes,"actual_initial_noise":binding["origins"]["full"]["inputs"]["latent"]}
        save(output/"source.json",source)
        def forecast(history,origin):
            origin_step = origin.payload["step"]
            source_before = identity(describe(origin.payload))
            native_before = counts["adapter_steps"]
            state.update(phase="forecast",history=history)
            shared = {key:value for key,value in binding.items() if key != "origins"}
            frontier = {**shared,"history":history,"logical_steps":list(range(origin_step,4)),
                "origin":binding["origins"][history],"counts_before_seal":dict(counts),**guard_snapshot(origin_step)}
            save(output/("binding-"+history+".json"),frontier)
            tx = ForecastTransaction(output/"transaction"/history,model_identity=adapter.model_identity,
                input_identity=identity(frontier),source_identity=identity(request["source_hashes"]),
                frontier={"kind":"architectural","binding":frontier},execution_scope=config["scope"],
                budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000,
                    "max_bytes":config[history+"_transaction_max_bytes"]})
            with tx:
                tx.frontier("source_binding",frontier)
                event("prediction_start",history=history,counts=dict(counts))
                tx.predict("passes",lambda _:{"predictions":predict_passes(frontier,archive_root=output,checkpoint=checkpoint,
                    transition=transition,boundary=boundary_math,reentry=reentry,config=config,event=event)})
                tx.predict("hypotheses",lambda _:{"predictions":config["hypotheses"]})
                integrity(origin,source_before,native_before)
                seal = tx.seal()
                seals[history] = seal
                event("history_"+history+"_sealed",history=history,seal=seal,counts=dict(counts),
                    blocked_entries=state["blocked"],**guard_snapshot(origin_step))
                def consume(context):
                    prediction = context.predictions["passes"]
                    validate_prediction(origin,prediction)
                    z4 = payload(prediction["passes"][-1]["commit"]["latent"],origin.payload["latent"])
                    derived = derive_terminal(adapter,origin,predicted_latent=z4,name=history+".complete-terminal")
                    store.save(derived.scaffold)
                    derivation_path = output/"derivations"/(history+".json")
                    save(derivation_path,derived.derivation)
                    nav = navigator(derived.scaffold,history+".predicted",1)
                    act,installed = apply(nav,derived.action,{"latent":z4,"context":None,"image":None,"text":None})
                    row = {"id":history+".predicted","mode":"forecast","history":history,
                        "source_cut":origin.fingerprint,"parent":derived.scaffold.fingerprint,"entry":boundary(derived.scaffold),
                        "installed":boundary(installed),"acts":[act],"continuation_receipts":[],
                        "derivation":{"path":derivation_path.relative_to(output).as_posix(),"sha256":sha(derivation_path)}}
                    state["phase"] = "predicted_vae"
                    row = render(row,nav)
                    integrity(origin,source_before,native_before)
                    expected = (0,1,1,1) if history == "full" else (56,2,2,2)
                    if tuple(counts[key] for key in ("adapter_steps","acts","images","vae_calls")) != expected:
                        raise ValueError("predicted image monotonic physical census differs")
                    first = {"history":history,"source_cut":origin.fingerprint,"image":row["image"],"counts":dict(counts),
                        "source_unchanged":True,"parameters_unchanged":True,"derivation_sha256":sha(derivation_path),
                        "native_steps_before":native_before,**guard_snapshot(origin_step),
                        "scope":"Actual native VAE image precedes all clock-qualified remaining denoiser targets; earlier native calls remain in global census."}
                    save(output/("first-image-"+history+".json"),first)
                    first_images[history] = first
                    event("predicted_image_complete",**first)
                    rows[row["id"]] = row
                    return {"row":row,"first_image":first,"counts":dict(counts)}
                tx.execute("predicted_terminal_consumer",consume)
            report = tx.report()
            save(output/"transaction"/history/"report.json",report)
            reports[history] = {"path":"transaction/"+history+"/report.json","sha256":sha(output/"transaction"/history/"report.json")}
            verifications[history] = verify_history_transaction(tx.root,config,history,seal)
            del tx
            gc.collect()
        forecast("full",initial)
        nav = navigator(initial,"reference",112)
        def advance_until(count):
            state["phase"] = "native"
            for position in range(len(native_receipts),count):
                wall()
                state["point"] = dict(nav.capture().execution_point)
                entry = nav.step()
                current = nav.capture()
                store.save(current)
                native_receipts.append(entry.to_dict())
                native_boundaries.append({"index":position,"receipt_id":entry.fingerprint,**boundary(current)})
                event("reference_transition_complete",index=position,point=dict(current.execution_point),counts=dict(counts))
        advance_until(56)
        matched = nav.capture()
        if any(matched.execution_point[key] != value for key,value in {"logical_step":2,"phase":"project","index":None,"local_clock":0}.items()):
            raise ValueError("matched native origin clock differs")
        binding["origins"]["matched"] = {**origin_record(matched,array),"counts_at_observation":dict(counts)}
        save(output/"binding.json",binding)
        save(output/"native-prefix.json",{"input_cut":initial.fingerprint,"boundaries":native_boundaries,
            "receipts":native_receipts,"matched_origin":binding["origins"]["matched"],"counts":dict(counts),
            "segment_counts":{"adapter_steps":56,"acts":0,"receipt_emissions":56}})
        forecast("matched",matched)
        advance_until(112)
        final = nav.capture()
        terminal(final)
        save(output/"native-segment.json",{"input_cut":initial.fingerprint,"boundaries":native_boundaries,"receipts":native_receipts,
            "native_Z2":native_boundaries[55],"native_final":boundary(final),
            "counts":dict(counts),"segment_counts":{"adapter_steps":112,"acts":0,"receipt_emissions":112}})
        state["phase"] = "terminal_vae"
        rows["native"] = render({"id":"native","mode":"native","parent":final.fingerprint,
            "entry":boundary(final),"acts":[],"continuation_receipts":[]},nav)
        sham = navigator(final,"sham",1)
        expected = {key:final.payload[key] for key in ("latent","context","image","text")}
        action,installed = apply(sham,closure_act(final,**expected,name="sham.complete-terminal"),expected)
        rows["sham"] = render({"id":"sham","mode":"sham","parent":final.fingerprint,"entry":boundary(final),
            "installed":boundary(installed),"acts":[action],"continuation_receipts":[]},sham)
    finally:
        for handle in handles:
            handle.remove()
    if (any(counts[key] != COUNTS[key] for key in ("adapter_steps","acts","receipt_emissions","candidate_calls","images","vae_calls"))
            or state["target_calls"] != target_counts(4) or state["target_calls_by_step"] != contract()["target_calls_by_step"] or state["blocked"] != 0):
        raise ValueError("initial-noise completed physical/clock-qualified target census differs")
    controls = {"sham_final_latent_exact_native":rows["sham"]["final_latent"] == rows["native"]["final_latent"],
        "sham_png_exact_native":rows["sham"]["image"]["sha256"] == rows["native"]["image"]["sha256"]}
    summary = {"schema":"mdb-flux-initial-noise-result-v1","job_id":request["job_id"],"request_sha256":request["request_sha256"],
        "counts":dict(counts),"mathematical_steps":168,"seals":seals,"transaction_reports":reports,
        "transaction_verification":verifications,"target_calls":state["target_calls"],"target_calls_by_step":state["target_calls_by_step"],
        "blocked_entries":state["blocked"],"branch_execution_order":ORDER,"rows":[rows[name] for name in ORDER],
        "first_images":first_images,"control_observations":controls,"parameter_artifacts":recipes,
        "model_identity":adapter.model_identity,"execution":dict(adapter.execution),"worker_seconds":time.monotonic()-started,
        "mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary)
    return normal(summary)


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
        raise ValueError("initial-noise request/source/transport identity differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path("flux-initial-noise-result")
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_FLUX_INITIAL_NOISE_BOOTSTRAP "+json.dumps({"job_id":request["job_id"],"seconds":time.monotonic()-started}),flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb_lanes import flux_derived_terminal, parameter_artifacts
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
        raise ValueError("initial-noise requires admitted CUDA")
    reference = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    source_abi = verify_source_abi(boundary=reference,reentry=reentry,closure_type=FluxFinalLatentClosureAdapter,
        config=config)
    save(output / "pre-model-source-abi.json",source_abi)
    scalar_qualification = reference["qualify_torch_scalar_promotion"](torch,device="cuda")
    save(output / "cuda-scalar-qualification.json",scalar_qualification)
    disk = shutil.disk_usage(output)
    if disk.free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("initial-noise output filesystem lacks declared10GiB free disk")
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
        print("MDB_FLUX_INITIAL_NOISE_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        now = time.monotonic()
        if now-started > config["max_wall_seconds"]:
            raise RuntimeError("finite initial-noise wall budget exhausted")
        if now-heartbeat["last"] >= 15:
            heartbeat["last"] = now
            event("heartbeat")
    array = step()["make_array_writer"](output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("initial-noise checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("initial-noise requires deterministic distilled five-joint/twenty-single Klein")
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
    implementations[inspect.getfile(parameter_artifacts)] = sha(inspect.getfile(parameter_artifacts))
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
            raise ValueError("initial-noise native schedule differs")
        session = adapter.session(latent=latent,conditioning=conditioning,img_ids=img_ids,txt_ids=txt_ids,
            timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
    source = {"schema":"mdb-flux-initial-noise-source-v1","job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_hashes":request["source_hashes"],"context":CONTEXT,
        "encoding":encoding,"encodings":encoding,"model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "checkpoint_sha256":identity(inventory),"checkpoint_files":inventory,"implementation_sources":implementations,
        "versions":{name:importlib.metadata.version(name) for name in ("torch","numpy","diffusers","transformers","saturn-pub","mdb-runtime")},
        "counts_before_initial_origin":dict(counts),"cuda_scalar_qualification":{"path":"cuda-scalar-qualification.json",
            "sha256":sha(output / "cuda-scalar-qualification.json")},
        "pre_model_source_abi":{"path":"pre-model-source-abi.json","sha256":sha(output / "pre-model-source-abi.json")},
        "scope":config["scope"]}
    checkpoint = CheckpointParameters(root,config["parameter_checkpoint_file"],inventory[config["parameter_checkpoint_file"]],pipe.transformer)
    summary = run(model=pipe.transformer,adapter=adapter,session=session,output=output,config=config,request=request,
        source=source,counts=counts,checkpoint=checkpoint,array=array,guard_components=(pipe.transformer,pipe.vae),wall=wall,event=event,started=started)
    if counts != COUNTS:
        raise ValueError("initial-noise global load/encoding/noise census differs")
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    manifest = step()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"],
        max_expanded_bytes=config["transport_limits"]["max_expanded_bytes"],
        max_member_bytes=config["transport_limits"]["max_member_bytes"])
    print("MDB_FLUX_INITIAL_NOISE_COMPLETE "+json.dumps(json_value({"summary":summary,"manifest":manifest}),sort_keys=True),flush=True)
    return output,summary


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-initial-noise-result")
        if output.is_dir() and not (output/"remote-manifest.json").exists():
            util()["save"](output/"failure.json",{"schema":"mdb-flux-initial-noise-failure-v1",
                "job_id":os.environ.get("MRUN_JOB_ID"),"error":f"{type(error).__name__}: {error}",
                "mechanics_status":"errored","scope":"Completed evidence preserved; no model retry, archive or fabricated report."})
        raise
