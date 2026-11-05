"""Seal a full remaining denoising step, then consume its predicted next latent.

The frozen complete-block reference is unchanged. Native state is observed only
through step2/pre6 before the seal; every later forecast stage consumes its own
complete predicted predecessor. Parameter archives are loaded one block at a
time. This worker returns evidence and delegates transport to its caller.
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
from pathlib import Path

CONTEXT = {"id": "p1-s89", "prompt_id": "p1", "seed": 89,
    "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BLOCK_INDICES = list(range(6, 20))
SAMPLE_INDICES = [0, 31, 511, 512, 543, 767, 1023, 1535]
IMAGE_SAMPLE_INDICES = [0, 31, 32, 255, 511, 512, 767, 1023]
BRANCHES = [{"id": "native", "mode": "native"}, {"id": "sham", "mode": "sham"},
    {"id": "predicted", "mode": "predicted"}]
ORDER = [row["id"] for row in BRANCHES]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 168,
    "acts": 2, "candidate_calls": 0, "receipt_emissions": 170, "images": 3, "vae_calls": 3, "physical_batch": 1}
OBSERVATION_PLAN = {"reference_rows": ["native"], "boundaries_per_branch": 29,
    "slots": ["image", "text", "latent", "noise_prediction"], "witness_count": 64,
    "reference_storage": "lossless-bf16-bits", "candidate_storage": "descriptor-and-fixed-coordinate-witness",
    "scope": "Initial next-step project plus28 native exits. All reference arrays retained; intermediate candidate full norms are worker observations with fixed-coordinate witnesses, not independently auditable full arrays."}
TRANSITION_REFERENCE_SHA256 = "f531e059635fe9940d55c38eaf5cc1d7ab9dd3a4cd1adeee5225ba9342f3bf83"
NATIVE_ADAPTER_MODULE_SHA256 = "2004b81971a258f1ca6295fe69d24767dc6bb6b92e103fa30cf6ccb0249b5a86"


def _util():
    return runpy.run_path(str(Path(__file__).with_name("flux_single_contexts.py")))


def verify_native_source_contract(*, boundary, norm_type, scheduler_type, adapter_type):
    """Refuse a native implementation outside the reference's exact source ABI."""
    pins = boundary["SOURCE_CONTRACT"]
    expected = {key:pins[key] for key in ("AdaLayerNormContinuous.module_sha256",
        "AdaLayerNormContinuous.forward_source_sha256","FlowMatchEulerDiscreteScheduler.module_sha256",
        "FlowMatchEulerDiscreteScheduler.step_source_sha256")}
    expected["Flux2KleinAdapter.module_sha256"] = NATIVE_ADAPTER_MODULE_SHA256
    sha = _util()["sha"]
    actual = {"AdaLayerNormContinuous.module_sha256":sha(inspect.getfile(norm_type)),
        "AdaLayerNormContinuous.forward_source_sha256":hashlib.sha256(inspect.getsource(norm_type.forward).encode()).hexdigest(),
        "FlowMatchEulerDiscreteScheduler.module_sha256":sha(inspect.getfile(scheduler_type)),
        "FlowMatchEulerDiscreteScheduler.step_source_sha256":hashlib.sha256(inspect.getsource(scheduler_type.step).encode()).hexdigest(),
        "Flux2KleinAdapter.module_sha256":sha(inspect.getfile(adapter_type))}
    if actual != expected:
        mismatched = [key for key in expected if actual[key] != expected[key]]
        raise ValueError("step-boundary native source ABI differs: "+", ".join(mismatched))
    return {"schema":"mdb-flux-step-boundary-native-source-abi-v1","verified":True,
        "expected":expected,"actual":actual,"model_calls":0,
        "types":{name:kind.__module__+"."+kind.__qualname__ for name,kind in
            (("norm",norm_type),("scheduler",scheduler_type),("adapter",adapter_type))},
        "scope":"Exact native Python implementation bytes/method sources bound to the mathematical reference; no claim of independent CUDA kernel byte closure."}


def contract(config=None):
    """One finite physical geometry; returned objects never alias constants."""
    if config is not None and config.get("block_indices", BLOCK_INDICES) != BLOCK_INDICES:
        raise ValueError("only current-step blocks6..19 are supported")
    return json.loads(json.dumps({"context": CONTEXT, "block_indices": BLOCK_INDICES,
        "branches": BRANCHES, "order": ORDER, "counts": COUNTS,
        "prefix_steps": 68, "reference_steps": 16, "suffix_steps": 28,
        "panel_counts": {"adapter_steps": 100, "acts": 2, "receipt_emissions": 102,
            "candidate_calls": 0, "images": 3, "vae_calls": 3}, "observation_plan": OBSERVATION_PLAN,
        "target_calls": {**{str(index): 4 for index in BLOCK_INDICES}, "norm_out": 4, "proj_out": 4}}))


def validate_config(config):
    expected = contract(config)
    fixed = {"context": expected["context"], "block_indices": BLOCK_INDICES, "branches": BRANCHES,
        "branch_execution_order": ORDER, "height": 512, "width": 512, "steps": 4,
        "selector": {"logical_step": 2, "phase": "single", "index": 6},
        "sample_indices": SAMPLE_INDICES, "image_sample_indices": IMAGE_SAMPLE_INDICES,
        "query_chunk": 64, "projection_chunk": 256, "neuron_chunk": 256, "retain_fields": False,
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64", "cpu_threads": 4,
        "observation_plan": OBSERVATION_PLAN, "transaction_max_bytes":768<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":8<<30,"max_member_bytes":1<<30}}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("step-boundary frozen scientific geometry differs")
    for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("step-boundary finite physical census differs")
    if (type(config.get("cpu_threads")) is not int or type(config.get("max_wall_seconds")) is not int
            or not 1 <= config["max_wall_seconds"] <= 900):
        raise ValueError("step-boundary execution budget differs")


def _record_receipt(output, store, counts, active, receipt):
    body = receipt.to_dict()
    counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
    counts["receipt_emissions"] += 1
    if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
        raise RuntimeError("step-boundary finite physical budget exhausted")
    store.receipt(receipt)
    with (output / "receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps({"sequence": counts["receipt_emissions"]-1, "branch": active,
            "receipt": receipt.fingerprint}, sort_keys=True)+"\n")


def run_prefix(*, adapter, session, output, store, counts, wall=lambda: None):
    """Exactly68 native transitions on freshly encoded C and initial noise."""
    from mdb.native_navigation import NativeCutNavigator

    initial = session.capture()
    store.save(initial)
    if (initial.execution_point["logical_step"] != 0 or initial.execution_point["phase"] != "project"
            or initial.payload["step"] != 0 or initial.payload["cursor"] != 0):
        raise ValueError("step-boundary prefix requires the fresh native initial state")
    nav = NativeCutNavigator(session, max_steps=68,
        on_receipt=lambda value: _record_receipt(output, store, counts, "prefix", value))
    path = []
    for _ in range(68):
        wall()
        path.append(nav.step().to_dict())
    pre6 = nav.capture()
    if any(pre6.execution_point[key] != value for key, value in
            {"logical_step": 2, "phase": "single", "index": 6, "local_clock": 6}.items()):
        raise ValueError("native prefix did not end at step2/pre6")
    store.save(pre6)
    record = {"schema": "mdb-flux-step-boundary-prefix-v1", "initial": initial.fingerprint,
        "pre6": pre6.fingerprint, "point": dict(pre6.execution_point), "receipts": path,
        "counts": {"adapter_steps": 68, "acts": 0, "receipt_emissions": 68},
        "scope": "One clean native seed89 history; no engineered low/high source or imported state."}
    _util()["save"](output / "prefix.json", record)
    return pre6, record


def bind_inputs(*, model, cut, array, counts, prefix_ref, config, profile, production_profile=True):
    """Archive native operands without retaining fourteen widened weight banks."""
    import torch
    from saturn_pub.values import describe

    from mdb.forward_recording import tensor_descriptor

    if any(cut.execution_point[key] != value for key, value in config["selector"].items()):
        raise ValueError("step-boundary binding frontier differs")
    if production_profile and (tuple(cut.payload["image"].shape) != (1,1024,3072)
            or tuple(cut.payload["text"].shape) != (1,512,3072)
            or tuple(cut.payload["latent"].shape) != (1,1024,128)
            or any(tuple(value.shape) != (1536,128) or value.dtype != torch.float32 for value in cut.payload["context"]["rope"])):
        raise ValueError("qualified production source dimensions/RoPE differ")
    def parameter(value):
        if value is None:
            return None
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("step-boundary native parameter must be finite BF16")
        return array(value)
    binding = {"schema": "mdb-flux-step-boundary-binding-v1", "input_cut": cut.fingerprint,
        "point": dict(cut.execution_point), "prefix": prefix_ref, "block_indices": BLOCK_INDICES,
        "counts_before_seal": dict(counts), "target_entries_before_seal": {key: 0 for key in contract()["target_calls"]},
        "inputs": {name: array(cut.payload[name]) for name in ("image", "text", "latent", "timesteps", "sigmas")},
        "context": {name: array(cut.payload["context"][name]) for name in ("single_mod", "temb")},
        "rope": {name: array(value) for name, value in zip(("cos", "sin"), cut.payload["context"]["rope"], strict=True)},
        "text_rows": cut.payload["text"].shape[1], "blocks": {},
        "carried_state": describe({key: value for key, value in cut.payload.items()
            if key not in {"image", "text", "latent", "context", "noise_prediction", "step", "phase", "cursor"}}),
        "clock_state": {key: cut.payload[key] for key in ("step", "phase", "cursor")}}
    for index in BLOCK_INDICES:
        block = model.single_transformer_blocks[index]
        attn = block.attn
        h, m = attn.inner_dim, attn.mlp_hidden_dim
        if (block.norm.elementwise_affine or block.norm.eps != 1e-6 or attn.norm_q.eps != attn.norm_k.eps
                or not attn.norm_q.elementwise_affine or not attn.norm_k.elementwise_affine
                or attn.norm_q.eps != 1e-6 or h != attn.heads*attn.head_dim
                or tuple(attn.to_qkv_mlp_proj.weight.shape) != (3*h+2*m,h)
                or tuple(attn.to_out.weight.shape) != (h,h+m)):
            raise ValueError("step-boundary complete block profile differs")
        if production_profile and (h != 3072 or m != 9216 or attn.heads != 24 or attn.head_dim != 128):
            raise ValueError("qualified production single-block dimensions differ")
        params = {"qkv_mlp_weight": attn.to_qkv_mlp_proj.weight, "qkv_mlp_bias": attn.to_qkv_mlp_proj.bias,
            "to_out_weight": attn.to_out.weight, "to_out_bias": attn.to_out.bias,
            "norm_q_weight": attn.norm_q.weight, "norm_k_weight": attn.norm_k.weight}
        binding["blocks"][str(index)] = {"parameters": {name: parameter(value) for name, value in params.items()},
            "native_parameters": {name: tensor_descriptor(value) for name, value in block.named_parameters()},
            "heads": attn.heads, "hidden_width": h, "mlp_width": m, "head_dim": attn.head_dim,
            "layernorm_eps": block.norm.eps, "qk_eps": attn.norm_q.eps, "attention_profile": profile(block)}
    readout = model.norm_out
    if (not isinstance(readout.norm,torch.nn.LayerNorm) or readout.norm.elementwise_affine
            or readout.norm.eps != 1e-6 or not isinstance(readout.silu, torch.nn.SiLU)):
        raise ValueError("step-boundary readout profile differs")
    params = {"modulation_weight": readout.linear.weight, "modulation_bias": readout.linear.bias,
        "projection_weight": model.proj_out.weight, "projection_bias": model.proj_out.bias}
    binding["readout"] = {"parameters": {name: parameter(value) for name, value in params.items()},
        "native_parameters": {name: tensor_descriptor(value) for name, value in model.norm_out.named_parameters()},
        "projection_native_parameters": {name: tensor_descriptor(value) for name, value in model.proj_out.named_parameters()},
        "eps": readout.norm.eps, "elementwise_affine": False}
    binding["commit"] = {"step_index": 2, "sigma": array(cut.payload["sigmas"][2]),
        "sigma_next": array(cut.payload["sigmas"][3]),
        "reset": {"image": None, "text": None, "context": None, "noise_prediction": None},
        "next_clock": {"step": 3, "phase": "project", "cursor": 0},
        "scope": "Only latent is numerically predicted at commit. Remaining carried state is preserved; working state is explicitly reset by the native deterministic commit law."}
    return binding


def predict_step(binding, *, archive_root, transition, boundary, config, event=lambda *args, **kwargs: None):
    """Only archived inputs and math; never a model/module/native continuation."""
    import numpy as np

    from mdb.tensor_archive import load_tensor_archive

    def load(record):
        return None if record is None else load_tensor_archive(archive_root, record)
    current = np.concatenate((load(binding["inputs"]["text"]), load(binding["inputs"]["image"])),axis=1)
    current.setflags(write=False)
    single_mod = load(binding["context"]["single_mod"])
    cosine, sine = (load(binding["rope"][name]) for name in ("cos", "sin"))
    steps = []
    for position,index in enumerate(BLOCK_INDICES):
        spec = binding["blocks"][str(index)]
        parameters = {name: load(value) for name,value in spec["parameters"].items()}
        prediction = transition["predict_transition"](current, single_mod, **parameters,
            heads=spec["heads"], layernorm_eps=spec["layernorm_eps"], qk_eps=spec["qk_eps"],
            rope_cos=cosine, rope_sin=sine, query_chunk=config["query_chunk"],
            projection_chunk=config["projection_chunk"], neuron_chunk=config["neuron_chunk"],
            retain_fields=False, text_rows=None, sample_indices=config["sample_indices"])
        del parameters
        if position and prediction["input_descriptor"] != steps[-1]["prediction"]["output_descriptor"]:
            raise ValueError("step-boundary prediction refreshed its own predecessor")
        steps.append({"step": position, "block_index": index,
            "input_source": "observed native step2/pre6" if position == 0 else "previous complete predicted output",
            "input_descriptor": prediction["input_descriptor"], "prediction": prediction})
        current = prediction["output"]
        event("block_forecast_complete", block_index=index)
        gc.collect()
    readout = boundary["predict_readout"](current[:,binding["text_rows"]:], load(binding["context"]["temb"]),
        **{name: load(value) for name,value in binding["readout"]["parameters"].items()},
        eps=binding["readout"]["eps"], neuron_chunk=config["projection_chunk"],
        sample_indices=config["image_sample_indices"], retain_fields=False)
    commit = boundary["predict_flow_commit"](load(binding["inputs"]["latent"]), readout["noise"],
        load(binding["commit"]["sigma"]), load(binding["commit"]["sigma_next"]))
    return {"steps": steps, "chain_length": len(steps), "input_descriptor": steps[0]["input_descriptor"],
        "output_descriptor": steps[-1]["prediction"]["output_descriptor"], "readout": readout, "commit": commit,
        "next_state_contract": {"clock": binding["commit"]["next_clock"], "reset": binding["commit"]["reset"],
            "carried_state": binding["carried_state"]},
        "scope": "Fourteen complete free block predictions then own predicted image readout and next latent commit; no native intermediate, native noise or native Z3 enters this forecast."}


def _descriptor(value):
    import numpy as np

    if (not isinstance(value,np.ndarray) or value.dtype != np.float32 or not np.isfinite(value).all()
            or np.any(np.ascontiguousarray(value).view(np.uint32)&np.uint32(65535))):
        raise ValueError("sealed arrays must preserve finite exact BF16 bits")
    value = np.ascontiguousarray(value)
    return {"shape":list(value.shape),"dtype":"float32-exact-bf16-widening",
        "fp32_le_sha256":hashlib.sha256(value.astype("<f4",copy=False).tobytes()).hexdigest(),
        "bf16_le_sha256":hashlib.sha256((value.view(np.uint32)>>16).astype("<u2").tobytes()).hexdigest()}


def validate_prediction(pre6, prediction):
    """Bind sealed full-array identities without a native future or repair cast."""
    import torch

    normal = _util()["json_value"]
    initial = torch.cat((pre6.payload["text"],pre6.payload["image"]),dim=1).float().cpu().numpy()
    expected = _descriptor(initial)
    if (prediction["chain_length"] != 14 or len(prediction["steps"]) != 14
            or normal(prediction["input_descriptor"]) != expected):
        raise ValueError("sealed step chain source/length differs")
    for position,step in enumerate(prediction["steps"]):
        value = step["prediction"]
        actual = _descriptor(value["output"])
        if (step["block_index"] != BLOCK_INDICES[position] or step["step"] != position
                or step["input_source"] != ("observed native step2/pre6" if position == 0 else "previous complete predicted output")
                or normal(step["input_descriptor"]) != expected or normal(value["input_descriptor"]) != expected
                or normal(value["output_descriptor"]) != actual or value["output"].shape != initial.shape
                or "text" in value or "image" in value):
            raise ValueError("sealed step chain own-state join or complete array identity differs")
        expected = actual
    readout,commit = prediction["readout"],prediction["commit"]
    image = prediction["steps"][-1]["prediction"]["output"][:,pre6.payload["text"].shape[1]:]
    if (normal(prediction["output_descriptor"]) != expected
            or normal(readout["input_descriptor"]) != _descriptor(image)
            or normal(readout["noise_descriptor"]) != _descriptor(readout["noise"])
            or normal(commit["noise_descriptor"]) != _descriptor(readout["noise"])
            or normal(commit["input_descriptor"]) != _descriptor(pre6.payload["latent"].float().cpu().numpy())
            or normal(commit["latent_descriptor"]) != _descriptor(commit["latent"])
            or commit["latent"].shape != tuple(pre6.payload["latent"].shape)):
        raise ValueError("sealed readout/commit array identity or own-state join differs")


def run_panel(*, adapter, pre6, output, config, store, array, wall, event, counts, prediction):
    """Native16 reference then native/sham/predicted next-step inputs and3×28."""
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb.native_navigation import NativeCutNavigator

    util = _util()
    observer_class = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))["ContinuationObserver"]
    def navigator(cut, branch, limit):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=limit,
            on_receipt=lambda value: _record_receipt(output, store, counts, branch, value))
    def boundary(cut):
        values = {name: array(cut.payload[name]) for name in ("image", "text", "latent", "noise_prediction")
            if isinstance(cut.payload[name], torch.Tensor)}
        return {"cut": cut.fingerprint, "point": dict(cut.execution_point), "arrays": values,
            "clock": {name: cut.payload[name] for name in ("step", "phase", "cursor")}}
    reference = navigator(pre6, "reference", 16)
    native, paths = [], []
    for index in range(16):
        wall()
        receipt = reference.step()
        paths.append(receipt.to_dict())
        cut = reference.capture()
        store.save(cut)
        native.append({"index": index, "receipt_id": receipt.fingerprint, **boundary(cut)})
        event("reference_transition_complete", index=index, point=dict(cut.execution_point))
    z3 = reference.capture()
    if (z3.execution_point["logical_step"] != 3 or z3.execution_point["phase"] != "project"
            or z3.payload["cursor"] != 0 or any(z3.payload[name] is not None for name in ("image", "text", "context", "noise_prediction"))):
        raise ValueError("reference did not produce the exact next-step commit/reset boundary")
    carried = lambda cut: describe({key: value for key,value in cut.payload.items()
        if key not in {"image", "text", "latent", "context", "noise_prediction", "step", "phase", "cursor"}})
    if carried(pre6) != carried(z3):
        raise ValueError("native commit changed carried conditioning/IDs/schedule")
    util["save"](output / "native-segment.json", {"input_cut": pre6.fingerprint, "boundaries": native,
        "receipts": paths, "next_input": boundary(z3), "counts": {"adapter_steps":16,"acts":0,"receipt_emissions":16}})
    validate_prediction(pre6,prediction)
    # A cast is not permission to repair a corrupted sealed forecast.
    forecast = prediction["commit"]["latent"]
    if (not isinstance(forecast,np.ndarray) or forecast.dtype != np.float32
            or forecast.shape != tuple(z3.payload["latent"].shape) or not np.isfinite(forecast).all()
            or np.any(np.ascontiguousarray(forecast).view(np.uint32)&np.uint32(65535))):
        raise ValueError("sealed Z3 must preserve finite exact BF16 bits and complete latent geometry")
    observer = observer_class(array=array, plan=config["observation_plan"])
    rows = []
    for spec in config["branches"]:
        name, mode = spec["id"], spec["mode"]
        nav = navigator(z3, name, 28)
        row = {**spec, "parent": z3.fingerprint, "acts": [], "continuation_receipts": []}
        if mode != "native":
            value = z3.payload["latent"] if mode == "sham" else torch.from_numpy(np.array(forecast,copy=True)).to(
                device=z3.payload["latent"].device,dtype=z3.payload["latent"].dtype)
            row["acts"].append(util["replace_slot"](nav,"latent",value,name=name+".next-step-input",store=store))
        start = nav.capture()
        if describe({key:value for key,value in start.payload.items() if key != "latent"}) != describe(
                {key:value for key,value in z3.payload.items() if key != "latent"}):
            raise ValueError("next-step latent Act changed undeclared carried/reset/clock state")
        row["next_input"] = boundary(start)
        def observe(current,index,incoming):
            entry = observer.observe(name,current,index=index,incoming_receipt=incoming)
            entry["complete_cut_retained"] = index in {0,28}
            for slot,saved in entry["slots"].items():
                point = current.execution_point
                if slot in {"image","text"}:
                    active = point["phase"] in {"joint","single","readout"}
                    saved["meaning"] = {"phase":"working-next-step-stream" if active else "absent-or-stale-working-stream",
                        "meaningful_propagation_observable": active and saved["present"], "stale":not active}
                elif slot == "latent" and index == 0:
                    saved["meaning"] = {"phase":"next-step-input-Z3", "meaningful_propagation_observable":True,
                        "constant_before_commit":True,"stale":False}
            return entry
        observe(start,0,(row["acts"][-1]["receipt_id"] if row["acts"] else paths[-1]["fingerprint"]))
        for index in range(28):
            wall()
            receipt = nav.step()
            row["continuation_receipts"].append(receipt.to_dict())
            observe(nav.capture(),index+1,receipt.fingerprint)
        final = nav.capture()
        if final.execution_point["logical_step"] != 4 or final.execution_point["phase"] != "project":
            raise ValueError("next-step native suffix did not finish at Z4")
        store.save(final)
        counts["vae_calls"] += 1
        image = adapter.decode(nav.session,height=512,width=512)
        image_path = output / (name+".png")
        image.save(image_path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint,final_latent=array(final.payload["latent"]),
            image={"path":image_path.name,"sha256":util["sha"](image_path)})
        util["save"](output / "rows" / (name+".json"),row)
        rows.append(row)
        event("branch_complete",row=name)
    if set(observer.rows) != set(ORDER) or any(len(value) != 29 for value in observer.rows.values()):
        raise ValueError("next-step reference observation census differs")
    util["save"](output / "continuation-observations.json", {"schema":"mdb-flux-step-boundary-continuation-v1",
        "plan":config["observation_plan"],"rows":observer.rows,"branch_execution_order":ORDER,"model_calls_added":0})
    controls = {"sham_final_latent_exact_native": rows[0]["final_latent"] == rows[1]["final_latent"],
        "sham_png_exact_native": rows[0]["image"]["sha256"] == rows[1]["image"]["sha256"]}
    return {"rows":rows,"counts":dict(counts),"control_observations":controls,
        "native_segment_sha256":util["sha"](output / "native-segment.json"),
        "continuation_observations_sha256":util["sha"](output / "continuation-observations.json")}


def make_array_writer(output, wall=lambda: None):
    """Lossless deduplicated native tensor archives; no widened weight cache."""
    from mdb.forward_recording import tensor_descriptor
    from mdb.tensor_archive import write_tensor_archive

    util, archived = _util(), {}
    def array(value):
        wall()
        descriptor = tensor_descriptor(value)
        key = util["identity"]({name:descriptor[name] for name in ("dtype","shape","sha256")})
        if key not in archived:
            archived[key] = write_tensor_archive(output / "arrays" / (key+".npy"),value,root=output)
        return archived[key]
    return array


def run(*, model, adapter, session, output, config, request, source, counts, array=None,
        guard_components=None, wall=lambda: None, event=lambda *args, **kwargs: None, started=None,
        production_profile=True):
    """Resident-model experiment entry; emits no archive and submits no jobs."""
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    started = time.monotonic() if started is None else started
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output / "plan.json").exists():
        raise ValueError("immutable step-boundary output already contains a plan")
    util = _util()
    save, sha, identity = (util[name] for name in ("save","sha","identity"))
    array = array or make_array_writer(output,wall)
    save(output / "plan.json",config)
    save(output / "plan-frozen.json",{"plan_sha256":sha(output / "plan.json"),
        "request_sha256":request["request_sha256"],"job_id":request["job_id"],
        "source_identity":identity(request["source_hashes"]),"counts":dict(counts),
        "scope":"Frozen before the68-transition source prefix; no native future provided."})
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    boundary = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    source_abi = verify_native_source_contract(boundary=boundary,norm_type=type(model.norm_out),
        scheduler_type=FlowMatchEulerDiscreteScheduler,adapter_type=type(adapter))
    save(output / "native-source-abi.json",source_abi)
    source = {**source,"native_source_abi":{"path":"native-source-abi.json","sha256":sha(output / "native-source-abi.json")}}
    store = LocalStore(output / "typed-store")
    pre6,prefix = run_prefix(adapter=adapter,session=session,output=output,store=store,counts=counts,wall=wall)
    prefix_ref = {"path":"prefix.json","sha256":sha(output / "prefix.json")}
    source = {**source,"prefix":prefix_ref}
    save(output / "source.json",source)
    old = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))
    transition_path = Path(__file__).with_name("flux_transition_reference.py")
    if sha(transition_path) != TRANSITION_REFERENCE_SHA256:
        raise ValueError("frozen complete-block reference changed")
    transition = runpy.run_path(str(transition_path))
    release = {"native":False,"blocked":0,"targets":{key:0 for key in contract()["target_calls"]}}
    targets = {model.single_transformer_blocks[index]:str(index) for index in BLOCK_INDICES}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out"})
    def guard(module,args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("native component entered before full step-boundary forecast seal")
        if module in targets:
            release["targets"][targets[module]] += 1
    handles = [module.register_forward_pre_hook(guard) for component in (guard_components or (model,))
        for module in component.modules()]
    try:
        binding = bind_inputs(model=model,cut=pre6,array=array,counts=counts,prefix_ref=prefix_ref,
            config=config,profile=old["attention_profile"],production_profile=production_profile)
        save(output / "binding.json",binding)
        tx = ForecastTransaction(output / "transaction",model_identity=adapter.model_identity,
            input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
            frontier={"kind":"architectural","binding":binding},execution_scope=config["scope"],
            budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000,
                "max_bytes":config["transaction_max_bytes"]})
        with tx:
            tx.frontier("source_binding",binding)
            event("prediction_start")
            tx.predict("step_boundary",lambda _: {"predictions":predict_step(binding,archive_root=output,
                transition=transition,boundary=boundary,config=config,event=event)})
            tx.predict("hypotheses",lambda _: {"predictions":config["hypotheses"]})
            seal = tx.seal()
            event("step_boundary_sealed",seal=seal,target_entries_before_seal=dict(release["targets"]),
                blocked_entries=release["blocked"])
            release["native"] = True
            tx.execute("native",lambda context:run_panel(adapter=adapter,pre6=pre6,output=output,config=config,
                store=store,array=array,wall=wall,event=event,counts=counts,
                prediction=context.predictions["step_boundary"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output / "transaction-report.json",tx.report())
    if (any(counts[name] != COUNTS[name] for name in ("adapter_steps","acts","receipt_emissions","images","vae_calls","candidate_calls"))
            or release["blocked"] != 0 or release["targets"] != contract()["target_calls"]):
        raise ValueError("step-boundary completed physical/target census differs")
    summary = {"schema":"mdb-flux-step-boundary-result-v1","job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"counts":dict(counts),"prefix_counts":prefix["counts"],
        "panel_counts":contract()["panel_counts"],"target_calls":release["targets"],"blocked_entries":release["blocked"],
        "seal":seal,"model_identity":adapter.model_identity,"transaction_verification":verify_transaction(tx.root),
        "control_observations":tx.observation("native")["control_observations"],
        "worker_seconds":time.monotonic()-started,
        "mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output / "summary.json",summary)
    return util["json_value"](summary)


def finalize_manifest(output, *, job_id, request_sha256, max_expanded_bytes=8<<30, max_member_bytes=1<<30, max_files=20000):
    """Finalize a regular-file-only tree once, before authenticated collection."""
    output = Path(output)
    util = _util()
    if (output.is_symlink() or any(type(value) is not int or value < 1
            for value in (max_expanded_bytes,max_member_bytes,max_files))):
        raise ValueError("invalid output custody root/budgets")
    if (output / "remote-manifest.json").exists() or (output / "files.json").exists():
        raise ValueError("immutable final manifest already exists")
    def inventory():
        result = {}
        total = 0
        for path in sorted(output.rglob("*")):
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise ValueError("output custody refuses symlink/nonregular file")
            if path.is_dir():
                continue
            size = path.stat().st_size
            total += size
            if size > max_member_bytes or total > max_expanded_bytes or len(result) >= max_files:
                raise ValueError("step-boundary output exceeds finite file/member/expanded bound")
            result[path.relative_to(output).as_posix()] = {"sha256":util["sha"](path),"bytes":size,"filetype":"regular"}
        return result,total
    files,_ = inventory()
    util["save"](output / "files.json",{name:record["sha256"] for name,record in files.items()})
    files,total = inventory()
    manifest = {"schema":"mdb-step-boundary-remote-manifest-v1","job_id":job_id,
        "request_sha256":request_sha256,"files":files}
    # Reserve the manifest's own bytes inside the declared expanded bound too.
    encoded = json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    if total+len(encoded) > max_expanded_bytes or len(encoded) > max_member_bytes or len(files)+1 > max_files:
        raise ValueError("final remote manifest exceeds finite custody bound")
    with (output / "remote-manifest.json").open("xb") as stream:
        stream.write(encoded)
    return {"manifest_sha256":util["sha"](output / "remote-manifest.json"),
        "files":len(files),"expanded_bytes_excluding_manifest":total,"manifest_bytes":len(encoded)}


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
        raise ValueError("step-boundary request/source/transport identity differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path("flux-step-boundary-result")
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_FLUX_STEP_BOUNDARY_BOOTSTRAP "+json.dumps({"job_id":request["job_id"],"seconds":time.monotonic()-started}),flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.normalization import AdaLayerNormContinuous
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("step-boundary requires admitted CUDA")
    reference = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    source_abi = verify_native_source_contract(boundary=reference,norm_type=AdaLayerNormContinuous,
        scheduler_type=FlowMatchEulerDiscreteScheduler,adapter_type=Flux2KleinAdapter)
    save(output / "pre-model-source-abi.json",source_abi)
    scalar_qualification = reference["qualify_torch_scalar_promotion"](torch,device="cuda")
    save(output / "cuda-scalar-qualification.json",scalar_qualification)
    disk = shutil.disk_usage(output)
    if disk.free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("step-boundary output filesystem lacks declared8GiB free disk")
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
        print("MDB_FLUX_STEP_BOUNDARY_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        now = time.monotonic()
        if now-started > config["max_wall_seconds"]:
            raise RuntimeError("finite step-boundary wall budget exhausted")
        if now-heartbeat["last"] >= 15:
            heartbeat["last"] = now
            event("heartbeat")
    array = make_array_writer(output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("step-boundary checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("step-boundary requires deterministic distilled five-joint/twenty-single Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind):sha(inspect.getfile(kind)) for kind in (type(pipe),type(pipe.text_encoder),
        type(pipe.transformer),type(pipe.transformer.single_transformer_blocks[6]),
        type(pipe.transformer.single_transformer_blocks[6].attn.processor),type(pipe.transformer.norm_out),
        type(pipe.vae),type(pipe.scheduler),Flux2KleinAdapter)}
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
    adapter = Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    with torch.inference_mode():
        conditioning,txt_ids = conditioning.to("cuda"),txt_ids.to("cuda")
        latent,img_ids = pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,conditioning.dtype,
            torch.device("cuda"),torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
        timesteps,steps = retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
        if steps != 4:
            raise ValueError("step-boundary native schedule differs")
        session = adapter.session(latent=latent,conditioning=conditioning,img_ids=img_ids,txt_ids=txt_ids,
            timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
    source = {"schema":"mdb-flux-step-boundary-source-v1","job_id":request["job_id"],
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
        raise ValueError("step-boundary global load/encoding/noise census differs")
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    manifest = finalize_manifest(output,job_id=request["job_id"],request_sha256=request["request_sha256"])
    print("MDB_FLUX_STEP_BOUNDARY_COMPLETE "+json.dumps(json_value({"summary":summary,"manifest":manifest}),sort_keys=True),flush=True)
    return output,summary


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-step-boundary-result")
        if output.is_dir() and not (output / "remote-manifest.json").exists():
            _util()["save"](output / "failure.json",{"schema":"mdb-flux-step-boundary-failure-v1",
                "job_id":os.environ.get("MRUN_JOB_ID"),"error":f"{type(error).__name__}: {error}",
                "mechanics_status":"errored","scope":"Completed evidence preserved; no model retry or fabricated transport report."})
        raise
