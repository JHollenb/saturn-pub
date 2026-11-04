"""Seal bounded complete FLUX chains before native current-step entries."""
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

CONTEXT = {"id": "p1-s79", "prompt_id": "p1", "seed": 79,
    "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
SAMPLE_INDICES = [0, 31, 511, 512, 543, 767, 1023, 1535]
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 212,
    "acts": 14, "candidate_calls": 0, "receipt_emissions": 226, "images": 7, "vae_calls": 7, "physical_batch": 1}
BRANCHES = [
    {"id": "low.native", "base": "zero", "mode": "native"},
    {"id": "high.native", "base": "text", "mode": "native"},
    {"id": "low.sham", "base": "zero", "mode": "sham"},
    {"id": "low.pred1", "base": "zero", "mode": "pred1"},
    {"id": "low.pred2", "base": "zero", "mode": "pred2"},
    {"id": "high.pred1", "base": "text", "mode": "pred1"},
    {"id": "high.pred2", "base": "text", "mode": "pred2"},
]
ORDER = [row["id"] for row in BRANCHES]
OBSERVATION_PLAN = {"reference_rows": ["low.native", "high.native"], "boundaries_per_branch": 15,
    "slots": ["image", "text", "latent", "noise_prediction"], "witness_count": 64,
    "reference_storage": "lossless-bf16-bits", "candidate_storage": "descriptor-and-fixed-coordinate-witness",
    "witness_rule": "unique floor(linspace(0,numel-1,min(64,numel))) in C-value order",
    "active_slots": "Working I/T through readout entry; native readout noise at scheduler entry; committed Z4 only after scheduler. Stored Z3 and post-readout I/T are stale.",
    "full_candidate_cuts": "Actual common post7 input and final committed Z4, plus both typed installation cuts and pred1 post6 input. No complete intermediate candidate cut claim."}


def contract(config):
    """Pure finite profile; two-block constants and serialized defaults persist."""
    indices = config.get("block_indices", [6, 7])
    if (type(indices) is not list or any(type(index) is not int for index in indices)
            or indices not in ([6, 7], [6, 7, 8])):
        raise ValueError("only declared two/three consecutive block variants are supported")
    count, last = len(indices), indices[-1]
    if count == 2:
        context, branches, observation = CONTEXT, BRANCHES, OBSERVATION_PLAN
    else:
        context = {**CONTEXT, "id": "p1-s83", "seed": 83}
        branches = BRANCHES[:3] + [{"id": label + ".pred" + str(depth), "base": role, "mode": "pred" + str(depth)}
            for label, role in (("low", "zero"), ("high", "text")) for depth in range(1, count + 1)]
        observation = {**OBSERVATION_PLAN, "boundaries_per_branch": 14,
            "full_candidate_cuts": "Actual common post8 input and final committed Z4, both typed installation cuts, and every native bridge post7/post8. No complete intermediate common-suffix candidate cut claim."}
    suffix = 21 - last
    panel_steps = 2*count + count*(count-1) + len(branches)*suffix
    panel_acts = 2*(len(branches)-2)
    panel = {"adapter_steps": panel_steps, "acts": panel_acts, "receipt_emissions": panel_steps+panel_acts,
        "candidate_calls": 0, "images": len(branches), "vae_calls": len(branches)}
    counts = {**COUNTS, **panel, "adapter_steps": 108+panel_steps, "acts": 4+panel_acts,
        "receipt_emissions": 112+panel_steps+panel_acts}
    return json.loads(json.dumps({"context": context, "block_indices": indices, "branches": branches,
        "order": [row["id"] for row in branches], "counts": counts, "panel_counts": panel,
        "target_calls": {str(index): 2*(position+1) for position,index in enumerate(indices)},
        "suffix_steps": suffix, "observation_plan": observation}))


def validate_config(config):
    declared = contract(config)
    if (config.get("context") != declared["context"] or config.get("height") != 512 or config.get("width") != 512
            or config.get("steps") != 4 or config.get("block_indices") != declared["block_indices"]
            or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 6}
            or config.get("branches") != declared["branches"] or config.get("branch_execution_order") != declared["order"]
            or config.get("sample_indices") != SAMPLE_INDICES or config.get("observation_plan") != declared["observation_plan"]
            or config.get("query_chunk") != 64 or config.get("projection_chunk") != 256
            or config.get("neuron_chunk") != 256 or config.get("retain_fields") is not False
            or config.get("array_storage") != "bf16-bits" or config.get("archive_integer_ids") != "native-int64"):
        raise ValueError("transition-chain frozen scientific geometry differs")
    for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls"):
        if type(config.get("max_" + key)) is not int or config["max_" + key] != declared["counts"][key]:
            raise ValueError("transition-chain finite census differs: " + key)
    for key, lo, hi in (("cpu_threads", 4, 4), ("max_wall_seconds", 1, 900), ("archive_compression_level", 1, 9)):
        if type(config.get(key)) is not int or not lo <= config[key] <= hi:
            raise ValueError("invalid transition-chain execution budget: " + key)


def bind_inputs(blocks, cuts, *, array, counts, builder_ref, preparation_ref, config, profile):
    import numpy as np
    import torch
    from saturn_pub.values import describe

    from mdb.forward_recording import tensor_descriptor

    def immutable(value):
        if value.dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("transition-chain source dtype is unqualified")
        result = np.array(value.detach().float().cpu().numpy(), copy=True)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite transition-chain operand")
        result.setflags(write=False)
        return result

    declared = contract(config)
    if list(blocks) != declared["block_indices"]:
        raise ValueError("actual block order differs from declared finite chain")
    binding = {"schema": "mdb-flux-transition-chain-binding-v1", "inputs": {}, "blocks": {},
        "counts_before_seal": dict(counts), "target_entries_before_seal": {str(index): 0 for index in declared["block_indices"]},
        "builder": builder_ref, "preparation": preparation_ref, "block_indices": declared["block_indices"],
        "sample_indices": SAMPLE_INDICES, "query_chunk": config["query_chunk"],
        "projection_chunk": config["projection_chunk"], "neuron_chunk": config["neuron_chunk"],
        "scope": "Observed pre6 complete text/image, upstream modulation/RoPE and actual block6/7 parameters. No native current6/7 activation or post6 state supplied to prediction."}
    if len(declared["block_indices"]) == 3:
        binding["scope"] = "Observed pre6 complete text/image, upstream modulation/RoPE and actual block6/7/8 parameters. Later blocks consume own complete predicted predecessors; no native target/intermediate activation supplied to prediction."
    operands = {"blocks": [], "cases": {}, "profile": None}
    for index, block in blocks.items():
        attn = block.attn
        h, m = attn.inner_dim, attn.mlp_hidden_dim
        if (h != attn.heads * attn.head_dim or m < 1 or block.norm.elementwise_affine or block.norm.eps != 1e-6
                or attn.norm_q.eps != 1e-6 or attn.norm_k.eps != 1e-6
                or not attn.norm_q.elementwise_affine or not attn.norm_k.elementwise_affine
                or tuple(attn.to_qkv_mlp_proj.weight.shape) != (3*h+2*m, h)
                or tuple(attn.to_out.weight.shape) != (h, h+m)):
            raise ValueError("transition-chain actual full-block parameter profile differs")
        params = {"qkv_mlp_weight": attn.to_qkv_mlp_proj.weight, "qkv_mlp_bias": attn.to_qkv_mlp_proj.bias,
            "to_out_weight": attn.to_out.weight, "to_out_bias": attn.to_out.bias,
            "norm_q_weight": attn.norm_q.weight, "norm_k_weight": attn.norm_k.weight}
        if any(value.dtype != torch.bfloat16 for value in params.values() if value is not None):
            raise ValueError("full-block parameters must be native BF16")
        binding["blocks"][str(index)] = {"parameters": {name: None if value is None else array(value) for name, value in params.items()},
            "native_parameters": {name: tensor_descriptor(value) for name, value in block.named_parameters()},
            "heads": attn.heads, "head_dim": attn.head_dim, "hidden_width": h, "mlp_width": m,
            "layernorm_eps": block.norm.eps, "qk_eps": attn.norm_q.eps, "attention_profile": profile(block)}
        operands["blocks"].append({"block_index": index, **{name: None if value is None else immutable(value) for name, value in params.items()}})
        common_profile = {"heads": attn.heads, "layernorm_eps": block.norm.eps, "qk_eps": attn.norm_q.eps}
        if operands["profile"] not in (None, common_profile):
            raise ValueError("consecutive block common attention profiles differ")
        operands["profile"] = common_profile
    for role, cut in cuts.items():
        image, text = cut.payload["image"], cut.payload["text"]
        if (image.ndim != 3 or text.ndim != 3 or image.shape[0] != 1 or text.shape[0] != 1
                or image.shape[-1] != h or text.shape[-1] != h
                or any(cut.execution_point[key] != value for key, value in config["selector"].items())):
            raise ValueError("transition-chain pre6 source geometry/clock differs")
        residual = torch.cat((text, image), dim=1)
        mod = cut.payload["context"]["single_mod"]
        cosine, sine = cut.payload["context"]["rope"]
        binding["inputs"][role] = {"cut": cut.fingerprint, "point": dict(cut.execution_point), "text_rows": text.shape[1],
            "residual": array(residual), "single_mod": array(mod), "rope_cos": array(cosine), "rope_sin": array(sine),
            "non_stream_state": describe({key: value for key, value in cut.payload.items() if key not in {"image", "text", "cursor"}}),
            "clock_state": {key: cut.payload[key] for key in ("step", "phase", "cursor")}}
        operands["cases"][role] = {"residual": immutable(residual), "single_mod": immutable(mod),
            "rope_cos": immutable(cosine), "rope_sin": immutable(sine), "text_rows": text.shape[1]}
    return binding, operands


def predict_cases(operands, *, helper, config):
    return {"cases": {role: helper["predict_chain"](case["residual"], case["single_mod"], operands["blocks"],
        **operands["profile"], rope_cos=case["rope_cos"], rope_sin=case["rope_sin"],
        query_chunk=config["query_chunk"], projection_chunk=config["projection_chunk"], neuron_chunk=config["neuron_chunk"],
        text_rows=case["text_rows"], sample_indices=SAMPLE_INDICES, retain_fields=False)
        for role, case in operands["cases"].items()}}


def run_panel(*, adapter, cuts, output, config, store, array, wall, event, counts, predictions):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb.native_navigation import NativeCutNavigator

    util = runpy.run_path("flux_single_mlp_campaign.py")
    legacy = runpy.run_path("flux_single_contraction.py")
    previous = runpy.run_path("flux_attention_replacement_campaign.py")
    save, sha = util["save"], util["sha"]
    declared = contract(config)
    indices, suffix = declared["block_indices"], declared["suffix_steps"]
    last, common_key = indices[-1], "common_post" + str(indices[-1])
    checked_predictions = set()
    source_non_stream = {role: describe({key: value for key, value in cut.payload.items() if key not in {"image", "text", "cursor"}})
        for role, cut in cuts.items()}
    active = {"branch": None}
    emissions, prefixes, common, rows = [], {}, {}, {}
    def receipt(value):
        body = value.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        counts["receipt_emissions"] += 1
        if counts["adapter_steps"] > declared["counts"]["adapter_steps"] or counts["acts"] > declared["counts"]["acts"]:
            raise RuntimeError("transition-chain finite panel census exhausted")
        store.receipt(value)
        item = {"sequence": len(emissions), "branch": active["branch"], "receipt": value.fingerprint}
        emissions.append(item)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    def navigator(cut):
        store.save(cut)
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=20, on_receipt=receipt)
    def boundary(cut):
        return {"cut": cut.fingerprint, "point": dict(cut.execution_point),
            "image": array(cut.payload["image"]), "text": array(cut.payload["text"]),
            "non_stream_state": describe({key: value for key, value in cut.payload.items() if key not in {"image", "text", "cursor"}}),
            "clock_state": {key: cut.payload[key] for key in ("step", "phase", "cursor")}}
    def non_stream_exact(cut, role):
        if describe({key: value for key, value in cut.payload.items() if key not in {"image", "text", "cursor"}}) != source_non_stream[role]:
            raise ValueError("full-block transition changed an undeclared non-I/T slot")
        point = cut.execution_point
        if (point["logical_step"] != 3 or point["phase"] != "single" or point["index"] not in {index+1 for index in indices}
                or point["local_clock"] != point["index"] or cut.payload["cursor"] != point["index"]
                or cut.payload["step"] != 3 or cut.payload["phase"] != "single"):
            raise ValueError("full-block transition clock state differs")
    def identity(value):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or not np.isfinite(value).all()
                or np.any(np.ascontiguousarray(value).view(np.uint32) & np.uint32(65535))):
            raise ValueError("sealed transition output must contain finite exact BF16 widenings")
        value = np.ascontiguousarray(value)
        return {"shape": list(value.shape), "dtype": "float32-exact-bf16-widening",
            "fp32_le_sha256": hashlib.sha256(value.astype("<f4",copy=False).tobytes()).hexdigest(),
            "bf16_le_sha256": hashlib.sha256((value.view(np.uint32)>>16).astype("<u2").tobytes()).hexdigest()}
    def validate_chain(role):
        if role in checked_predictions:
            return
        original = torch.cat((cuts[role].payload["text"],cuts[role].payload["image"]),dim=1).float().cpu().numpy()
        expected = identity(original)
        chain = predictions[role]
        normal = util["json_value"]
        if (chain["chain_length"] != len(indices) or len(chain["steps"]) != len(indices)
                or normal(chain["input_descriptor"]) != expected):
            raise ValueError("sealed chain source identity or length differs")
        text_rows = cuts[role].payload["text"].shape[1]
        for position,step in enumerate(chain["steps"]):
            prediction = step["prediction"]
            value = prediction["output"]
            actual = identity(value)
            if (value.shape != original.shape or step["step"] != position or step["block_index"] != indices[position]
                    or step["input_source"] != ("caller pre-entry residual" if position == 0 else "previous predicted full output")
                    or normal(step["input_descriptor"]) != expected or normal(prediction["input_descriptor"]) != expected
                    or normal(prediction["output_descriptor"]) != actual
                    or not np.array_equal(prediction["text"].view(np.uint32),value[:,:text_rows].view(np.uint32))
                    or not np.array_equal(prediction["image"].view(np.uint32),value[:,text_rows:].view(np.uint32))):
                raise ValueError("sealed chain own-state input/output identity or both-stream join differs")
            expected = actual
        if normal(chain["output_descriptor"]) != expected or identity(chain["output"]) != expected:
            raise ValueError("sealed chain final output identity differs")
        checked_predictions.add(role)
    def predicted_tensor(role, stage, slot, reference):
        validate_chain(role)
        full = predictions[role]["steps"][stage]["prediction"]["output"]
        text_rows = cuts[role].payload["text"].shape[1]
        image_rows = cuts[role].payload["image"].shape[1]
        width = cuts[role].payload["image"].shape[-1]
        if not isinstance(full, np.ndarray) or full.dtype != np.float32 or full.shape != (1, text_rows + image_rows, width):
            raise ValueError("sealed transition output geometry differs")
        if not np.isfinite(full).all() or np.any(np.ascontiguousarray(full).view(np.uint32) & np.uint32(0xffff)):
            raise ValueError("sealed transition output must contain finite exact BF16 widenings")
        selected = full[:, :text_rows] if slot == "text" else full[:, text_rows:]
        payload = torch.from_numpy(np.array(selected, copy=True)).to(device=reference.device, dtype=reference.dtype)
        if not np.array_equal(payload.float().cpu().numpy().view(np.uint32), np.ascontiguousarray(selected).view(np.uint32)):
            raise ValueError("installed transition payload differs from sealed values")
        return payload
    observer = previous["ContinuationObserver"](array=array, plan=config["observation_plan"])
    native = {}
    for role in ("zero", "text"):
        active["branch"] = "prefix." + role
        nav = navigator(cuts[role])
        paths, states = [], {}
        for index in indices:
            wall()
            entry = nav.step()
            current = nav.capture()
            store.save(current)
            non_stream_exact(current, role)
            if current.execution_point["index"] != index + 1 or current.execution_point["local_clock"] != index + 1:
                raise ValueError("native block-exit clock differs")
            paths.append(entry.to_dict())
            states[str(index)] = current
        native[role] = states
        prefixes[role] = {"input_cut": cuts[role].fingerprint,
            **{"post"+str(index): boundary(states[str(index)]) for index in indices}, "receipts": paths}
    save(output / "native-boundaries.json", {"prefixes": prefixes, "scope": "All native current-target boundaries observed only after the complete free-chain seal."})
    for spec in config["branches"]:
        name, role, mode = spec["id"], spec["base"], spec["mode"]
        active["branch"] = name
        install_index = indices[int(mode[4:])-1] if mode.startswith("pred") else last
        parent = native[role][str(install_index)]
        nav = navigator(parent)
        row = {**spec, "parent": parent.fingerprint, "start_point": nav.point, "acts": [], "bridge_receipts": [],
            "installation_boundary": install_index, "continuation_policy": "recipient-stored-C-no-further-project"}
        if mode != "native":
            for slot in ("image", "text"):
                reference = nav.capture().payload[slot]
                value = reference if mode == "sham" else predicted_tensor(role, install_index - 6, slot, reference)
                row["acts"].append(legacy["install_port"](nav, slot, value, name=name + "." + slot, store=store, single_index=install_index))
            installed = nav.capture()
            non_stream_exact(installed, role)
            row["installed"] = boundary(installed)
            if len(indices) == 3:
                row["bridge_boundaries"] = []
            for bridge_index in indices:
                if bridge_index <= install_index:
                    continue
                wall()
                entry = nav.step()
                row["bridge_receipts"].append(entry.to_dict())
                bridged = nav.capture()
                non_stream_exact(bridged, role)
                if bridged.execution_point["index"] != bridge_index+1:
                    raise ValueError("native bridge exited at a different block cursor")
                if len(indices) == 3:
                    store.save(bridged)
                    row["bridge_boundaries"].append(boundary(bridged))
        current = nav.capture()
        if any(current.execution_point[key] != value for key, value in {"logical_step": 3, "phase": "single", "index": last+1, "local_clock": last+1}.items()):
            raise ValueError("all chain rows must meet at actual common post" + str(last) + " boundary")
        non_stream_exact(current, role)
        store.save(current)
        row[common_key] = common[name] = boundary(current)
        initial_receipt = (row["bridge_receipts"] or row["acts"] or prefixes[role]["receipts"])[-1]
        initial_receipt = initial_receipt.get("receipt_id", initial_receipt.get("fingerprint"))
        observation = observer.observe(name, current, index=0, incoming_receipt=initial_receipt)
        observation["complete_cut_retained"] = True
        path = []
        while nav.point["logical_step"] < 4:
            wall()
            entry = nav.step()
            path.append(entry.to_dict())
            observation = observer.observe(name, nav.capture(), index=len(path), incoming_receipt=entry.fingerprint)
            observation["complete_cut_retained"] = len(path) == suffix
        if len(path) != suffix:
            raise ValueError("transition-chain common native suffix census differs")
        final = nav.capture()
        store.save(final)
        if describe(final.payload["conditioning"]) != describe(cuts[role].payload["conditioning"]):
            raise ValueError("chain future changed stored conditioning")
        counts["vae_calls"] += 1
        image = adapter.decode(nav.session, height=512, width=512)
        image_file = output / (name + ".png")
        image.save(image_file)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint, final_latent=array(final.payload["latent"]),
            continuation_receipts=path, image={"path": image_file.name, "sha256": sha(image_file)})
        save(output / "rows" / (name + ".json"), row)
        rows[name] = row
        event("branch_complete", row=name)
    if set(observer.rows) != set(declared["order"]) or any(len(value) != suffix+1 for value in observer.rows.values()):
        raise ValueError("chain observation boundary census differs")
    save(output / "continuation-observations.json", {"schema": "mdb-flux-chain-continuation-observations-v1",
        "plan": config["observation_plan"], "branch_execution_order": declared["order"], "rows": observer.rows,
        "reference_rows": config["observation_plan"]["reference_rows"], "model_calls_added": 0,
        "scope": ("Actual common post7 plus14 exits. Full reference arrays and full candidate endpoints retained; intermediate full candidate norms are worker observations with64 fixed witnesses. Inherited observer capture logic reused, final-cut flags explicitly adapted to15 boundaries." if len(indices) == 2 else
            "Actual common post8 plus13 exits. Full native reference arrays and candidate endpoints retained; intermediate full candidate norms are worker observations with64 fixed witnesses. All native bridges before common post8 retain complete I/T arrays and full cuts; common-suffix final flags adapted to14 boundaries.")})
    save(output / "common-boundaries.json", {"rows": common})
    controls = {"sham_png_exact_low": rows["low.sham"]["image"]["sha256"] == rows["low.native"]["image"]["sha256"],
        "sham_final_latent_exact_low": rows["low.sham"]["final_latent"] == rows["low.native"]["final_latent"]}
    return {"rows": [rows[name] for name in declared["order"]], "counts": dict(counts), "control_observations": controls,
        "native_boundaries_sha256": sha(output / "native-boundaries.json"),
        "common_boundaries_sha256": sha(output / "common-boundaries.json"),
        "continuation_observations_sha256": sha(output / "continuation-observations.json")}


def worker():
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    print("MDB_FLUX_TRANSITION_CHAIN_BOOTSTRAP " + json.dumps({"phase": "worker_enter", "unix_time": time.time()}), flush=True)
    util = runpy.run_path("flux_single_mlp_campaign.py")
    save, sha, identity, json_value = (util[key] for key in ("save", "sha", "identity", "json_value"))
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    declared = contract(config)
    context = declared["context"]
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("transition-chain request/source bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-transition-chain-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"plan_sha256": sha(output / "plan.json"), "request_sha256": request["request_sha256"],
        "source_identity": identity(request["source_hashes"]), "job_id": request["job_id"],
        "work": {key: 0 for key in COUNTS if key != "physical_batch"}, "scope": config["scope"]})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    imports_started = time.monotonic()
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.tensor_archive import write_tensor_archive

    engine = runpy.run_path("flux_single_contexts.py")
    previous = runpy.run_path("flux_attention_replacement_campaign.py")
    predictor = runpy.run_path("flux_transition_reference.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("transition-chain requires admitted CUDA")
    save(output / "startup.json", {"runtime_import_seconds": time.monotonic() - imports_started,
        "worker_preimport_seconds": imports_started - started, "scope": "Environment resolution is outside worker clock"})
    counts = {key: 0 for key in COUNTS}
    counts["physical_batch"] = 1
    progress = {"phase": "inventory", "last_heartbeat": time.monotonic()}
    sequence = 0
    def wall():
        now = time.monotonic()
        if now - started > config["max_wall_seconds"]:
            raise RuntimeError("finite transition-chain wall budget exhausted")
        if now - progress["last_heartbeat"] >= 15:
            progress["last_heartbeat"] = now
            event("heartbeat", phase=progress["phase"])
    def event(kind, **fields):
        nonlocal sequence
        body = json_value({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_TRANSITION_CHAIN_PHASE " + json.dumps(body, sort_keys=True), flush=True)
    arrays = {}
    def array(value):
        wall()
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in arrays:
            arrays[key] = write_tensor_archive(output / "arrays" / (key + ".npy"), value, root=output)
        return arrays[key]
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("transition-chain checkpoint inventory differs")
    progress["phase"] = "model_loading"
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires deterministic five-joint/twenty-single distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
        (type(pipe), type(pipe.text_encoder), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[6]),
         type(pipe.transformer.single_transformer_blocks[6].attn.processor), type(pipe.vae), type(pipe.scheduler), Flux2KleinAdapter)}
    progress["phase"] = "prompt_encoding"
    event("encoder_placement")
    pipe.text_encoder.to("cuda")
    with torch.inference_mode():
        conditioning, txt_ids = pipe.encode_prompt(context["prompt"], device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
    counts["prompt_encodings"] += 1
    encodings = {"conditioning": array(conditioning), "txt_ids": array(txt_ids)}
    conditioning, txt_ids = conditioning.cpu(), txt_ids.cpu()
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    event("encoder_released")
    progress["phase"] = "denoiser_placement"
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    source = {"schema": "mdb-flux-transition-chain-source-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory),
        "checkpoint_files": inventory, "execution": dict(adapter.execution), "implementation_sources": implementations,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "context": context, "encoding": encodings, "encodings": encodings, "counts_after_encoding": dict(counts),
        "cpu_thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "blas_configuration": np.__config__.show(mode="dicts"), "scope": config["scope"]}
    progress["phase"] = "noise_schedule"
    with torch.inference_mode():
        conditioning, txt_ids = conditioning.to("cuda"), txt_ids.to("cuda")
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512, conditioning.dtype,
            torch.device("cuda"), torch.Generator(device="cuda").manual_seed(context["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("transition-chain native schedule differs")
        session = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
            timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
    progress["phase"] = "source_builder"
    event("source_builder_start")
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=output / "builder", context=context,
        job_id=request["job_id"], wall=wall)
    builder_ref = {"path": "builder/builder.json", "sha256": sha(output / "builder/builder.json")}
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key]
    event("source_builder_complete")
    progress["phase"] = "source_preparation"
    cuts, preparation = util["prepare_sources"](adapter=adapter, cuts=cuts, output=output / "preparation", builder_ref=builder_ref, wall=wall)
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key] + preparation["counts"][key]
    preparation_ref = {"path": "preparation/preparation.json", "sha256": sha(output / "preparation/preparation.json")}
    source.update(builder=builder_ref, preparation=preparation_ref)
    save(output / "source.json", source)
    event("sources_prepared")
    blocks = {index: pipe.transformer.single_transformer_blocks[index] for index in declared["block_indices"]}
    binding, operands = bind_inputs(blocks, cuts, array=array, counts=counts, builder_ref=builder_ref,
        preparation_ref=preparation_ref, config=config, profile=previous["attention_profile"])
    if (operands["profile"]["heads"] != 24
            or any(value["hidden_width"] != 3072 or value["mlp_width"] != 9216 or value["head_dim"] != 128 for value in binding["blocks"].values())
            or any(case["residual"].shape != (1, 1536, 3072) or case["text_rows"] != 512 for case in operands["cases"].values())):
        raise ValueError("qualified production complete-chain dimensions differ")
    save(output / "binding.json", binding)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(binding),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": binding},
        execution_scope=config["scope"], budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000, "max_bytes": 256 << 20})
    release = {"native": False, "blocked": 0, "targets": {str(index): 0 for index in declared["block_indices"]}}
    def guard(module, args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("native model entered before complete transition-chain seal")
        for index, block in blocks.items():
            if module is block:
                release["targets"][str(index)] += 1
    handles = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    try:
        with tx:
            tx.frontier("source_binding", binding)
            progress["phase"] = "transition_prediction"
            event("prediction_start")
            tx.predict("transition_chain", lambda _: {"predictions": predict_cases(operands, helper=predictor, config=config)})
            tx.predict("chain_hypotheses", lambda _: {"predictions": config["hypotheses"]})
            seal = tx.seal()
            event("transition_chain_sealed", seal=seal, target_entries_before_seal=release["targets"], blocked_entries=release["blocked"])
            release["native"] = True
            progress["phase"] = "native_panel"
            tx.execute("native", lambda context: run_panel(adapter=adapter, cuts=cuts, output=output, config=config,
                store=LocalStore(output / "typed-store"), array=array, wall=wall, event=event, counts=counts,
                predictions=context.predictions["transition_chain"]["cases"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output / "transaction-report.json", tx.report())
    if counts != declared["counts"] or release["targets"] != declared["target_calls"] or release["blocked"] != 0:
        raise ValueError("transition-chain completed physical census differs")
    summary = {"schema": "mdb-flux-transition-chain-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "counts": dict(counts), "builder_counts": {"adapter_steps": 108, "acts": 4, "receipt_emissions": 112},
        "panel_counts": declared["panel_counts"],
        "seal": seal, "transaction_verification": verify_transaction(tx.root), "model_identity": adapter.model_identity,
        "control_observations": tx.observation("native")["control_observations"], "target_calls": release["targets"],
        "worker_seconds": time.monotonic() - started, "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    raw_output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if raw_output_bytes + (1 << 20) > config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("transition-chain pre-export output exceeds expanded custody limit")
    print("MDB_FLUX_TRANSITION_CHAIN_ARCHIVE_START " + json.dumps({"raw_output_bytes": raw_output_bytes, "counts": counts}), flush=True)
    util["emit"](output, summary, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        result = Path("flux-transition-chain-result")
        if result.is_dir():
            utilities = runpy.run_path("flux_single_mlp_campaign.py")
            req = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-transition-chain-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": req["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            utilities["save"](result / "failure.json", failure)
            utilities["emit"](result, failure, req["config"])
        raise
