"""Prospective direct-E18 removal with genuine project-entry conditioning Acts.

Each lane forms its own Qwen E and initial noise, then forks sham/zero from one
typed parent. This removes the direct h18 route; h27 still carries its history.
The functional composition uses shared Torch primitives, not independent kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

PRIOR = "flux_raw_to_image.py"
PRIOR_SHA256 = "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a"
FINALIZER_SHA256 = "b9efb61874005c8b11307c197a2ea5299078b42d5b4ab4b5cc719f7a4a4a32bb"
PROMPT = ("A cobalt blue cube to the right of a matte orange ring on a gray tabletop. "
          "Two separate objects, simple studio product photograph.")
CASES = ("sham", "e18-zero")
EXECUTION_ID = "mdb-flux-e18-conditioning-act-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prior():
    path = Path(__file__).with_name(PRIOR)
    require(sha(path) == PRIOR_SHA256, "frozen raw composition differs")
    return runpy.run_path(str(path))


def contract(config=None):
    return dict(schema="mdb-flux-e18-conditioning-count-contract-v1", execution_id=EXECUTION_ID,
                branches=list(CASES), seed=277, batch_size=1, conditioning_columns=[2560, 5120],
                qwen_functional_calls=1, qwen_native_calls=1, qwen_forecast_layers=27,
                qwen_native_layers=36, functional_transitions=224, native_transitions=224,
                act_receipts=4, receipt_occurrences=228, context_operand_occurrences=8,
                functional_decode_calls=2, native_vae_calls=4, noise_draws=2, pngs=6,
                component_hydrations=3, cuda_placements=6, cpu_offloads=6,
                max_gpu_components=1, forecast_module_entries=0, persistent_weight_banks=0)


def make_config(base):
    config = {key: value for key, value in base.items() if key not in
              {"worker_sha256", "request_sha256", "job_id", "_request_record", "output",
               "cases", "case_order", "expected_mask_regimes", "response_recipe"}}
    config.update(schema="mdb-flux-e18-conditioning-plan-v1", prompt=PROMPT, seed=277,
                  branches=list(CASES), conditioning_columns=[2560, 5120],
                  intervention={"address": "conditioning", "mode": "replace", "value": "positive-BF16-zero",
                                "rows": "all512", "site": "step0/project", "h9_h27": "unchanged"},
                  output_name="e18-conditioning-result", max_wall_seconds=1200,
                  forecast_transaction_bytes=1 << 30,
                  transport_limits={"max_expanded_bytes": 8 << 30, "max_member_bytes": 134221824},
                  signed_response={"orientation": "e18-zero-minus-sham", "arithmetic": "float64(edit)-float64(sham)",
                                   "hash_encoding": "C-order little-endian float64", "duplicate_delta_bank": False})
    config["helper_source_sha256"] = {**config.get("helper_source_sha256", {}), PRIOR: PRIOR_SHA256,
                                     "flux_image_tail_microscope.py": FINALIZER_SHA256}
    return config


def validate_config(config):
    fixed = make_config({})
    for key in ("schema", "prompt", "seed", "branches", "conditioning_columns", "intervention",
                "transport_limits", "signed_response"):
        require(config.get(key) == fixed[key], "fixed E18 configuration differs: " + key)
    old_config = dict(config, schema="mdb-flux-raw-to-image-plan-v1")
    old_config.pop("worker_sha256", None)
    prior()["validate_config"](old_config)
    require(config["attention_dispatch"]["backend"] == "sdpa", "SDPA profile required")
    require(config["attention_dispatch"]["gqa_policy"] == "repeat_kv", "short padded profile requires repeated KV")
    require(0 < config["forecast_transaction_bytes"] <= 1 << 30, "transaction bound differs")
    if config.get("worker_sha256"):
        require(sha(__file__) == config["worker_sha256"], "worker source differs")
    return contract(config)


def cut_clock(cut):
    """The pinned Flux2 adapter exposes these exact clock fields publicly."""
    point = cut.execution_point
    return {"step": point["logical_step"], "phase": point["phase"], "cursor": point["local_clock"]}


def cut_record(cut, store):
    return {"cut": cut.fingerprint, "parent": cut.parent, "model_identity": cut.model_identity, "boundary": cut.boundary,
            "execution_point": dict(cut.execution_point or {}), "store": store,
            "clock": cut_clock(cut)}


def write_receipt(output, counters, lane, case, kind, receipt):
    row = {"occurrence": counters["receipt_occurrences"], "lane": lane, "branch": case,
           "kind": kind, "receipt": receipt.to_dict()}
    with (Path(output) / "receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    counters["receipt_occurrences"] += 1
    counters["act_receipts"] += kind == "act"


def prepare_branches(adapter, frontend, conditioning, *, columns, lane, output, on_receipt=None):
    """Four production calls total to the real public patch operation, not a shim."""
    import torch
    from saturn_pub.core import Act
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe, digest

    require(lane in ("forecast", "native"), "unknown lane")
    require(conditioning.dtype == torch.bfloat16, "conditioning Act requires BF16 E")
    lo, hi = columns
    require(0 <= lo < hi <= conditioning.shape[-1], "conditioning slice outside E")
    session = adapter.session(latent=frontend["packed_noise"], conditioning=conditioning,
                              img_ids=frontend["img_ids"], txt_ids=frontend["txt_ids"],
                              timesteps=frontend["timesteps"], sigmas=frontend["sigmas"])
    common = session.capture(retain=False)
    require(cut_clock(common) == {"step": 0, "phase": "project", "cursor": 0},
            "Act parent is not initial project")
    store_name = "acts-store"
    store = LocalStore(Path(output) / store_name)
    store.save(common)
    before_source = describe(common.payload)
    sessions, cuts, records = {}, {}, {}
    for case in CASES:
        branch = session.fork(common)
        before = branch.capture(retain=False)
        store.save(before)
        own = before.payload["conditioning"]
        mask = torch.zeros_like(own, dtype=torch.bool)
        mask[..., lo:hi] = True
        act = Act.patch("conditioning", own if case == "sham" else 0, mask,
                        mode="replace", expected_digest=digest(describe(own)))
        receipt = branch.apply(act)
        after = branch.capture(retain=False)
        store.save(after)
        body = receipt.to_dict()
        require(body["parent"] == before.fingerprint and body["result"] == after.fingerprint,
                "actual Act receipt/cut join differs")
        original, changed = before.payload, after.payload
        require(describe({key: value for key, value in original.items() if key != "conditioning"}) ==
                describe({key: value for key, value in changed.items() if key != "conditioning"}),
                "conditioning Act changed another slot")
        a, b = original["conditioning"], changed["conditioning"]
        require(torch.equal(a.view(torch.int16)[~mask], b.view(torch.int16)[~mask]), "Act changed outside mask")
        if case == "e18-zero":
            require(bool((b.view(torch.int16)[mask] == 0).all()), "replacement is not positive BF16 zero")
        else:
            require(describe(original) == describe(changed), "same-value sham changed state")
        record = {"lane": lane, "branch": case, "common_parent": cut_record(common, store_name),
                  "before": cut_record(before, store_name), "after": cut_record(after, store_name),
                  "receipt": body, "operation_id": act.operation_id, "columns": [lo, hi],
                  "exact_shape_mask": list(mask.shape), "selected_values": int(mask.sum().item()),
                  "changed_values": int((a.view(torch.int16) != b.view(torch.int16)).sum().item()),
                  "conditioning_before": describe(a), "conditioning_after": describe(b),
                  "all_other_slots_unchanged": True}
        if on_receipt:
            on_receipt(lane, case, "act", receipt)
        sessions[case], cuts[case], records[case] = branch, after, record
    require(describe(common.payload) == before_source, "common Act parent mutated")
    return sessions, cuts, records


@contextmanager
def context_operand_recorder(model, cuts, active, rows):
    """Join the actual native linear operand, not only a copied state label."""
    from saturn_pub.values import describe

    def consume(_module, args):
        require(active.get("phase") == "project", "context projection outside declared project")
        actual = describe(args[0])
        expected = describe(cuts[active["branch"]].payload["conditioning"])
        require(actual == expected, "context_embedder did not consume branch acted conditioning")
        rows.append({"branch": active["branch"], "logical_step": active["step"],
                     "operand": actual, "source_act_cut": cuts[active["branch"]].fingerprint,
                     "actual_forward_pre_hook": True, "full_operand_exact": True})

    handle = model.context_embedder.register_forward_pre_hook(consume)
    try:
        yield
    finally:
        handle.remove()


def forecast_trajectory(output, model, adapter, initial, graph, *, max_transitions):
    """Own-state graph after an actual Act; all exits are component-derived cuts."""
    from saturn_pub.store import LocalStore

    old = prior()
    root = Path(output)
    store = LocalStore(root / "forecast-state-store")
    rows = [old["_save_cut"](root, store, adapter, initial.payload, lane="forecast",
                             number=-1, parent=initial.fingerprint)]

    def boundary(event, state):
        row = old["_save_cut"](root, store, adapter, state, lane="forecast", number=len(rows) - 1,
                               parent=rows[-1]["cut"])
        row["event"] = dict(event)
        rows.append(row)

    final = graph["functional_trajectory"](model, initial.payload, on_boundary=boundary,
                                           max_transitions=max_transitions)
    require(len(rows) == max_transitions + 1, "functional transition census differs")
    save(root / "forecast-state-rows.json", rows)
    return final, rows


def native_trajectory(output, session, *, lane, case, max_transitions, on_receipt=None, active=None):
    """Each recorded exit is the actual result of one public Session continuation."""
    import torch
    from saturn_pub.store import LocalStore

    from mdb_lanes.native_navigation import NativeCutNavigator

    store = LocalStore(Path(output) / "native-state-store")
    nav = NativeCutNavigator(session, max_steps=max_transitions)
    rows = []
    initial = nav.session.capture(retain=False)
    store.save(initial)
    row = cut_record(initial, "native-state-store")
    row["number"] = -1
    rows.append(row)
    for number in range(max_transitions):
        before = nav.session.capture(retain=False)
        require(before.fingerprint == rows[-1]["cut"], "native before cut differs from saved previous after")
        clock = cut_clock(before)
        if active is not None:
            active.update(branch=case, step=clock["step"], phase=clock["phase"])
        with torch.inference_mode():
            receipt = nav.step()
        after = nav.session.capture(retain=False)
        store.save(after)
        body = receipt.to_dict()
        require(body["parent"] == before.fingerprint and body["result"] == after.fingerprint,
                "native receipt/cut join differs")
        if on_receipt:
            on_receipt(lane, case, "native", receipt)
        row = cut_record(after, "native-state-store")
        row.update(number=number, receipt=body["fingerprint"], receipt_parent=body["parent"],
                   event={"step": clock["step"], "phase": clock["phase"], "index": clock["cursor"]})
        rows.append(row)
    save(Path(output) / "native-state-rows.json", rows)
    return after.payload, rows


def signed_hash(left, right):
    """FP64 edit-minus-sham content identity; no duplicate response tensor bank."""
    import numpy as np

    require(tuple(left.shape) == tuple(right.shape), "signed response shape differs")
    digest = hashlib.sha256()
    a, b = left.detach().cpu().reshape(-1), right.detach().cpu().reshape(-1)
    for start in range(0, a.numel(), 1 << 18):
        value = b[start:start + (1 << 18)].double() - a[start:start + (1 << 18)].double()
        digest.update(np.asarray(value.numpy(), dtype="<f8").tobytes())
    return {"shape": list(left.shape), "elements": left.numel(), "dtype": "float64",
            "orientation": "e18-zero-minus-sham", "sha256": digest.hexdigest()}


def tensor_leaves(value, name=""):
    if hasattr(value, "detach"):
        yield name, value
    elif hasattr(value, "items"):
        for key, child in value.items():
            yield from tensor_leaves(child, name + "." + str(key) if name else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, name + "." + str(index))


def bind_qwen_input(graph, ids, mask, tokenizer_files):
    return graph["bind_prospective_tokenized_input"](ids, mask, tokenizer_binding={
        "cohort": "prospective-new", "parser_stage": "tokenizer", "implementation_pinned": True,
        "template_files": tokenizer_files})


def uncropped_input_count(tokenizer):
    record = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, padding=False, truncation=False, return_dict=True)
    return len(record["input_ids"])


def lane_phases(lane, place, qwen_phase, flux_phase, vae_phase):
    """One device placement per component per lane; offload also on failure."""
    conditioning = None
    for phase in ("qwen", "flux", "vae"):
        place(phase, True)
        try:
            if phase == "qwen":
                conditioning = qwen_phase(lane)
            elif phase == "flux":
                flux_phase(lane, conditioning)
            else:
                vae_phase(lane)
        finally:
            place(phase, False)


def seal_responses(output):
    import numpy as np
    import torch
    from saturn_pub.store import LocalStore

    roots = {case: Path(output) / "cases" / case for case in CASES}
    stores = {case: LocalStore(root / "forecast-state-store") for case, root in roots.items()}
    rows = {case: read(root / "forecast-state-rows.json") for case, root in roots.items()}
    result = {"schema": "mdb-e18-signed-response-v1", "orientation": "e18-zero-minus-sham", "denoiser": []}
    for left, right in zip(rows["sham"], rows["e18-zero"], strict=True):
        require(left["clock"] == right["clock"], "forecast response clocks differ")
        a = dict(tensor_leaves(stores["sham"].load(left["cut"]).payload))
        b = dict(tensor_leaves(stores["e18-zero"].load(right["cut"]).payload))
        require(a.keys() == b.keys(), "response field liveness differs")
        result["denoiser"].append({"number": left["number"], "clock": left["clock"],
                                    "fields": {name: signed_hash(a[name], b[name]) for name in a}})
        del a, b

    def load(root, record):
        require(sha(root / record["path"]) == record["sha256"], "forecast decoder source differs")
        value = np.load(root / record["path"], allow_pickle=False)
        return torch.from_numpy(value).view(torch.bfloat16) if value.dtype == np.uint16 else torch.from_numpy(value)

    dec = {case: read(root / "decoder-forecast.json") for case, root in roots.items()}
    result["decoder"] = {name: signed_hash(load(roots["sham"], dec["sham"][name]), load(roots["e18-zero"], dec["e18-zero"][name]))
                          for name in ("decoded", "pixels")}
    save(Path(output) / "forecast-responses.json", result)
    return result


def run(config, *, progress=None):
    declared, old = validate_config(config), prior()
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.values import describe
    from transformers import AutoTokenizer

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output = Path(config["output"]).resolve()
    require(all(not output.is_relative_to(Path(config[key]).resolve().parent) for key in old["_source_keys"]()), "output overlaps immutable sources")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    if config.get("_request_record"):
        save(output / "request.json", config["_request_record"])
    started = time.monotonic()
    counters = {key: 0 for key in ("component_hydrations", "cuda_placements", "cpu_offloads", "noise_draws",
                "functional_transitions", "native_transitions", "functional_decode_calls", "native_vae_calls",
                "native_qwen_calls", "native_module_entries", "guard_refusals", "act_receipts", "receipt_occurrences")}
    released = {"native": False}
    placements, events, acts, context_consumers = [], [], {}, []
    data = {case: {"root": output / "cases" / case} for case in CASES}
    for item in data.values():
        item["root"].mkdir(parents=True)

    def event(kind, **details):
        row = {"kind": kind, "seconds": time.monotonic() - started, **details}
        require(row["seconds"] <= config["max_wall_seconds"], "finite E18 wall budget exceeded")
        events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress:
            progress(row)

    def receipt(lane, case, kind, value):
        write_receipt(output, counters, lane, case, kind, value)

    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(config.get("source_reduction_policy", True))
    require(torch.cuda.is_available(), "production E18 requires CUDA")
    modules = old["_helper_modules"]({key: value for key, value in config.items() if key != "worker_sha256"})
    qw, qg = modules[old["QWEN_WORKER"]], modules[old["QWEN_HELPER"]]
    dg, vg = modules[old["DENOISER_HELPER"]], modules[old["DECODER_HELPER"]]
    rope = config.get("rope_buffer_policy", config.get("qwen_rope_buffer_policy"))
    config = dict(config, rope_buffer_policy=rope)
    runtime = qw["_runtime_contract"](config, torch)
    runtime["native_diffusers_sources"] = old["_native_source_contract"](config)["sources"]
    save(output / "runtime-contract.json", runtime)
    sources = {}
    for key in old["_source_keys"]():
        source = Path(config[key])
        target = output / "source" / key / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[key] = {"path": str(target.relative_to(output)), "sha256": sha(target), "bytes": target.stat().st_size}
    binding = read(config["qwen_source_binding"])
    tokenizer_files = qw["_copy_tokenizer_sources"](output, binding)
    tokenizer = AutoTokenizer.from_pretrained(binding["checkpoint"]["model_root"] + "/tokenizer", local_files_only=True, padding_side="right")
    parser_meta = read(output / "source/tokenizer/tokenizer_config.json")
    require(tokenizer.is_fast and type(tokenizer).__name__ == parser_meta["tokenizer_class"], "pinned parser differs")
    formatted, ids, mask = qw["_tokenize"](tokenizer, PROMPT, config)
    save(output / "input.json", {"prompt": PROMPT, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
         "formatted_chat": formatted, "uncropped_token_count": uncropped_input_count(tokenizer),
         "attended_rows": int(mask.sum().item()), "padded_rows": mask.numel() - int(mask.sum().item())})
    qb, qm = qw["_load_recipe_bank"](Path(config["qwen_recipe_helper"]), Path(config["qwen_manifest"]), config["qwen_parameter_root"])
    qwen = qw["_load_qwen"](config, torch.device("cpu"), qb, qm)
    flux, _fb, _fm = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae, _vb, _vm = old["_load_vae"](config, runpy.run_path(config["vae_recipe_helper"]))
    models = {"qwen": qwen, "flux": flux, "vae": vae}
    prefixes = {"qwen": None, "flux": None, "vae": ("post_quant_conv.", "decoder.", "bn.")}
    identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    tied = qwen.lm_head.weight is qwen.model.embed_tokens.weight
    require(tied, "Qwen tied alias missing")
    counters["component_hydrations"] = 3
    pipeline = {"Flux2KleinPipeline": Flux2KleinPipeline}
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1) * 2)
    tx, seal_record, forecast_inventory = None, None, {}

    def place(phase, cuda):
        model = models[phase]
        before = old["_component_signature"](model, prefixes=prefixes[phase])
        device = "cuda" if cuda else "cpu"
        if phase == "vae":
            model.post_quant_conv.to(device)
            model.decoder.to(device)
            require(model.bn.running_mean.device.type == model.bn.running_var.device.type == "cpu", "BN frontend left CPU")
        else:
            model.to(device)
        after = old["_component_signature"](model, prefixes=prefixes[phase])
        old["_assert_signature_equal"](before, after, phase)
        placements.append({"component": phase, "direction": device, "before_signature": before, "after_signature": after})
        counters["cuda_placements" if cuda else "cpu_offloads"] += 1
        event("component_placement", component=phase, direction=device)
        if not cuda:
            torch.cuda.empty_cache()

    def qwen_phase(lane):
        x, m = ids.to("cuda"), mask.to("cuda")
        with torch.inference_mode():
            if lane == "forecast":
                parameters = {"model.embed_tokens.weight": qwen.model.embed_tokens.weight}
                state = qwen.state_dict()
                for index in range(27):
                    parameters.update({f"model.layers.{index}.{suffix}": state[f"model.layers.{index}.{suffix}"] for suffix in qg["QWEN_LAYER_PARAMETER_SUFFIXES"]})
                bound = bind_qwen_input(qg, x, m, tokenizer_files)
                result = qg["functional_qwen_prefix"](bound["input_ids"], bound["attention_mask"], parameters, qwen.config,
                            attention_dispatch=config["attention_dispatch"], rope_buffer_policy=rope, prefix_layers=27, selected_hidden_states=(9, 18, 27))
                states, conditioning = result["hidden_states"], result["conditioning"]
            else:
                event("native_qwen_entry", forecast_seal=seal_record["seal_sha256"])
                result = qwen(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False)
                states = result.hidden_states
                require(len(states) == 37, "full native Qwen state census differs")
                conditioning = torch.cat([states[index] for index in (9, 18, 27)], dim=-1)
                counters["native_qwen_calls"] += 1
        record = {"input_ids": old["_save_tensor"](output, f"qwen/{lane}/ids.npy", x, logical_dtype="torch.int64"),
                  "attention_mask": old["_save_tensor"](output, f"qwen/{lane}/mask.npy", m, logical_dtype="torch.int64"),
                  "hidden_states": [old["_save_tensor"](output, f"qwen/{lane}/hidden-{index:02d}.npy", states[index]) for index in range(28)],
                  "conditioning": old["_save_tensor"](output, f"qwen/{lane}/conditioning.npy", conditioning),
                  "selected_hidden_states": [9, 18, 27], "executed_full_layers": 36 if lane == "native" else None}
        if lane == "forecast":
            record.update(position_ids=old["_save_tensor"](output, "qwen/forecast/positions.npy", result["position_ids"], logical_dtype="torch.int64"), profile=result["profile"])
        else:
            record["last_hidden_state"] = old["_save_tensor"](output, "qwen/native/final-normalized.npy", states[-1])
        save(output / (lane + "-qwen.json"), record)
        event("qwen_completed", lane=lane, conditioning=old["_tensor_signature"](conditioning),
              native_qwen_calls=counters["native_qwen_calls"])
        return conditioning.detach().cpu()

    lane_frontends = {}

    def flux_phase(lane, conditioning):
        dg["validate_model"](flux)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
        adapter = Flux2KleinAdapter(flux, scheduler, granularity="operation", device="cuda")
        conditioning = conditioning.to("cuda")
        require(tuple(conditioning.shape) == (1, 512, 7680), "production own-E geometry differs")
        frontend = old["_frontend"](config, conditioning, "cuda", pipeline)
        counters["noise_draws"] += 1
        lane_frontends[lane] = {key: value.detach().cpu() for key, value in frontend.items() if hasattr(value, "detach")}
        frontend_record = old["_frontend_record"](output, lane, frontend)
        sessions, cuts, records = prepare_branches(adapter, frontend, conditioning, columns=config["conditioning_columns"],
                    lane=lane, output=output, on_receipt=receipt)
        acts[lane] = records
        save(output / (lane + "-acts.json"), records)
        save(output / ("input-lane-" + lane + ".json"), {"frontend": frontend_record, "own_conditioning": describe(conditioning), "branches": records})
        active = {}

        @contextmanager
        def consumer_scope():
            if lane == "native":
                with context_operand_recorder(flux, cuts, active, context_consumers):
                    yield
            else:
                yield

        with consumer_scope():
            for case in CASES:
                root = data[case]["root"]
                event("denoiser_branch_entry", lane=lane, branch=case, input_act_cut=cuts[case].fingerprint)
                if lane == "forecast":
                    state, rows = forecast_trajectory(root, flux, adapter, cuts[case], dg, max_transitions=112)
                else:
                    state, rows = native_trajectory(root, sessions[case], lane=lane, case=case, max_transitions=112,
                                                     on_receipt=receipt, active=active)
                counters["functional_transitions" if lane == "forecast" else "native_transitions"] += len(rows) - 1
                require((state["step"], state["phase"], state["cursor"]) == (4, "project", 0), "branch terminal clock differs")
                require(all(state[key] is None for key in ("image", "text", "context", "noise_prediction")), "terminal working slots not reset")
                data[case][lane + "_state"] = {key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in state.items()}
                event("denoiser_branch_complete", lane=lane, branch=case, transitions=112)

    def vae_phase(lane):
        for case in CASES:
            root = data[case]["root"]
            if lane == "forecast":
                record = old["_functional_decoder"](root, vae, data[case]["forecast_state"], config, vg, pipeline, processor, {})
                counters["functional_decode_calls"] += 1
                save(root / "decoder-forecast.json", record)
            else:
                for origin, state_key, basename in (("own_forecast", "forecast_state", "own"), ("native_reference", "native_state", "reference")):
                    record = old["_native_decoder"](root, vae, data[case][state_key], config, origin, pipeline, processor, {}, {})
                    counters["native_vae_calls"] += 1
                    save(root / ("decoder-native-" + basename + ".json"), record)
            event("decoder_branch_complete", lane=lane, branch=case)

    try:
        with old["_learned_forward_guard"](list(models.values()), released, counters):
            lane_phases("forecast", place, qwen_phase, flux_phase, vae_phase)
            require(counters["native_module_entries"] == 0, "native learned target before global forecast seal")
            responses = seal_responses(output)
            forecast_inventory = old["_inventory"](output, exclude=("timeline.jsonl", "receipt-emissions.jsonl"))
            tx = ForecastTransaction(output / "forecast-transaction", model_identity="flux2-klein-4b-original",
                input_identity=hashlib.sha256(canonical({"prompt": PROMPT, "seed": 277, "intervention": config["intervention"]})).hexdigest(),
                source_identity=hashlib.sha256(canonical({key: config[key + "_sha256"] for key in old["_source_keys"]()})).hexdigest(),
                frontier={"kind": "raw-input", "declared_intervention": "conditioning-Act", "prompt": PROMPT, "seed": 277,
                          "intervention": config["intervention"], "source_inventory": forecast_inventory},
                execution_scope="Two complete own-E functional branches and signed responses sealed before native Qwen; direct E18 removal, h27 unchanged; shared primitives, not independent kernels.",
                budgets={"max_predictions": 1, "max_executions": 1, "max_native_calls": 1,
                         "max_bytes": config["forecast_transaction_bytes"], "max_files": 30000,
                         "max_wall_ms": config["max_wall_seconds"] * 1000})
            tx.predict("e18-complete-composition", lambda _context: {"predictions": {"external_files": forecast_inventory,
                "responses": {"path": "forecast-responses.json", "sha256": sha(output / "forecast-responses.json"), "orientation": responses["orientation"]},
                "acts": acts["forecast"], "native_targets_at_seal": 0}})
            seal_record = tx.seal()
            event("forecast_globally_sealed", seal_sha256=seal_record["seal_sha256"], native_module_entries=0,
                  forecast_act_receipts=counters["act_receipts"], functional_transitions=counters["functional_transitions"])

            def execute(_context):
                old["_verify_inventory"](output, forecast_inventory)
                released["native"] = True
                event("native_targets_released", forecast_seal=seal_record["seal_sha256"])
                lane_phases("native", place, qwen_phase, flux_phase, vae_phase)
                return {"native_transitions": counters["native_transitions"], "native_vae_calls": counters["native_vae_calls"],
                        "native_act_receipts": 2, "context_operand_occurrences": len(context_consumers)}

            tx.execute("native-own-E-branches", execute)
            tx.close()
        for key in lane_frontends["forecast"]:
            require(torch.equal(lane_frontends["forecast"][key], lane_frontends["native"][key]), "independent lane frontend differs: " + key)
        after = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
        for name in models:
            old["_assert_signature_equal"](identities[name], after[name], name)
        require((qwen.lm_head.weight is qwen.model.embed_tokens.weight) == tied, "tied Qwen alias changed")
        old["_verify_inventory"](output, forecast_inventory)
        for key in ("component_hydrations", "cuda_placements", "cpu_offloads", "noise_draws", "functional_transitions",
                    "native_transitions", "functional_decode_calls", "native_vae_calls", "act_receipts", "receipt_occurrences"):
            require(counters[key] == declared[key], "actual physical census differs: " + key)
        require(counters["native_qwen_calls"] == 1, "native Qwen census differs")
        require(len(context_consumers) == 8 and {(row["branch"], row["logical_step"]) for row in context_consumers} ==
                {(case, step) for case in CASES for step in range(4)}, "actual context consumer census differs")
        save(output / "act-records.json", acts)
        save(output / "context-consumers.json", context_consumers)
        save(output / "native-contract.json", {"counts": counters, "placement_records": placements,
             "component_identity_before": identities, "component_identity_after": after,
             "qwen_tied_alias_before": tied, "qwen_tied_alias_after": True,
             "forecast_module_entries": 0, "context_operand_occurrences": 8, "selected_kernel": "unprofiled",
             "act_scope": "4 genuine conditioning replace Acts at real step0/project; not upstream Qwen MLP interventions"})
        save(output / "source.json", {"schema": "mdb-flux-e18-conditioning-source-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
             "request_sha256": config.get("request_sha256"), "worker_sha256": sha(__file__),
             "helper_source_sha256": config["helper_source_sha256"], "source_files": sources, "tokenizer_files": tokenizer_files,
             "runtime_contract_sha256": sha(output / "runtime-contract.json"), "forecast_seal": seal_record})
        summary = {"schema": "mdb-flux-e18-conditioning-summary-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                   "request_sha256": config.get("request_sha256"), "counts": counters, "contract": declared,
                   "forecast_seal": seal_record, "transaction": verify_transaction(output / "forecast-transaction",
                       expected_seal_sha256=seal_record["seal_sha256"], max_bytes=config["forecast_transaction_bytes"]),
                   "context_operand_occurrences": len(context_consumers), "mechanics_status": "complete",
                   "trend_status": "observed-uninterpreted", "terminal_status": "unassessed", "seconds": time.monotonic() - started}
        save(output / "summary.json", summary)
        return summary
    finally:
        if tx is not None:
            tx.close()
        torch.cuda.empty_cache()


def worker(config_path):
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print("MDB_E18_CONDITIONING_BOOTSTRAP", flush=True)
    config = read(config_path)
    if os.environ.get("SATURN_DEBUG_REQUEST"):
        request = read(os.environ["SATURN_DEBUG_REQUEST"])
        body = {key: value for key, value in request.items() if key != "request_sha256"}
        require(hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "request identity differs")
        require(request["config"] == config, "request/config differs")
        for name, expected in request["source_hashes"].items():
            path = Path.cwd() / name
            if not path.is_file():
                path = Path(config.get(name, name))
            require(path.is_file() and sha(path) == expected, "staged source differs: " + name)
        config = dict(config, request_sha256=request["request_sha256"], _request_record=request)
    if "output" not in config:
        config = dict(config, output=str(Path(config["output_scratch"]) / os.environ.get("MRUN_JOB_ID", "local") / config["output_name"]))
    summary = run(config, progress=lambda row: print("MDB_E18_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    finalizer_path = Path(__file__).with_name("flux_image_tail_microscope.py")
    require(sha(finalizer_path) == FINALIZER_SHA256, "frozen finalizer source differs")
    finalizer = runpy.run_path(str(finalizer_path))
    ready = finalizer["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)


if __name__ == "__main__":
    main()
