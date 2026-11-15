"""Own Qwen H18 sign history and two globally sealed complete FLUX futures.

The Qwen edit is an explicit source-qualified component suffix, not a Qwen Act.
Four genuine public FLUX conditioning Acts install each lane's own direct or
coherent E. Shared primitive forecasts precede every native learned target.
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
from functools import lru_cache
from pathlib import Path

E18_SHA256 = "45c512de2dff3252c38a182cf4291b392cf35a159c2bd83b35900c28005cb343"
HISTORY_SHA256 = "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036"
CASES = ("direct", "coherent")
EXECUTION_ID = "mdb-flux-h18-state-history-v1"
PROMPT = ("A cobalt blue cube to the right of a matte orange ring on a gray tabletop. "
          "Two separate objects, simple studio product photograph.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@lru_cache(maxsize=1)
def helpers():
    result = {}
    for name, expected in (("flux_e18_conditioning_act.py", E18_SHA256),
                           ("qwen_h18_state_history.py", HISTORY_SHA256)):
        path = Path(__file__).with_name(name)
        require(sha(path) == expected, "frozen composition source differs: " + name)
        result[name] = runpy.run_path(str(path))
    return result["flux_e18_conditioning_act.py"], result["qwen_h18_state_history.py"]


def contract(config=None):
    # Production dimensions are fixed; qualification helpers accept explicit tiny
    # dimensions without weakening this production admission contract.
    horizon, edit, full, steps, joints, singles = 27, 18, 36, 4, 5, 20
    suffix, branches = horizon - edit, len(CASES)
    forecast_blocks, native_blocks = horizon + suffix, full + suffix
    transitions = branches * steps * (1 + joints + singles + 2)
    return dict(schema="mdb-flux-h18-state-history-count-contract-v1", execution_id=EXECUTION_ID,
        branches=list(CASES), seed=277, batch_size=1, conditioning_columns=[2560, 7680],
        qwen_functional_calls=1, qwen_functional_baseline_blocks=horizon, qwen_functional_suffix_blocks=suffix,
        qwen_functional_blocks=forecast_blocks, qwen_functional_SDPA=forecast_blocks, qwen_functional_linear=7 * forecast_blocks,
        qwen_functional_RMSNorm=4 * forecast_blocks, qwen_functional_rotary_applications=2 * forecast_blocks,
        qwen_native_calls=1, qwen_native_baseline_blocks=full, qwen_native_suffix_blocks=suffix,
        qwen_native_blocks=native_blocks, qwen_native_SDPA=native_blocks, qwen_native_authorizations=1 + suffix,
        qwen_parent_conditioning_concats=2, genuine_qwen_acts=0,
        functional_transitions=transitions, native_transitions=transitions, act_receipts=2 * branches,
        receipt_occurrences=transitions + 2 * branches, context_operand_occurrences=steps * branches,
        functional_decode_calls=branches, native_vae_calls=2 * branches, noise_draws=2, pngs=3 * branches,
        component_hydrations=3, cuda_placements=6, cpu_offloads=6,
        max_gpu_components=1, forecast_module_entries=0, persistent_weight_banks=0,
        extra_learned_oddness_projections=0)


def make_config(base):
    e, _ = helpers()
    config = e["make_config"](base)
    config.update(schema="mdb-flux-h18-state-history-plan-v1", branches=list(CASES),
        conditioning_columns=[2560, 7680], output_name="h18-state-history-result",
        history={"edit_index": 18, "horizon": 27, "selected": [9, 18, 27],
                 "operation": "native BF16 unary negation of all512 rows/all2560 features",
                 "native_route": "full36-plus-LM then own-context layers18..26; use_cache=False",
                 "genuine_qwen_act": False},
        intervention={"address": "conditioning", "mode": "replace", "site": "step0/project",
                      "rows": "all512", "columns": [2560, 7680], "both_branches_same_mask": True,
                      "direct": "cat(own H9,own -H18,own baseline H27)",
                      "coherent": "cat(own H9,own -H18,own regenerated H27minus)",
                      "H9": "unchanged", "direct_H27": "same-value replacement"},
        signed_response={"orientation": "coherent-minus-direct", "arithmetic": "float64(coherent)-float64(direct)",
                         "hash_encoding": "C-order little-endian float64", "duplicate_delta_bank": False})
    config["helper_source_sha256"].update({"flux_e18_conditioning_act.py": E18_SHA256,
                                            "qwen_h18_state_history.py": HISTORY_SHA256})
    return config


def validate_config(config):
    e, _ = helpers()
    fixed = make_config({})
    for key in ("schema", "prompt", "seed", "branches", "conditioning_columns", "history", "intervention",
                "transport_limits", "signed_response"):
        require(config.get(key) == fixed[key], "fixed H18 history configuration differs: " + key)
    old_config = dict(config, schema="mdb-flux-raw-to-image-plan-v1")
    old_config.pop("worker_sha256", None)
    e["prior"]()["validate_config"](old_config)
    require(config["attention_dispatch"]["backend"] == "sdpa" and
            config["attention_dispatch"]["gqa_policy"] == "repeat_kv", "original short padded SDPA profile required")
    require(0 < config["forecast_transaction_bytes"] <= 1 << 30, "transaction bound differs")
    for name, value in (("flux_e18_conditioning_act.py", E18_SHA256), ("qwen_h18_state_history.py", HISTORY_SHA256)):
        require(config["helper_source_sha256"].get(name) == value, "declared source differs: " + name)
    if config.get("worker_sha256"):
        require(sha(__file__) == config["worker_sha256"], "worker source differs")
    return contract(config)


def history_values(result, selected):
    """Build the same-lane original E parent and validate both exact recipes."""
    import torch

    a, edit, last = selected
    common, minus = result["common"], result["minus"]
    base = torch.cat((common[a], common[edit], common[last]), dim=-1)
    direct, coherent = result["direct"], result["coherent"]
    require(torch.equal(minus[edit].contiguous().view(torch.uint8), (-common[edit]).contiguous().view(torch.uint8)),
            "history entry is not exact own BF16 negation")
    width = common[a].shape[-1]
    for key, value, last_part in (("direct", direct, common[last]), ("coherent", coherent, minus[last])):
        require(value.shape == base.shape and value.dtype == base.dtype and value.device == base.device,
                "helper conditioning schema differs: " + key)
        for index, part in enumerate((common[a], minus[edit], last_part)):
            require(torch.equal(value[..., index * width:(index + 1) * width].contiguous().view(torch.uint8),
                                part.contiguous().view(torch.uint8)), "helper conditioning recipe differs: " + key)
    require(torch.equal(direct[..., :2 * width].contiguous().view(torch.uint8),
                        coherent[..., :2 * width].contiguous().view(torch.uint8)), "held E9/negative E18 differ")
    return {"base_conditioning": base, "direct": result["direct"], "coherent": result["coherent"]}


def prepare_history_branches(adapter, frontend, history, *, columns, lane, output, on_receipt=None):
    """Use real same-shaped Boolean replacement Acts for BOTH complete E values."""
    import torch
    from saturn_pub.core import Act
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe, digest

    e, _ = helpers()
    require(lane in ("forecast", "native"), "unknown lane")
    conditioning = history["base_conditioning"]
    lo, hi = columns
    require(conditioning.dtype == torch.bfloat16 and (lo, hi) == (conditioning.shape[-1] // 3, conditioning.shape[-1]),
            "history replacement requires whole E18+E27 columns")
    session = adapter.session(latent=frontend["packed_noise"], conditioning=conditioning,
        img_ids=frontend["img_ids"], txt_ids=frontend["txt_ids"], timesteps=frontend["timesteps"], sigmas=frontend["sigmas"])
    common = session.capture(retain=False)
    require(e["cut_clock"](common) == {"step": 0, "phase": "project", "cursor": 0}, "Act parent is not initial project")
    store_name, before_source = "acts-store", describe(common.payload)
    store = LocalStore(Path(output) / store_name)
    store.save(common)
    sessions, cuts, records = {}, {}, {}
    for case in CASES:
        branch = session.fork(common)
        before = branch.capture(retain=False)
        store.save(before)
        original = before.payload
        own, value = original["conditioning"], history[case]
        require(value.shape == own.shape and value.dtype == own.dtype and value.device == own.device, "replacement E schema differs")
        mask = torch.zeros_like(own, dtype=torch.bool)
        mask[..., lo:hi] = True
        require(torch.equal(own[..., :lo].contiguous().view(torch.uint8), value[..., :lo].contiguous().view(torch.uint8)), "H9 changed")
        act = Act.patch("conditioning", value, mask, mode="replace", expected_digest=digest(describe(own)))
        receipt = branch.apply(act)
        after = branch.capture(retain=False)
        store.save(after)
        body, changed = receipt.to_dict(), after.payload
        require(body["parent"] == before.fingerprint and body["result"] == after.fingerprint, "Act receipt/cut join differs")
        require(describe({k: v for k, v in original.items() if k != "conditioning"}) ==
                describe({k: v for k, v in changed.items() if k != "conditioning"}), "conditioning Act changed another slot")
        require(torch.equal(changed["conditioning"].contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)), "Act did not install own E")
        record = dict(lane=lane, branch=case, common_parent=e["cut_record"](common, store_name),
            before=e["cut_record"](before, store_name), after=e["cut_record"](after, store_name), receipt=body,
            operation_id=act.operation_id, columns=list(columns), exact_shape_mask=list(mask.shape), selected_values=int(mask.sum()),
            changed_values=int((own.view(torch.int16) != changed["conditioning"].view(torch.int16)).sum()),
            conditioning_before=describe(own), conditioning_after=describe(changed["conditioning"]),
            replacement=describe(value), all_other_slots_unchanged=True,
            recipe="own H9 / own negative H18 / own " + ("baseline H27" if case == "direct" else "regenerated H27minus"))
        if on_receipt:
            on_receipt(lane, case, "act", receipt)
        sessions[case], cuts[case], records[case] = branch, after, record
    require(describe(common.payload) == before_source, "common Act parent mutated")
    return sessions, cuts, records


def signed_hash(direct, coherent):
    e, _ = helpers()
    return {**e["signed_hash"](direct, coherent), "orientation": "coherent-minus-direct"}


def save_history(output, lane, result, ports, ids, mask, *, selected):
    """Serialize actual own history and actual first-suffix callback ports."""
    e, h = helpers()
    old = e["prior"]()
    root = Path(output)

    def tensor(name, value):
        return old["_save_tensor"](root, f"qwen-history/{lane}/{name}.npy", value)

    def tree(name, value):
        if hasattr(value, "detach"):
            return {"tensor": tensor(name, value)}
        if isinstance(value, dict):
            return {key: tree(name + "/" + key, child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [tree(name + "/" + str(index), child) for index, child in enumerate(value)]
        require(value is None or isinstance(value, (str, int, float, bool)), "unserializable own Qwen context")
        return value

    values = history_values(result, selected)
    record = dict(schema="mdb-qwen-h18-history-record-v1", lane=lane, genuine_qwen_act=False,
        selected=list(selected), input_ids=tensor("ids", ids), attention_mask=tensor("mask", mask),
        common={str(k): tensor(f"common-{k:02d}", value) for k, value in result["common"].items()},
        minus={str(k): tensor(f"minus-{k:02d}", value) for k, value in result["minus"].items()},
        conditioning={name: tensor(name, value) for name, value in values.items()},
        first_suffix_ports={sign: {name: tensor("ports/" + sign + "/" + name, value)
                                  for name, value in row.items()} for sign, row in ports.items()},
        counts=result["counts"], parent_conditioning_concats=1,
        port_provenance=result.get("port_provenance", "functional source primitive returns; KV pre-repetition; head after transpose"),
        writer_diagnostic=h["writer_diagnostic"](ports["plus"], ports["minus"]),
        history_response={str(k): signed_hash(result["common"][k], result["minus"][k]) for k in result["minus"]},
        conditioning_response=signed_hash(result["direct"], result["coherent"]))
    if lane == "forecast":
        record.update(position_ids=tensor("positions", result["position_ids"]), profile=result["profile"], mask_profile=result["mask_profile"])
    else:
        record.update(source_context=tree("context", result["source_context"]),
            consumed=[dict(index=row["index"], hidden=tensor(f"consumed-{row['index']:02d}", row["hidden"]),
                           context_source="source_context", context=tree(f"consumed-context-{row['index']:02d}", row["kwargs"]))
                      for row in result["consumed"]], block_entries=result["block_entries"], module_entries=result["module_entries"],
            sdpa_calls=result["sdpa_calls"], baseline_logits_shape=result["baseline_logits_shape"],
            baseline_state_scope="common outputs include raw interior/full block states; final model norm/LM executed, not a forecast target")
    save(root / (lane + "-qwen-history.json"), record)
    return values, record


def seal_responses(output):
    import numpy as np
    import torch
    from saturn_pub.store import LocalStore

    e, _ = helpers()
    roots = {case: Path(output) / "cases" / case for case in CASES}
    stores = {case: LocalStore(root / "forecast-state-store") for case, root in roots.items()}
    rows = {case: read(root / "forecast-state-rows.json") for case, root in roots.items()}
    result = dict(schema="mdb-h18-history-signed-response-v1", orientation="coherent-minus-direct", denoiser=[])
    for left, right in zip(rows["direct"], rows["coherent"], strict=True):
        require(left["clock"] == right["clock"], "response clocks differ")
        a = dict(e["tensor_leaves"](stores["direct"].load(left["cut"]).payload))
        b = dict(e["tensor_leaves"](stores["coherent"].load(right["cut"]).payload))
        require(a.keys() == b.keys(), "response field liveness differs")
        result["denoiser"].append(dict(number=left["number"], clock=left["clock"],
                                           fields={name: signed_hash(a[name], b[name]) for name in a}))
        del a, b

    def load(root, row):
        require(sha(root / row["path"]) == row["sha256"], "decoder response source differs")
        value = np.load(root / row["path"], allow_pickle=False)
        return torch.from_numpy(value).view(torch.bfloat16) if value.dtype == np.uint16 else torch.from_numpy(value)

    dec = {case: read(root / "decoder-forecast.json") for case, root in roots.items()}
    result["decoder"] = {name: signed_hash(load(roots["direct"], dec["direct"][name]), load(roots["coherent"], dec["coherent"][name]))
                         for name in ("decoded", "pixels")}
    save(Path(output) / "forecast-responses.json", result)
    return result


def run(config, *, progress=None):
    declared = validate_config(config)
    e, history_helper = helpers()
    old = e["prior"]()
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.values import describe
    from transformers import AutoTokenizer

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    output = Path(config["output"]).resolve()
    require(all(not output.is_relative_to(Path(config[key]).resolve().parent) for key in old["_source_keys"]()), "output overlaps immutable sources")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    save(output / "count-contract.json", declared)
    if config.get("_request_record"):
        save(output / "request.json", config["_request_record"])
    started = time.monotonic()
    counters = {key: 0 for key in ("component_hydrations", "cuda_placements", "cpu_offloads", "noise_draws",
        "functional_transitions", "native_transitions", "functional_decode_calls", "native_vae_calls",
        "native_qwen_calls", "native_module_entries", "guard_refusals", "act_receipts", "receipt_occurrences",
        "qwen_functional_blocks", "qwen_functional_SDPA", "qwen_native_blocks", "qwen_native_SDPA", "qwen_native_authorizations")}
    released, placements, acts, consumers = {"native": False}, [], {}, []
    data = {case: {"root": output / "cases" / case} for case in CASES}
    for value in data.values():
        value["root"].mkdir(parents=True)

    def event(kind, **details):
        row = dict(kind=kind, seconds=time.monotonic() - started, **details)
        require(row["seconds"] <= config["max_wall_seconds"], "finite history wall budget exceeded")
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress:
            progress(row)

    def receipt(lane, case, kind, value):
        e["write_receipt"](output, counters, lane, case, kind, value)

    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(config.get("source_reduction_policy", True))
    require(torch.cuda.is_available(), "production history requires CUDA")
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
        sources[key] = dict(path=str(target.relative_to(output)), sha256=sha(target), bytes=target.stat().st_size)
    binding = read(config["qwen_source_binding"])
    tokenizer_files = qw["_copy_tokenizer_sources"](output, binding)
    tokenizer = AutoTokenizer.from_pretrained(binding["checkpoint"]["model_root"] + "/tokenizer", local_files_only=True, padding_side="right")
    parser_meta = read(output / "source/tokenizer/tokenizer_config.json")
    require(tokenizer.is_fast and type(tokenizer).__name__ == parser_meta["tokenizer_class"], "pinned parser differs")
    formatted, ids, mask = qw["_tokenize"](tokenizer, PROMPT, config)
    save(output / "input.json", dict(prompt=PROMPT, prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        formatted_chat=formatted, uncropped_token_count=e["uncropped_input_count"](tokenizer),
        attended_rows=int(mask.sum()), padded_rows=mask.numel() - int(mask.sum())))
    qb, qm = qw["_load_recipe_bank"](Path(config["qwen_recipe_helper"]), Path(config["qwen_manifest"]), config["qwen_parameter_root"])
    qwen = qw["_load_qwen"](config, torch.device("cpu"), qb, qm)
    flux, _fb, _fm = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae, _vb, _vm = old["_load_vae"](config, runpy.run_path(config["vae_recipe_helper"]))
    models = dict(qwen=qwen, flux=flux, vae=vae)
    prefixes = dict(qwen=None, flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
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
        placements.append(dict(component=phase, direction=device, before_signature=before, after_signature=after))
        counters["cuda_placements" if cuda else "cpu_offloads"] += 1
        event("component_placement", component=phase, direction=device)
        if not cuda:
            torch.cuda.empty_cache()

    def authorize_native_qwen():
        require(released["native"] and seal_record is not None, "native Qwen history before global seal")
        counters["qwen_native_authorizations"] += 1
        event("native_qwen_authorized", occurrence=counters["qwen_native_authorizations"], forecast_seal=seal_record["seal_sha256"])

    def qwen_phase(lane):
        x, m = ids.to("cuda"), mask.to("cuda")
        ports = {}

        def observe(sign, name, value):
            require(name not in ports.setdefault(sign, {}), "duplicate first-suffix port")
            ports[sign][name] = value.detach().clone()

        with torch.inference_mode():
            if lane == "forecast":
                state = qwen.state_dict()
                parameters = {"model.embed_tokens.weight": qwen.model.embed_tokens.weight}
                for index in range(27):
                    parameters.update({f"model.layers.{index}.{suffix}": state[f"model.layers.{index}.{suffix}"] for suffix in qg["QWEN_LAYER_PARAMETER_SUFFIXES"]})
                bound = e["bind_qwen_input"](qg, x, m, tokenizer_files)
                result = history_helper["functional_state_history"](qg, bound["input_ids"], bound["attention_mask"], parameters,
                    qwen.config, attention_dispatch=config["attention_dispatch"], rope_buffer_policy=rope, observe=observe)
                counters["qwen_functional_blocks"] += result["counts"]["blocks"]
                counters["qwen_functional_SDPA"] += result["counts"]["SDPA"]
            else:
                result = history_helper["native_state_history"](qwen, x, m, authorize_native=authorize_native_qwen, observe=observe)
                counters["native_qwen_calls"] += result["counts"]["full_model_calls"]
                counters["qwen_native_blocks"] += result["counts"]["baseline_blocks"] + result["counts"]["suffix_blocks"]
                counters["qwen_native_SDPA"] += result["counts"]["SDPA"]
            values, _record = save_history(output, lane, result, ports, x, m, selected=(9, 18, 27))
        event("qwen_history_completed", lane=lane, counts=result["counts"],
              direct=describe(values["direct"]), coherent=describe(values["coherent"]))
        return {name: value.detach().cpu() for name, value in values.items()}

    lane_frontends = {}

    def flux_phase(lane, values):
        dg["validate_model"](flux)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
        adapter = Flux2KleinAdapter(flux, scheduler, granularity="operation", device="cuda")
        values = {name: value.to("cuda") for name, value in values.items()}
        require(tuple(values["base_conditioning"].shape) == (1, 512, 7680), "production own-E geometry differs")
        frontend = old["_frontend"](config, values["base_conditioning"], "cuda", pipeline)
        counters["noise_draws"] += 1
        lane_frontends[lane] = {key: value.detach().cpu() for key, value in frontend.items() if hasattr(value, "detach")}
        frontend_record = old["_frontend_record"](output, lane, frontend)
        sessions, cuts, records = prepare_history_branches(adapter, frontend, values, columns=config["conditioning_columns"],
            lane=lane, output=output, on_receipt=receipt)
        acts[lane] = records
        save(output / (lane + "-acts.json"), records)
        save(output / ("input-lane-" + lane + ".json"), dict(frontend=frontend_record,
            own_conditioning={name: describe(value) for name, value in values.items()}, branches=records))
        active = {}

        @contextmanager
        def consumer_scope():
            if lane == "native":
                with e["context_operand_recorder"](flux, cuts, active, consumers):
                    yield
            else:
                yield

        with consumer_scope():
            for case in CASES:
                root = data[case]["root"]
                event("denoiser_branch_entry", lane=lane, branch=case, input_act_cut=cuts[case].fingerprint)
                if lane == "forecast":
                    state, rows = e["forecast_trajectory"](root, flux, adapter, cuts[case], dg, max_transitions=112)
                else:
                    state, rows = e["native_trajectory"](root, sessions[case], lane=lane, case=case, max_transitions=112,
                        on_receipt=receipt, active=active)
                counters["functional_transitions" if lane == "forecast" else "native_transitions"] += len(rows) - 1
                require((state["step"], state["phase"], state["cursor"]) == (4, "project", 0), "terminal clock differs")
                require(all(state[key] is None for key in ("image", "text", "context", "noise_prediction")), "terminal slots not reset")
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
                for origin, key, basename in (("own_forecast", "forecast_state", "own"), ("native_reference", "native_state", "reference")):
                    record = old["_native_decoder"](root, vae, data[case][key], config, origin, pipeline, processor, {}, {})
                    counters["native_vae_calls"] += 1
                    save(root / ("decoder-native-" + basename + ".json"), record)
            event("decoder_branch_complete", lane=lane, branch=case)

    try:
        with old["_learned_forward_guard"](list(models.values()), released, counters):
            e["lane_phases"]("forecast", place, qwen_phase, flux_phase, vae_phase)
            require(counters["native_module_entries"] == 0, "native target before global seal")
            responses = seal_responses(output)
            forecast_inventory = old["_inventory"](output, exclude=("timeline.jsonl", "receipt-emissions.jsonl"))
            tx = ForecastTransaction(output / "forecast-transaction", model_identity="flux2-klein-4b-original",
                input_identity=hashlib.sha256(canonical(dict(prompt=PROMPT, seed=277, intervention=config["intervention"]))).hexdigest(),
                source_identity=hashlib.sha256(canonical({key: config[key + "_sha256"] for key in old["_source_keys"]()})).hexdigest(),
                frontier=dict(kind="raw-input", prompt=PROMPT, seed=277, history=config["history"],
                              intervention=config["intervention"], source_inventory=forecast_inventory),
                execution_scope="Own baseline27+edited9 Qwen; two own E/112-step/decoder forecasts and complete signed responses before native; no Qwen Act or independent kernel claim.",
                budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1,
                             max_bytes=config["forecast_transaction_bytes"], max_files=30000,
                             max_wall_ms=config["max_wall_seconds"] * 1000))
            tx.predict("h18-complete-composition", lambda _ctx: {"predictions": dict(external_files=forecast_inventory,
                responses=dict(path="forecast-responses.json", sha256=sha(output / "forecast-responses.json"), orientation=responses["orientation"]),
                acts=acts["forecast"], native_targets_at_seal=0)})
            seal_record = tx.seal()
            event("forecast_globally_sealed", seal_sha256=seal_record["seal_sha256"], native_module_entries=0,
                  forecast_act_receipts=counters["act_receipts"], functional_transitions=counters["functional_transitions"],
                  qwen_functional_blocks=counters["qwen_functional_blocks"])

            def execute(_ctx):
                old["_verify_inventory"](output, forecast_inventory)
                released["native"] = True
                event("native_targets_released", forecast_seal=seal_record["seal_sha256"])
                e["lane_phases"]("native", place, qwen_phase, flux_phase, vae_phase)
                return dict(native_transitions=counters["native_transitions"], native_vae_calls=counters["native_vae_calls"],
                            qwen_native_blocks=counters["qwen_native_blocks"], native_act_receipts=2, context_operand_occurrences=len(consumers))

            tx.execute("native-own-history-branches", execute)
            tx.close()
        for key in lane_frontends["forecast"]:
            require(torch.equal(lane_frontends["forecast"][key], lane_frontends["native"][key]), "independent frontend differs: " + key)
        after = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
        for name in models:
            old["_assert_signature_equal"](identities[name], after[name], name)
        require((qwen.lm_head.weight is qwen.model.embed_tokens.weight) == tied, "tied Qwen alias changed")
        old["_verify_inventory"](output, forecast_inventory)
        for key in ("component_hydrations", "cuda_placements", "cpu_offloads", "noise_draws", "functional_transitions",
                    "native_transitions", "functional_decode_calls", "native_vae_calls", "act_receipts", "receipt_occurrences",
                    "qwen_functional_blocks", "qwen_functional_SDPA", "qwen_native_blocks", "qwen_native_SDPA", "qwen_native_authorizations"):
            require(counters[key] == declared[key], "physical census differs: " + key)
        require(counters["native_qwen_calls"] == 1, "native Qwen full-model census differs")
        require(len(consumers) == 8 and {(row["branch"], row["logical_step"]) for row in consumers} ==
                {(case, step) for case in CASES for step in range(4)}, "context consumer census differs")
        save(output / "act-records.json", acts)
        save(output / "context-consumers.json", consumers)
        save(output / "native-contract.json", dict(counts=counters, placement_records=placements,
            component_identity_before=identities, component_identity_after=after,
            qwen_tied_alias_before=tied, qwen_tied_alias_after=True, forecast_module_entries=0,
            context_operand_occurrences=8, selected_kernel="unprofiled", genuine_qwen_act=False,
            history_helper_sha256=HISTORY_SHA256,
            scope="Qwen component suffix intervention; four genuine FLUX conditioning Acts; ideal SwiGLU identity is prose only, no extra learned RHS projection"))
        save(output / "source.json", dict(schema="mdb-flux-h18-state-history-source-v1", job_id=os.environ.get("MRUN_JOB_ID"),
            request_sha256=config.get("request_sha256"), worker_sha256=sha(__file__), helper_source_sha256=config["helper_source_sha256"],
            source_files=sources, tokenizer_files=tokenizer_files, runtime_contract_sha256=sha(output / "runtime-contract.json"),
            forecast_seal=seal_record))
        summary = dict(schema="mdb-flux-h18-state-history-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"),
            request_sha256=config.get("request_sha256"), counts=counters, contract=declared, forecast_seal=seal_record,
            transaction=verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal_record["seal_sha256"],
                                           max_bytes=config["forecast_transaction_bytes"]),
            context_operand_occurrences=len(consumers), mechanics_status="complete", trend_status="observed-uninterpreted",
            terminal_status="unassessed", seconds=time.monotonic() - started)
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
    print("MDB_H18_STATE_HISTORY_BOOTSTRAP", flush=True)
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
    summary = run(config, progress=lambda row: print("MDB_H18_HISTORY_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    e, _ = helpers()
    finalizer_path = Path(__file__).with_name("flux_image_tail_microscope.py")
    require(sha(finalizer_path) == e["FINALIZER_SHA256"], "frozen finalizer differs")
    ready = runpy.run_path(str(finalizer_path))["finalize"](config["output"], job_id=summary["job_id"],
        request_sha256=summary["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)


if __name__ == "__main__":
    main()
