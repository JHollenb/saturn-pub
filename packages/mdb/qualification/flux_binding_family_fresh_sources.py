"""Observed native prefixes ending at fresh step2/joint0 input cuts only.

No step2 joint0 block, numerical joint0 target, decoder or image is allowed.
The full frozen family rule is required before native preparation begins.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

SCHEMA = "mdb-flux-binding-family-fresh-inputs-v1"
RESULT = "flux-binding-family-fresh-inputs"
HERE = Path(__file__).resolve().parent
ORIGINAL_REQUEST = "c07bdd490d7105dd1ad042cbdc59f455588cbb2b7d922d9c7c1d9fa123a2a479"
MODEL_ID = "a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0"
POLICY = "original-native-33op-pair/image-prime23op/conditioning-Amid/direct-T-A-at-A-or-B-v1"
HELPERS = {
    "flux_binding_generalization.py": "1d9bbe842cb5cef0feaae4e84766828911e61b854eb6db95635743724f5a6276",
    "flux_binding_context_controls.py": "ea9401afcf8120c630e755504260f10666321e849f2619115f68287e6a26a26d",
    "flux_binding_matched_dose.py": "e3eafa2f00c312fbd71df5ee8bd79b69fe8c6c7c2319573433915a476f3dc14f",
    "flux_binding_route_payload_math.py": "b47e4d63550c58d46b4a6e47c8967ca78c36217bf19009af82310fa2cfb586fb",
    "flux_binding_generalization_audit.py": "d3301d522baed51d7d267632611af7b1f5de61e0b10fca7e9c6d37c50f6fac16",
    "flux_raw_to_image.py": "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "retrieve_flux_encoder.py": "68942b955d24ccca9383e18e0144beeb6f2470f80bef586a468a2233e59818ef",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def contexts():
    rows = []
    templates = (
        ("plain", "A {a} sphere on the left and a {b} cube on the right, on a plain white background."),
        ("background", "On a plain white background, a {b} cube is to the right of a {a} sphere."),
    )
    for name, template in templates:
        for seed in (2027, 2063):
            rows.append(dict(id=f"sphere-cube-{name}-{seed}", seed=seed, panel="primary",
                attributes=["red", "blue"], objects=["sphere", "cube"],
                prompts=dict(base=template.format(a="red", b="blue"), source=template.format(a="blue", b="red"))))
    for first, second in (("cone", "cylinder"), ("sphere", "cylinder")):
        template = "A {a} " + first + " on the left and a {b} " + second + " on the right, on a plain white background."
        rows.append(dict(id=f"{first}-{second}-probe-2081", seed=2081, panel="shape-probe",
            attributes=["red", "blue"], objects=[first, second],
            prompts=dict(base=template.format(a="red", b="blue"), source=template.format(a="blue", b="red"))))
    return rows


def contract():
    return dict(contexts=6, prompt_encodes=12, noise_draws=3, natural_prefix_transitions=396,
        image_primed_transitions=138, projection_transitions=12, native_transitions=546,
        native_blocks=474, acts=24, rewinds=12, input_cuts=12,
        forbidden_step2_joint0_calls=0, decoder_calls=0, images=0,
        pipeline_loads=1, encoder_cuda_placements=1, transformer_cuda_placements=1)


def prefix_point(point, *, phase=None, index=None):
    """Allow native preparation only before the fresh joint0 source boundary."""
    require(isinstance(point, dict) and type(point.get("logical_step")) is int
            and point["logical_step"] >= 0 and point.get("edge") == "before", "native prefix point differs")
    require(point["logical_step"] < 2 or (point["logical_step"] == 2 and point.get("phase") == "project"),
            "prefix stop frontier crossed")
    if phase is not None:
        require(point["logical_step"] < 2 and point.get("phase") == phase and point.get("index") == index,
                "native prefix block/full-clock differs")
    return point


def helpers():
    for name, expected in HELPERS.items():
        require(sha(HERE / name) == expected, "frozen source helper differs: " + name)
    return {name: runpy.run_path(str(HERE / name)) for name in HELPERS}


def make_config(parent, family_rule, family_rule_sha256, original_request):
    require(request_identity(original_request) == original_request["request_sha256"] == ORIGINAL_REQUEST, "original generalization request differs")
    config = {key: value for key, value in parent.items() if key not in (
        "schema", "job_id", "request_sha256", "worker_sha256", "output", "_request_record", "contexts", "arms", "mode")}
    config.update(schema=SCHEMA, output_name=RESULT, contexts=contexts(), source_policy=POLICY,
        family_rule=family_rule, family_rule_sha256=family_rule_sha256, family_rule_path="family-rule.json",
        original_request_sha256=ORIGINAL_REQUEST, original_request_path="original-generalization-request.json",
        checkpoint_files=original_request["config"]["checkpoint_files"], schedule=original_request["config"]["schedule"],
        original_source_sha256=original_request["config"]["original_source_sha256"],
        model_path=original_request["config"]["model_path"], expected_model_identity=MODEL_ID,
        max_wall_seconds=900, cpu_threads=4, no_joint0_outputs=True,
        expected_counts=contract(), helper_source_sha256=dict(HELPERS),
        transport_limits=dict(max_expanded_bytes=6 << 30, max_member_bytes=134221824),
        stop_frontier=dict(step=2, phase="joint", index=0, edge="before", next_operation="joint_block.0"),
        prefix_scope="Observed native source-assisted prefixes only; no joint0 output or numerical/image forecast",
        wrong_row_policy="valid unchanged raw-content cyclic scan from(max(S)+17)%512; disjoint same cardinality, ascending output",
        retries=0, nested_mrun=False)
    return config


def validate_rule(config, *, files=True):
    digest = config.get("family_rule_sha256", "")
    require(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
            and digest != "0" * 64, "frozen family-rule digest required")
    require(config["family_rule"].get("fresh_contexts") == contexts(), "rule must bind exact six fresh contexts")
    if files:
        require(sha(config["family_rule_path"]) == digest and read(config["family_rule_path"]) == config["family_rule"], "frozen family-rule bytes differ")
    return digest


def validate_config(config, *, files=True):
    require(config.get("schema") == SCHEMA and config.get("output_name") == RESULT
            and config.get("contexts") == contexts() and config.get("source_policy") == POLICY
            and config.get("expected_model_identity") == MODEL_ID and config.get("expected_counts") == contract()
            and config.get("no_joint0_outputs") is True and config.get("retries") == 0
            and not config.get("nested_mrun"), "prefix-only source contract differs")
    validate_rule(config, files=files)
    require(config.get("height") == config.get("width") == 512 and config.get("steps") == 4
            and config.get("batch_size") == 1 and config.get("dtype") == "bfloat16"
            and config.get("device") == "cuda", "original source geometry differs")
    if files:
        original = read(config["original_request_path"])
        require(request_identity(original) == original["request_sha256"] == ORIGINAL_REQUEST, "original source request bytes differ")
        require(config["checkpoint_files"] == original["config"]["checkpoint_files"]
                and config["schedule"] == original["config"]["schedule"]
                and config["original_source_sha256"] == original["config"]["original_source_sha256"], "original encoder/schedule metadata differs")
    return contract()


def select_addresses(context, tokens, namespaces):
    old = namespaces["flux_binding_generalization.py"]
    selected = old["resolve_rows"](tokens["base"], tokens["source"])
    a = selected["changed"]
    span_rows = sorted({row for role in ("base", "source") for span in tokens[role]["attribute_spans"] for row in span["rows"]})
    require(a == span_rows and len(a) == 2, "changed color span instrument is not aligned two-row support")
    b_info = namespaces["flux_binding_context_controls.py"]["select_rows"](context, tokens, selected)
    b = b_info["rows"]
    support = sorted(set(a) | set(b))
    start, end = b_info["raw_prompt_character_interval"]
    source_prompt = context["prompts"]["source"]
    origins = {span["start"] - source_prompt.index(span["attribute"]) for span in tokens["source"]["attribute_spans"]}
    require(len(origins) == 1, "source prompt origin differs")
    source_start = origins.pop()
    source_end = source_start + len(source_prompt)
    valid = [row for row, (offset, token) in enumerate(zip(tokens["base"]["offsets"], tokens["base"]["ids"], strict=True))
             if row not in support and tokens["base"]["mask"][row] and tokens["source"]["mask"][row]
             and token == tokens["source"]["ids"][row] and start <= offset[0] < offset[1] <= end
             and source_start <= tokens["source"]["offsets"][row][0] < tokens["source"]["offsets"][row][1] <= source_end
             and not tokens["base"]["pieces"][row].startswith("<|")]
    beginning = (max(support) + 17) % 512
    wrong = sorted(sorted(valid, key=lambda row: (row - beginning) % 512)[:len(support)])
    require(len(support) == 4 and len(wrong) == len(support) and not set(wrong) & set(support), "disjoint address-control instrument incomplete")
    return dict(A=a, B=b, S=support, wrong_rows=wrong, B_metadata=b_info,
                wrong_candidates=valid, wrong_cyclic_start=beginning, selection_observations="token IDs/offsets only")


def pair_text(base, amid, addresses, math):
    """Historical A@A/A@B transport, from supplied BF16 text bits only."""
    import torch

    require(base.shape == amid.shape and base.dtype == amid.dtype == torch.bfloat16, "projected text profile differs")
    b, a = (value.detach().contiguous().cpu().view(torch.uint16).numpy().copy() for value in (base, amid))
    ra, rb = addresses["A"], addresses["B"]
    complement = [row for row in range(base.shape[1]) if row not in ra]
    require((b[:, complement] == a[:, complement]).all(), "A.mid projection changed text outside A")
    target = math["norm"](b[:, ra], a[:, ra])
    result, recipes = {}, {}
    for role, rows in (("P", ra), ("Q", rb)):
        alpha, value, landed = math["match"](b[:, rows], b[:, ra], a[:, ra], target)
        full = b.copy()
        full[:, rows] = value
        result[role] = torch.from_numpy(full).view(torch.bfloat16).to(base.device)
        recipes[role] = dict(payload="A.mid delivered text direction", support="A" if role == "P" else "B",
            rows=rows, alpha=alpha, target_l2=target, landed_l2=landed,
            relative_norm_error=(landed-target)/target if target else None,
            arithmetic="frozen match: FP64 destination+alpha*(source_target-source_base), explicitFP32 then ties-evenBF16")
    return result, recipes


def run(config):
    import importlib.metadata

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    expected = validate_config(config)
    namespace = helpers()
    old = namespace["flux_raw_to_image.py"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    save(output / "request.json", config["_request_record"])
    shutil.copyfile(config["family_rule_path"], output / "family-rule.json")
    shutil.copyfile(config["original_request_path"], output / "original-generalization-request.json")
    counts = {key: 0 for key in expected}
    counts["contexts"] = 6
    started, events, receipts, actions = time.monotonic(), [], [], []
    active, released = dict(stage="setup", clock=None, context=None), dict(prefix=False)

    def event(kind, **fields):
        require(time.monotonic()-started <= config["max_wall_seconds"], "finite prefix wall budget exceeded")
        row = dict(number=len(events), kind=kind, seconds=time.monotonic()-started, counters=dict(counts), **fields)
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        print("MDB_FRESH_PREFIX " + canonical(row).decode(), flush=True)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "qualified native CUDA source required")
    require(str(torch.__version__).split("+")[0] == "2.13.0", "qualified Torch source changed")
    for name, digest in config["original_source_sha256"].items():
        require(sha(importlib.metadata.distribution("diffusers").locate_file(name)) == digest, "original Diffusers source changed")
    for name, digest in config["checkpoint_files"].items():
        require(sha(Path(config["model_path"]) / name) == digest, "original encoder/tokenizer changed")
    manifest = read(config["flux_manifest"])
    for entry in read(config["vae_manifest"])["artifacts"]:
        entry = json.loads(json.dumps(entry))
        entry["source"]["file"] = "vae/" + entry["source"]["file"]
        manifest["artifacts"].append(entry)
    manifest["defaults"] = dict(checkpoint_root=config["model_path"], artifact_root="parameter-cache", ssh_host="beast", python="python3")
    save(output / "parameter-artifacts.json", manifest)
    shutil.copyfile(config["flux_retrieval_script"], output / "retrieve_weights.py")
    namespace["flux_source_receiver.py"]["RecipeBank"](manifest, config["model_path"])
    save(output / "encoder-source.json", dict(checkpoint_files=config["checkpoint_files"], original_request=ORIGINAL_REQUEST))
    shutil.copyfile(HERE / "retrieve_flux_encoder.py", output / "retrieve_flux_encoder.py")
    pipe = Flux2KleinPipeline.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    counts["pipeline_loads"] += 1
    tokens, addresses = {}, {}
    for context in contexts():
        cid = context["id"]
        tokens[cid] = {}
        for role, prompt in context["prompts"].items():
            text = pipe.tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            token = pipe.tokenizer(text, padding="max_length", truncation=True, max_length=512, return_offsets_mapping=True)
            origin = text.index(prompt)
            spans = []
            for attribute in context["attributes"]:
                start = origin + prompt.index(attribute)
                spans.append(dict(attribute=attribute, start=start, end=start+len(attribute),
                    rows=[row for row, (a, b) in enumerate(token["offset_mapping"]) if a < start+len(attribute) and b > start]))
            tokens[cid][role] = dict(ids=token["input_ids"], mask=token["attention_mask"], offsets=token["offset_mapping"],
                pieces=pipe.tokenizer.convert_ids_to_tokens(token["input_ids"]), attribute_spans=spans)
        addresses[cid] = select_addresses(context, tokens[cid], namespace)
    save(output / "token-addresses.json", dict(tokens=tokens, addresses=addresses))
    store = LocalStore(output / "state-store")
    handles, encoder_handles = [], []

    def guard(_module, _args):
        require(released["prefix"], "native source work before policy seal")

    def forbidden_decoder(_module, _args):
        counts["decoder_calls"] += 1
        raise ValueError("decoder/image forbidden in prefix source job")

    def block_guard(phase, index):
        def check(_module, _args):
            guard(_module, _args)
            clock = active["clock"]
            if clock is not None and clock["logical_step"] >= 2:
                counts["forbidden_step2_joint0_calls"] += 1
                raise ValueError("step2 joint0 or later block forbidden in prefix job")
            prefix_point(clock, phase=phase, index=index)
            require(counts["native_blocks"] < expected["native_blocks"], "finite prefix block budget exhausted")
            counts["native_blocks"] += 1
        return check

    def receipt(kind, value, **fields):
        row = dict(number=len(receipts), context=active["context"], kind=kind, receipt=value.to_dict(), **fields)
        receipts.append(row)
        with (output / "receipts.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")

    def retain(session):
        cut = session.capture(retain=False)
        store.save(cut)
        return cut

    def cut_record(cut):
        return dict(cut=cut.fingerprint, parent=cut.parent, model_identity=cut.model_identity,
                    execution=dict(cut.execution), execution_point=dict(cut.execution_point), store="state-store")

    def act(session, slot, value, purpose):
        from saturn_pub import Act

        before = retain(session)
        applied = session.apply(Act.replace(slot, value, name=purpose))
        after = retain(session)
        require(describe(after.payload) == describe({**dict(before.payload), slot: value}), "source Act changed undeclared state")
        counts["acts"] += 1
        receipt("act", applied, purpose=purpose)
        actions.append(dict(context=active["context"], purpose=purpose, slot=slot,
                            before=cut_record(before), after=cut_record(after), receipt=applied.to_dict()))
        return after

    def step(session, group):
        point = dict(session.capture(retain=False).execution_point)
        require(counts["native_transitions"] < expected["native_transitions"]
                and counts[group] < expected[group], "finite prefix operation budget exhausted")
        prefix_point(point)
        active["clock"] = point
        result = session.continue_()
        counts["native_transitions"] += 1
        counts[group] += 1
        receipt("continue", result, point=point)
        active["clock"] = None

    def rewind(session, parent):
        result = session.restore(parent)
        require(session.capture(retain=False).fingerprint == parent.fingerprint, "complete source parent rewind differs")
        counts["rewinds"] += 1
        receipt("restore", result, parent=parent.fingerprint)

    def execute(_context):
        released["prefix"] = True
        encodings = {}
        pipe.text_encoder.eval().requires_grad_(False).to("cuda")
        counts["encoder_cuda_placements"] += 1
        active["stage"] = "encoder"
        for context in contexts():
            active["context"] = context["id"]
            for role, prompt in context["prompts"].items():
                value, ids = pipe.encode_prompt(prompt, device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
                encodings[context["id"], role] = (value.cpu(), ids.cpu())
                counts["prompt_encodes"] += 1
            event("prompt_pair_encoded", context=context["id"])
        for handle in encoder_handles:
            handle.remove()
        encoder_handles.clear()
        pipe.text_encoder = None
        pipe.tokenizer = None
        del value, ids
        gc.collect()
        torch.cuda.empty_cache()
        pipe.transformer.eval().requires_grad_(False).to("cuda")
        pipe.vae.eval().requires_grad_(False)
        require(pipe.vae.bn.running_var.device.type == "cpu", "native BN normalization must remain CPU")
        counts["transformer_cuda_placements"] += 1
        adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", device="cuda")
        require(adapter.model_identity == MODEL_ID, "original complete model identity differs")
        noise = {}
        for seed in sorted({context["seed"] for context in contexts()}):
            latent, image_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                torch.bfloat16, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(seed))
            noise[seed] = (latent.cpu(), image_ids.cpu())
            counts["noise_draws"] += 1
        timesteps = torch.tensor(config["schedule"]["timesteps"], dtype=torch.float32, device="cuda")
        sigmas = torch.tensor(config["schedule"]["sigmas"], dtype=torch.float32, device="cuda")
        roots, recipes, prefix_parents = {}, {}, {}
        for context in contexts():
            cid = context["id"]
            active.update(stage="prefix", context=cid)
            natural = {}
            for role in ("base", "source"):
                latent, image_ids = (value.to("cuda") for value in noise[context["seed"]])
                conditioning, text_ids = (value.to("cuda") for value in encodings[cid, role])
                session = adapter.session(latent=latent.clone(), conditioning=conditioning, img_ids=image_ids,
                                          txt_ids=text_ids, timesteps=timesteps.clone(), sigmas=sigmas.clone())
                for _ in range(33):
                    step(session, "natural_prefix_transitions")
                natural[role] = retain(session)
                point = natural[role].execution_point
                require((point["logical_step"], point["phase"], point["index"]) == (1, "joint", 4), "natural prefix clock differs")
                del session
            master = Session.from_cut(adapter, natural["base"])
            prime = master.fork(natural["base"])
            act(prime, "image", natural["source"].payload["image"], "source-image-prime")
            for _ in range(23):
                step(prime, "image_primed_transitions")
            primed = retain(prime)
            require((primed.execution_point["logical_step"], primed.execution_point["phase"]) == (2, "project"), "primed prefix clock differs")
            rewind(prime, natural["base"])
            base_c, source_c = (encodings[cid, role][0].to("cuda") for role in ("base", "source"))
            project = Session.from_cut(adapter, primed)
            step(project, "projection_transitions")
            baseline = retain(project)
            require(describe(baseline.payload["conditioning"]) == describe(base_c), "baseline conditioning lineage differs")
            a, b = (addresses[cid][name] for name in ("A", "B"))
            as_bits = lambda value: value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
            panel = namespace["flux_binding_matched_dose.py"]["dose_panel"](
                as_bits(base_c[:, a]), as_bits(source_c[:, a]), as_bits(base_c[:, b]), as_bits(source_c[:, b]))
            amid = next(spec for spec in panel if spec["arm"] == "A.mid")
            mixed = base_c.clone()
            mixed[:, a] = torch.from_numpy(amid["value"].copy()).view(torch.bfloat16).to("cuda")
            temporary = Session.from_cut(adapter, primed).fork(primed)
            act(temporary, "conditioning", mixed, "source-only-A.mid-conditioning")
            step(temporary, "projection_transitions")
            amid_input = retain(temporary)
            texts, transport = pair_text(baseline.payload["text"], amid_input.payload["text"], addresses[cid], namespace["flux_binding_route_payload_math.py"])
            rewind(temporary, primed)
            common = Session.from_cut(adapter, baseline)
            inputs = {}
            for role in ("P", "Q"):
                branch = common.fork(baseline)
                cut = act(branch, "text", texts[role], "source-A-at-" + ("A" if role == "P" else "B"))
                require(cut.execution_point["logical_step"] == 2 and cut.execution_point["phase"] == "joint"
                        and cut.execution_point["index"] == 0, "published source not at joint0 input")
                inputs[role] = cut
            nontext = lambda cut: describe({key: value for key, value in cut.payload.items() if key != "text"})
            require(nontext(inputs["P"]) == nontext(inputs["Q"]) == nontext(baseline), "paired input nontext differs")
            complement = [row for row in range(512) if row not in addresses[cid]["S"]]
            require((as_bits(inputs["P"].payload["text"][:, complement]) == as_bits(inputs["Q"].payload["text"][:, complement])).all(), "paired inputs differ outside S")
            roots[cid] = dict(inputs={role: cut_record(cut) for role, cut in inputs.items()}, support=addresses[cid]["S"],
                wrong_rows=addresses[cid]["wrong_rows"], base_conditioning=old["_save_tensor"](output, f"inputs/{cid}/base-conditioning.npy", base_c), known={})
            amid_record = {key: value for key, value in amid.items() if key != "value"}
            recipes[cid] = dict(A_mid=amid_record, transport=transport,
                baseline_text=old["_save_tensor"](output, f"inputs/{cid}/baseline-text.npy", baseline.payload["text"]),
                amid_text=old["_save_tensor"](output, f"inputs/{cid}/Amid-text.npy", amid_input.payload["text"]),
                encodings={role: old["_save_tensor"](output, f"inputs/{cid}/{role}-conditioning.npy", encodings[cid, role][0]) for role in ("base", "source")})
            prefix_parents[cid] = dict(natural={role: cut_record(cut) for role, cut in natural.items()},
                                      primed=cut_record(primed), baseline=cut_record(baseline), amid=cut_record(amid_input))
            counts["input_cuts"] += 2
            event("paired_joint0_inputs_retained", context=cid, inputs=roots[cid]["inputs"])
            del natural, master, prime, primed, project, baseline, temporary, amid_input, common, inputs, branch, cut, texts, mixed, panel
        save(output / "source-parents.json", prefix_parents)
        save(output / "actions.json", actions)
        save(output / "noise.json", {str(seed): dict(latent=old["_save_tensor"](output, f"noise/{seed}/latent.npy", values[0]),
            img_ids=old["_save_tensor"](output, f"noise/{seed}/img_ids.npy", values[1])) for seed, values in noise.items()})
        save(output / "fresh-inputs.json", dict(schema=SCHEMA, contexts=contexts(), roots=roots, recipes=recipes,
            token_addresses=dict(tokens=tokens, addresses=addresses), family_rule_sha256=config["family_rule_sha256"],
            source_policy=POLICY, counters=counts))
        return dict(counters=counts, roots=roots)

    try:
        encoder_handles.extend(module.register_forward_pre_hook(guard) for module in pipe.text_encoder.modules())
        handles.extend(module.register_forward_pre_hook(guard) for module in pipe.transformer.modules())
        handles.extend(module.register_forward_pre_hook(forbidden_decoder) for module in pipe.vae.modules())
        for phase, blocks in (("joint", pipe.transformer.transformer_blocks), ("single", pipe.transformer.single_transformer_blocks)):
            handles.extend(block.register_forward_pre_hook(block_guard(phase, index)) for index, block in enumerate(blocks))
        tx = ForecastTransaction(output / "source-transaction", model_identity=MODEL_ID,
            input_identity=hashlib.sha256(canonical(contexts())).hexdigest(), source_identity=config["family_rule_sha256"],
            frontier=dict(kind="raw-input", family_rule_sha256=config["family_rule_sha256"]),
            execution_scope="Observed native prefixes through step2/joint0 INPUT; joint0 outputs forbidden",
            budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1024, max_wall_ms=config["max_wall_seconds"]*1000))
        tx.frontier("raw-contexts-and-frozen-rule", dict(contexts=contexts(), family_rule=config["family_rule"], source_policy=POLICY))
        tx.predict("prefix-policy", lambda _ctx: dict(predictions=dict(stop_frontier=config["stop_frontier"], counts=expected, token_addresses=addresses)))
        seal = tx.seal()
        event("prefix_policy_sealed", seal_sha256=seal["seal_sha256"], family_rule_sha256=config["family_rule_sha256"])
        with torch.inference_mode():
            tx.execute("native-prefix-inputs", execute)
        tx.close()
    finally:
        for handle in (*encoder_handles, *handles):
            handle.remove()
    require(counts == expected, "actual prefix-only census differs")
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], counters=counts,
        family_rule_sha256=config["family_rule_sha256"], source_policy=POLICY, seal_sha256=seal["seal_sha256"],
        transaction=verify_transaction(output / "source-transaction", expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="prefix inputs retained", instrument_status="native source observation only; no joint0 output", seconds=time.monotonic()-started)
    save(output / "summary.json", result)
    return result


def worker(config_path):
    config = read(config_path)
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    require(request_identity(request) == request["request_sha256"] and request["config"] == config, "source request/config differs")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "staged source changed")
    job = os.environ["MRUN_JOB_ID"]
    require(job.startswith("job-") and "/" not in job, "invalid source job identity")
    derived = dict(config, job_id=job, request_sha256=request["request_sha256"], _request_record=request,
                   output=str(Path(config["output_scratch"]) / job / RESULT))
    result = run(derived)
    ready = helpers()["flux_image_tail_microscope.py"]["finalize"](derived["output"], job_id=job,
        request_sha256=request["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    return result


def load_sources(config, output, store, adapter, base):
    """Authenticate and copy only paired input cuts and baseline conditioning."""
    import torch
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import verify_transaction

    source = config["fresh_source"]
    require(adapter.model_identity == MODEL_ID, "fresh consumer original model identity differs")
    adapter.validate_execution()
    root = Path(source.get("root", source.get("output_root", ""))).resolve()
    output = Path(output).resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output)
            and not Path(store.root).resolve().is_relative_to(root), "consumer overlaps source")
    require(sha(root / "files.json") == source.get("manifest_sha256", source.get("files_manifest_sha256")), "fresh source inventory differs")
    manifest = read(root / "files.json")
    require(manifest["job_id"] == source["job_id"] and manifest["request_sha256"] == source["request_sha256"], "fresh source publication identity differs")
    consumed = {}

    def member(relative):
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe fresh source member")
        path = root / relative
        require(path.resolve().is_relative_to(root) and not any(item.is_symlink() for item in (path, *path.parents)), "escaped fresh source member")
        expected = manifest["files"][relative]
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], "fresh source member changed: " + relative)
        consumed[relative] = dict(expected)
        return path

    request = read(member("request.json"))
    require(request_identity(request) == request["request_sha256"] == source["request_sha256"], "fresh source request differs")
    require(request["source_hashes"].get(Path(__file__).name) == sha(__file__), "executed fresh source worker differs")
    validate_config(request["config"], files=False)
    require(request["config"]["family_rule"] == config["family_rule"] and request["config"]["contexts"] == contexts(), "frozen fresh rule/cohort differs")
    rule_path = member("family-rule.json")
    require(sha(rule_path) == request["config"]["family_rule_sha256"] == config.get("family_rule_file_sha256")
            and read(rule_path) == config["family_rule"], "published rule bytes differ")
    summary = read(member("summary.json"))
    table = read(member("fresh-inputs.json"))
    require(summary["schema"] == SCHEMA and summary["job_id"] == source["job_id"]
            and summary["request_sha256"] == source["request_sha256"] and summary["source_policy"] == POLICY
            and summary["family_rule_sha256"] == request["config"]["family_rule_sha256"], "fresh source summary identity differs")
    require(table["schema"] == SCHEMA and table["contexts"] == contexts() and table["counters"] == summary["counters"] == contract()
            and table["family_rule_sha256"] == request["config"]["family_rule_sha256"] and table["source_policy"] == POLICY, "fresh source stop/counters differ")
    for name in manifest["files"]:
        if name.startswith("source-transaction/"):
            member(name)
    verification = verify_transaction(root / "source-transaction", expected_seal_sha256=summary["seal_sha256"])
    original = LocalStore(root / "state-store")
    old = helpers()["flux_raw_to_image.py"]
    result = {}
    for cid, row in table["roots"].items():
        cuts = {}
        for role in ("P", "Q"):
            record = row["inputs"][role]
            metadata = read(member("state-store/cuts/" + record["cut"] + ".json"))
            require(metadata["schema"] == "saturn-pub-local-cut-v2", "authoritative public v2 input cut required")
            for page in metadata["slots"].values():
                member("state-store/pages/" + page + ".json")
            for blob in metadata["physical_blobs"]:
                member("state-store/blobs/" + blob + ".safetensors")
            cut = original.load(record["cut"], device="cpu")
            adapter.validate(cut.payload)
            require(cut.model_identity == record["model_identity"] == adapter.model_identity
                    and dict(cut.execution_point) == record["execution_point"] == adapter.point(cut.payload).to_dict()
                    and cut.parent == record["parent"] and dict(cut.execution) == record["execution"], "fresh input model/full point differs")
            require(cut.execution_point["logical_step"] == 2 and cut.execution_point["phase"] == "joint"
                    and cut.execution_point["index"] == 0, "fresh input frontier differs")
            expected_execution = dict(adapter.execution, device=dict(cut.execution)["device"])
            require(dict(cut.execution) == expected_execution and cut.execution["dtype"] == "torch.bfloat16"
                    and cut.execution["device"]["type"] == "cuda", "fresh input original execution profile differs")
            store.save(cut)
            cuts[role] = cut
        p, q = cuts["P"], cuts["Q"]
        require(describe({k: v for k, v in p.payload.items() if k != "text"}) == describe({k: v for k, v in q.payload.items() if k != "text"}), "fresh P/Q nontext differs")
        selected = table["token_addresses"]["addresses"][cid]
        require(selected == select_addresses(next(context for context in contexts() if context["id"] == cid),
                table["token_addresses"]["tokens"][cid], helpers()), "fresh token-only row recipe differs")
        require(row["support"] == selected["S"] and row["wrong_rows"] == selected["wrong_rows"], "fresh support/control rows differ")
        complement = [index for index in range(512) if index not in row["support"]]
        require(torch.equal(p.payload["text"][:, complement].contiguous().view(torch.uint8), q.payload["text"][:, complement].contiguous().view(torch.uint8)), "fresh input text differs outside support")
        member(row["base_conditioning"]["path"])
        conditioning = base["load_tensor"](root, row["base_conditioning"], "cpu")
        require(describe(conditioning) == describe(q.payload["conditioning"]), "fresh baseline conditioning differs")
        result[cid] = dict(inputs={role: base["cut_record"](cut) for role, cut in cuts.items()},
            support=row["support"], wrong_rows=row["wrong_rows"], known={},
            base_conditioning=old["_save_tensor"](output, f"fresh-inputs/{cid}/base-conditioning.npy", conditioning))
    require(set(result) == {context["id"] for context in contexts()}, "fresh input cohort incomplete")
    return result, dict(source=source, family_rule_sha256=sha(rule_path), source_policy=POLICY,
        transaction=verification, verification_scope="Only paired input v2 CAS closures, input metadata and baseline conditioning copied",
        consumed_members=consumed, model_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
