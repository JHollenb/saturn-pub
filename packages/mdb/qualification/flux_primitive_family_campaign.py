"""Execute a compiled finite primitive family with one global numeric barrier."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "mdb-flux-primitive-family-campaign-v1"
RESULT = "flux-primitive-family-result"
MODEL_ID = "a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0"
CONSUMER_PORTS = ("image_in", "text_in", "A", "M", "image_out", "text_out", "projection_return", "gate")
HELPERS = {
    "flux_primitive_family_writer.py": "fb4832ce7a1083e7f14881ff70bc253f2e5f5533a5c0f9b04193989ac66fe9f0",
    "flux_primitive_family_sources.py": "967fc9c06dd069449045474b87d2e9c7cc889bce746a1f25eaa5ac42011f260e",
    "flux_parallel_family_writer.py": "23c02673a04bfa8e0f28e6fd70a2f104395f3c22434acd0de1f550c8da0a372f",
    "flux_binding_residual_write.py": "7f8cd8152a20fa7586ef215fdaf89e2a8b969171b203806130f4b03b1b7b8430",
    "flux_binding_residual_sources.py": "5b7f41c17d976566e75fc734d5f9fad901448bd23c4a466bf3a65b61748e7caa",
    "flux_lifecycle_campaign.py": "bfd2abf8e67329d0770cb6543ca0691063ada1aa49dcb0e66afda3161e833034",
    "flux_binding_support_sources.py": "217331e363a698babfdd9349fad8b4bc786601eb2d76220a3c733507b26e41a0",
    "flux_common_differential_campaign.py": "7bd33723fc2d8bfd0de5af7fc4bf46cda7d9bf7ae01d58b995ab3354049c51a6",
    "flux_centered_write_campaign.py": "9010b467812ce7efd2caf73197d18b79f24a10e3569b5ba727d8440e2f01ad67",
    "flux_raw_to_image.py": "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def common_helpers():
    require(sha(HERE / "flux_binding_residual_write.py") == HELPERS["flux_binding_residual_write.py"], "original helper root differs")
    return runpy.run_path(str(HERE / "flux_binding_residual_write.py"))["common_helpers"]()


def make_config(parent):
    config = copy.deepcopy({key: value for key, value in parent.items()
        if key not in {"schema", "output_name", "job_id", "request_sha256", "output", "_request_record", "worker_sha256"}})
    config.update(schema=SCHEMA, output_name=RESULT, frozen_helpers=dict(HELPERS), cpu_threads=4,
        max_wall_seconds=1200, forecast_transaction_bytes=12 << 30,
        transport_limits=dict(max_expanded_bytes=12 << 30, max_member_bytes=134221824),
        retry=False, nested_mrun=False, scientific_gates=[])
    return config


def contract(config):
    size, actions = len(config["contexts"]), config["compiled_protocol"]["actions"]
    require(len(actions) == 6 and sum(row["role"] != "natural-reference" for row in actions) == 4,
        "qualified two-factor six-action panel required")
    return dict(functional_source_blocks=2*size, native_source_blocks=2*size,
        functional_primitive_calls=4*size, native_primitive_calls=4*size,
        functional_hybrid_projections=4*size, native_hybrid_projections=4*size,
        functional_transitions=606*size, native_target_transitions=606*size, native_blocks=542*size,
        functional_vae=6*size, native_vae=6*size, acts=10*size, forecast_acts=5*size,
        native_acts=5*size, rewinds=size)


def validate_config(config):
    """Structural staging validation; no arrays, checkpoints or paths are opened."""
    from mdb.computation_protocol import verify_plan

    require(config.get("schema") == SCHEMA and config.get("output_name") == RESULT
        and config.get("family") in {"sdpa", "swiglu"}, "unknown primitive campaign")
    plan = verify_plan(config["compiled_protocol"])
    require(plan["specification"]["model_identity"] == MODEL_ID
        and plan["specification"]["continuation"]["transitions"] == 101,
        "qualified original model/horizon differs")
    contexts = config["contexts"]
    require(isinstance(contexts, list) and 1 <= len(contexts) <= 4
        and len({row["id"] for row in contexts}) == len(contexts)
        and contexts == config["family_rule"]["fresh_contexts"], "explicit context subset differs")
    require(config["family_rule"]["phase"] in {"development", "fresh"}
        and config["family_rule"]["protocol_sha256"] == plan["plan_sha256"]
        and config["arms"] == [row["id"] for row in plan["actions"]], "compiled assay metadata differs")
    effect = config["family_rule"].get("effect_rule")
    if config["family_rule"]["phase"] == "fresh":
        require(isinstance(effect, dict) and effect.get("plan_sha256") == plan["plan_sha256"]
            and effect.get("fresh_contexts") == [row["id"] for row in contexts]
            and not set(effect.get("development_contexts", [])).intersection(effect["fresh_contexts"]), "fresh effect rule missing or overlaps")
    else:
        require(effect is None, "development cannot consume a selected fresh rule")
    require(config["frozen_helpers"] == HELPERS and config.get("scientific_gates") == []
        and not config.get("retry") and not config.get("nested_mrun"), "single finite exploratory lease required")
    require(config.get("height") == config.get("width") == 512 and config.get("steps") == 4
        and config.get("batch_size") == 1 and config.get("device") == "cuda"
        and config.get("dtype") == "bfloat16", "original geometry/profile differs")
    require(isinstance(config["fresh_source"], dict) and {"output_root", "files_manifest_sha256", "job_id", "request_sha256"} <= set(config["fresh_source"])
        and len(config["family_rule_file_sha256"]) == 64, "source publication/frozen policy required")
    return contract(config)


def rehydrate_source_ports(writer, record, active, state, family, load):
    """Reconnect owned saved values to original live-layout declarations.

    ``load`` follows the declared NPY reload convention. The primitive adapter
    restores consumed layouts separately; it never confuses a C reload with
    the original padded fused-projection view.
    """
    names = writer["source_ports"](family)
    require(set(record["ports"]) == set(names) == set(record["retained_descriptors"]), "saved source retention differs")
    ports = writer["source_writer"]()["Ports"](active, record["profile"], names, None)
    ports.update({name: load(value) for name, value in record["ports"].items()})
    ports.descriptors.update(record["descriptors"])
    ports.retained_descriptors.update(record["retained_descriptors"])
    ports.state = state
    return ports


def run(config):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from saturn_pub.core import Session, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.computation_protocol import execute_program
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    expected = validate_config(config)
    for name, digest in HELPERS.items():
        require(sha(HERE / name) == digest, "frozen helper differs: " + name)
    writer = runpy.run_path(str(HERE / "flux_primitive_family_writer.py"))
    source_writer = writer["source_writer"]()
    prefix = runpy.run_path(str(HERE / "flux_primitive_family_sources.py"))
    parent = runpy.run_path(str(HERE / "flux_binding_residual_write.py"))
    lifecycle, _ = parent["frozen_helpers"]()
    _, base, old = common_helpers()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started, phase, released, observed = time.monotonic(), dict(kind="loading"), dict(native=False), {}
    counts = dict.fromkeys(expected, 0)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    save(output / "request.json", config["_request_record"])
    require(sha("family-rule.json") == config["family_rule_file_sha256"] and read("family-rule.json") == config["family_rule"], "assay bytes differ")
    shutil.copyfile("family-rule.json", output / "family-rule.json")

    def event(kind, **fields):
        value = dict(kind=kind, seconds=time.monotonic()-started, counts=dict(counts), **fields)
        require(value["seconds"] <= config["max_wall_seconds"], "finite campaign wall allowance exhausted")
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(value) + b"\n")
        print("MDB_PRIMITIVE_FAMILY " + canonical(value).decode(), flush=True)

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "qualified CUDA profile required")
    modules = old["_helper_modules"](config)
    qw, graph, decoder = (modules[old[key]] for key in ("QWEN_WORKER", "DENOISER_HELPER", "DECODER_HELPER"))
    runtime = qw["_runtime_contract"](config, torch)
    runtime["native_diffusers_sources"] = old["_native_source_contract"](config)["sources"]
    save(output / "runtime-contract.json", runtime)
    for key in old["_source_keys"]():
        path = Path(config[key])
        require(sha(path) == config[key + "_sha256"], "original metadata differs: " + key)
        target = output / "source" / key / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = read(config["flux_manifest"])
    for entry in read(config["vae_manifest"])["artifacts"]:
        entry = copy.deepcopy(entry)
        entry["path"] = "vae-" + entry["path"]
        entry["source"]["file"] = "vae/" + entry["source"]["file"]
        manifest["artifacts"].append(entry)
    manifest["defaults"] = dict(artifact_root="parameter-cache", checkpoint_root=config["flux_parameter_root"], ssh_host="beast", python="python3")
    manifest["provenance"] = dict(schema=SCHEMA, originalParametersOnly=True, scope="169transformer+142decoder recipes; no parameter exports")
    runpy.run_path(config["flux_retrieval_script"])["validate_manifest"](manifest)
    save(output / "parameter-artifacts.json", manifest)
    shutil.copyfile(config["flux_retrieval_script"], output / "retrieve_weights.py")
    flux, _, _ = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae = AutoencoderKLFlux2.from_pretrained(config["vae_parameter_root"], torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False)
    graph["validate_model"](flux)
    models, parameter_prefix = dict(flux=flux, vae=vae), dict(flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
    identities = {name: old["_component_signature"](model, prefixes=parameter_prefix[name]) for name, model in models.items()}

    def place(name, device):
        if name == "flux":
            flux.to(device)
        else:
            vae.post_quant_conv.to(device)
            vae.decoder.to(device)
        old["_assert_signature_equal"](identities[name], old["_component_signature"](models[name], prefixes=parameter_prefix[name]), name)
        if device == "cpu":
            torch.cuda.empty_cache()
        event("placement", component=name, device=device)

    def adapter(device):
        return parent["fresh_binding_adapter"](flux, vae, config, lifecycle, device=device)

    store = LocalStore(output / "state-store")

    def retain(cut):
        store.save(cut)
        return base["cut_record"](cut)

    def component(active, state, prior):
        return retain(StateCut(active.model_identity, {**dict(active.execution), "composition": SCHEMA}, active.boundary(state),
            prior, state, active.point(state).to_dict(), active.surface(state).to_dict()))

    active = adapter("cpu")
    require(active.model_identity == MODEL_ID, "strict original pipeline identity differs")
    save(output / "source-identity-qualification.json", dict(model_identity=active.model_identity,
        execution=dict(active.execution), transformer_config=dict(flux.config),
        original_full_vae_identity=True, original_transformer_parameter_signature=True,
        placement="CPU before any forecast or native selected block", identity_override=False))
    roots, provenance = prefix["load_sources"](config, output, store, active)
    save(output / "source-roots.json", roots)
    save(output / "source-provenance.json", provenance)
    del active
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels)-1)*2)
    family, plan, actions = config["family"], config["compiled_protocol"], config["compiled_protocol"]["actions"]
    states, records, endpoints, constructions, paths, consumers = {}, {}, {}, {}, {}, dict(forecast={}, native={})
    transaction = {}

    def save_ports(relative, values):
        return {name: old["_save_tensor"](output, f"{relative}/{name}.npy", value) for name, value in values.items()}

    def capture_sources(lane):
        active = adapter("cuda")
        for cid, root in roots.items():
            for role in ("Q", "P"):
                cut = store.load(root["inputs"][role]["cut"], device="cuda")
                if lane == "forecast":
                    result = writer["functional_single"](active, cut.payload, family=family, expected_model_identity=active.model_identity)
                    state = component(active, result["state"], cut.fingerprint)
                    counts["functional_source_blocks"] += 1
                else:
                    session = Session(active, cut.payload, parent=cut.parent)
                    with writer["native_single"](active, cut.payload, family=family, expected_model_identity=active.model_identity) as result:
                        session.continue_()
                    require(describe(result["state"]) == describe(session.capture(retain=False).payload), "native source state differs")
                    state = retain(session.capture(retain=False))
                    counts["native_source_blocks"] += 1
                    del session
                states[cid, role, lane] = state
                records[cid, role, lane] = dict(state=state, ports=save_ports(f"writer/{cid}/{role}/{lane}", result["ports"]),
                    descriptors=result["descriptors"], retained_descriptors=result["retained_descriptors"], retention=result["retention"],
                    profile=result["profile"], calls=result.get("native_calls", result.get("operation_counts")))
                del result, cut
            event(lane + "_sources_complete", context=cid)
        return dict(contexts=len(roots), source_blocks=2*len(roots))

    def reload_ports(active, cid, role, lane):
        record = records[cid, role, lane]
        return rehydrate_source_ports(writer, record, active, store.load(record["state"]["cut"], device="cuda").payload,
            family, lambda value: base["load_tensor"](output, value, "cuda"))

    def construct_endpoints(lane):
        active = adapter("cuda")
        for cid in roots:
            q, p = (reload_ports(active, cid, role, lane) for role in ("Q", "P"))
            for action in actions:
                aid = action["id"]
                result = writer["construct_endpoint"](q, p, action, family=family, plan=plan, native=lane == "native")
                endpoints[cid, lane, aid] = old["_save_tensor"](output, f"endpoints/{cid}/{lane}/{aid}.npy", result["image"])
                retained = {} if action["role"] == "natural-reference" else {
                    name: old["_save_tensor"](output, f"endpoints/{cid}/{lane}/{aid}-{name}.npy", result["ports"][name])
                    for name in ("primitive_return", "projection_return")}
                constructions[cid, lane, aid] = dict(descriptors=result["descriptors"], construction=result["construction"], ports=retained,
                    retention=dict(full_arrays=sorted(retained), hash_and_layout_only=sorted(set(result["descriptors"]) - set(retained))))
                if action["role"] != "natural-reference":
                    counts[("functional" if lane == "forecast" else "native") + "_primitive_calls"] += 1
                    counts[("functional" if lane == "forecast" else "native") + "_hybrid_projections"] += 1
                del result
            del q, p
            event(lane + "_endpoints_complete", context=cid)
        save(output / (lane + "-source-writers.json"), {cid: dict(sources={role: records[cid, role, lane] for role in ("P", "Q")},
            endpoints={action["id"]: endpoints[cid, lane, action["id"]] for action in actions},
            constructions={action["id"]: constructions[cid, lane, action["id"]] for action in actions}) for cid in roots})
        return dict(contexts=len(roots), primitive_calls=4*len(roots), projections=4*len(roots))

    def edit(session, value, lane):
        before, after, receipt = base["apply_slot_edit"](session, "image", value, mode="replace")
        require(describe({key: value for key, value in before.payload.items() if key != "image"})
            == describe({key: value for key, value in after.payload.items() if key != "image"}), "image Act changed another slot")
        counts["acts"] += 1
        counts[lane + "_acts"] += 1
        retain(before)
        return retain(after), receipt.to_dict()

    def continuations(lane):
        active = adapter("cuda")
        for cid in roots:
            q = store.load(states[cid, "Q", lane]["cut"], device="cuda")
            master = Session(active, q.payload, parent=q.fingerprint if lane == "forecast" else q.parent)
            common = master.capture(retain=False)
            retain(common)
            for action in actions:
                aid, key = action["id"], cid + "/" + action["id"]
                branch = output / "cases" / cid / aid
                branch.mkdir(parents=True, exist_ok=True)
                if action["role"] == "natural-reference" and action["source"] == "P":
                    p = store.load(states[cid, "P", lane]["cut"], device="cuda")
                    session = Session(active, p.payload, parent=p.fingerprint if lane == "forecast" else p.parent)
                    start = retain(session.capture(retain=False))
                    save(branch / (lane + "-entry-act.json"), dict(reference="natural-P", source=states[cid, "P", lane], after=start, receipt=None))
                    del p
                else:
                    session = master.fork(common)
                    start, receipt = edit(session, base["load_tensor"](output, endpoints[cid, lane, aid], "cuda"), lane)
                    if action["role"] == "natural-reference":
                        require(action["source"] == "Q" and describe(session.capture(retain=False).payload) == describe(common.payload), "natural-Q sham changed full state")
                    save(branch / (lane + "-entry-act.json"), dict(common=base["cut_record"](common), after=start, receipt=receipt))
                state, prior = session.capture(retain=False).payload, start["cut"]
                rows = [dict(number=0, fields=base["tensor_tree"](state, old))]
                for number in range(1, 102):
                    clock = {name: state[name] for name in ("step", "phase", "cursor")}
                    if number == 1:
                        if lane == "forecast":
                            result = source_writer["functional_single"](active, state, index=5,
                                expected_model_identity=active.model_identity, retain_ports=CONSUMER_PORTS)
                            state = result["state"]
                        else:
                            with source_writer["native_single"](active, state, index=5,
                                    expected_model_identity=active.model_identity, retain_ports=CONSUMER_PORTS) as result:
                                session.continue_()
                            state = session.capture(retain=False).payload
                            require(describe(result["state"]) == describe(state), "native first consumer state differs")
                        consumers[lane][key] = dict(ports=save_ports(f"cases/{cid}/{aid}/consumer/{lane}", result["ports"]),
                            descriptors=result["descriptors"], retained_descriptors=result["retained_descriptors"], retention=result["retention"])
                        del result
                    elif lane == "forecast":
                        state = base["step_equations"](graph, flux, state)
                    else:
                        session.continue_()
                        state = session.capture(retain=False).payload
                    counts["functional_transitions" if lane == "forecast" else "native_target_transitions"] += 1
                    rows.append(dict(number=number, event=clock, fields=base["tensor_tree"](state, old)))
                require((state["step"], state["phase"], state["cursor"]) == (4, "project", 0), "suffix horizon differs")
                final = component(active, state, prior) if lane == "forecast" else retain(session.capture(retain=False))
                save(branch / (lane + "-rows.json"), rows)
                paths.setdefault(key, dict(context=cid, arm=aid, action=action, root=str(branch.relative_to(output))))[lane + "_final"] = final
                if lane == "native" and action["role"] == "natural-reference" and action["source"] == "Q":
                    session.restore(common)
                    require(session.capture(retain=False).fingerprint == common.fingerprint, "natural-Q parent rewind differs")
                    counts["rewinds"] += 1
                del state, session, rows
                event(lane + "_suffix_complete", case=key)
            del q, common, master
        save(output / (lane + "-paths.json"), paths)
        save(output / (lane + "-consumer-ports.json"), consumers[lane])
        return dict(futures=6*len(roots), transitions=606*len(roots))

    def decodes(lane):
        for path in paths.values():
            branch = output / path["root"]
            state = store.load(path[lane + "_final"]["cut"], device="cpu").payload
            if lane == "forecast":
                argument = old["_decoder_input"](vae, state, config, {"Flux2KleinPipeline": Flux2KleinPipeline}).to("cuda")
                decoded = decoder["functional_decode"](vae, argument, counts={}, profile="original")
                record = base["decoder_result"](old, branch, lane, decoded, processor)
                record["decoder_input"] = old["_save_tensor"](branch, "decoder/forecast/decoder-input.npy", argument)
                del argument, decoded
            else:
                record = old["_native_decoder"](branch, vae, state, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, {}, {})
            save(branch / ("decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json"), record)
            counts["functional_vae" if lane == "forecast" else "native_vae"] += 1
            del state
        return dict(decodes=6*len(roots))

    def seal_forecasts(_command):
        inventory = old["_inventory"](output, exclude=("timeline.jsonl", "program-receipts.jsonl"))
        tx = ForecastTransaction(output / "forecast-transaction", model_identity=MODEL_ID,
            input_identity=hashlib.sha256(canonical(roots)).hexdigest(), source_identity=plan["plan_sha256"],
            frontier=dict(kind="architectural", scope="Observed natural single4 input cuts; generated primitive factors and full own-state futures", external_files=inventory),
            execution_scope="Original-weight equations and shared primitive kernels; no intermediate native refresh",
            budgets=dict(max_predictions=1, max_executions=4, max_bytes=config["forecast_transaction_bytes"], max_files=30000,
                max_wall_ms=config["max_wall_seconds"]*1000))
        tx.frontier("compiled-family-and-inputs", dict(roots=roots, plan=plan, rule=config["family_rule"]))
        tx.predict("all-numeric-futures", lambda _: dict(predictions=dict(external_files=inventory, paths=paths,
            native_source_blocks_at_seal=counts["native_source_blocks"], native_primitive_calls_at_seal=counts["native_primitive_calls"],
            native_targets_at_seal=counts["native_target_transitions"], native_decodes_at_seal=counts["native_vae"])))
        seal = tx.seal()
        transaction.update(tx=tx, seal=seal, inventory=inventory)
        event("all_forecasts_sealed", seal=seal["seal_sha256"])
        return dict(seal_sha256=seal["seal_sha256"], native_calls_before_seal=0)

    def native_stage(stage, callback):
        old["_verify_inventory"](output, transaction["inventory"])
        released["native"] = True
        phase["kind"] = "target-" + stage
        result = {}

        def execute(_context):
            if stage == "sources":
                place("flux", "cuda")
            result.update(callback("native"))
            if stage == "continuations":
                place("flux", "cpu")
            if stage == "decodes":
                place("vae", "cpu")
            return dict(result)

        if stage == "decodes":
            place("vae", "cuda")
        transaction["tx"].execute("native-" + stage, execute)
        return result

    def finish(_command):
        transaction["tx"].close()
        old["_verify_inventory"](output, transaction["inventory"])
        require(counts == expected, "physical compiled campaign census differs")
        for name, model in models.items():
            old["_assert_signature_equal"](identities[name], old["_component_signature"](model, prefixes=parameter_prefix[name]), name)
        save(output / "state-index.json", paths)
        save(output / "consumer-ports.json", consumers)
        save(output / "native-contract.json", dict(counts=counts, observed=observed, identities=identities))
        return dict(mechanics_status="completed", scientific_outcomes="preserved for independent saved-only audit", counts=counts)

    def forecast_stage(stage, callback):
        phase["kind"] = "forecast-" + stage
        if stage == "sources":
            place("flux", "cuda")
        if stage == "decodes":
            place("flux", "cpu")
            place("vae", "cuda")
        result = callback("forecast")
        if stage == "decodes":
            place("vae", "cpu")
        return result

    def entered(_module, _args):
        require(released["native"] and phase["kind"].startswith("target"), "native transformer before global numeric seal")
        counts["native_blocks"] += 1

    def record_program(row):
        with (output / "program-receipts.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")

    handles = [block.register_forward_pre_hook(entered) for block in (*flux.transformer_blocks, *flux.single_transformer_blocks)]
    handlers = {
        "forecast_sources": lambda _: forecast_stage("sources", capture_sources),
        "forecast_endpoints": lambda _: forecast_stage("endpoints", construct_endpoints),
        "forecast_continuations": lambda _: forecast_stage("continuations", continuations),
        "forecast_decodes": lambda _: forecast_stage("decodes", decodes), "seal": seal_forecasts,
        "native_sources": lambda _: native_stage("sources", capture_sources),
        "native_endpoints": lambda _: native_stage("endpoints", construct_endpoints),
        "native_continuations": lambda _: native_stage("continuations", continuations),
        "native_decodes": lambda _: native_stage("decodes", decodes), "audit": finish,
    }
    try:
        with base["instrument"](list(models.values()), phase, released, observed), torch.inference_mode():
            program = execute_program(plan, handlers, record=record_program)
    finally:
        for handle in handles:
            handle.remove()
    save(output / "program.json", program)
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], family=family,
        protocol_sha256=plan["plan_sha256"], counts=counts, observed=observed, contexts=list(roots), arms=config["arms"],
        forecast_seal=transaction["seal"]["seal_sha256"], transaction=verify_transaction(output / "forecast-transaction"),
        mechanics_status="completed", instrument_status="awaiting-independent-saved-audit", terminal_status="unassessed", seconds=time.monotonic()-started)
    save(output / "summary.json", result)
    return result


def worker(config_path):
    config, request = read(config_path), read(os.environ["SATURN_DEBUG_REQUEST"])
    identity = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "request/config differs")
    for name, digest in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == digest, "executed payload differs")
    validate_config(config)
    job = os.environ["MRUN_JOB_ID"]
    config.update(job_id=job, request_sha256=identity, _request_record=request, output=str(Path(config["output_scratch"]) / job / RESULT))
    result = run(config)
    ready = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))["finalize"](config["output"], job_id=job,
        request_sha256=identity, **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
