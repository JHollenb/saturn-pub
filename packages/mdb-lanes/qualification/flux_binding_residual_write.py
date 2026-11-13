"""Sealed residual-input versus finite joint0-net-write binding futures.

Observed paired states and original parameters are inputs. Every own-state
functional future and image precedes the single release of new native targets.
"""
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
SCHEMA = "mdb-flux-binding-residual-write-v1"
RESULT = "flux-binding-residual-write-result"
CONTEXTS = ("anchor", "wording")
ARMS = ("whole_Q", "residual_only", "write_only", "both_RW")
SOURCE = dict(root="/mnt/big/mdb/research/job-b0941d817597/flux-lifecycle-result",
    manifest_sha256="9629243e77453ee394d57a24106b4df421724027a606851f4f1c9af77270d1e7",
    request_sha256="57eda08824c5b130080c9ac46e81ca7cfdbdbc2c320d8618eee76b3b1ac2bbbb", job_id="job-b0941d817597")
HELPERS = {
    "flux_binding_residual_sources.py": "e49457864aaefe8ec3d429aa8dc7f5d906b12b61359d49ce4c65d5369033a5eb",
    "flux_lifecycle_campaign.py": "af2eeaf0ee069c57c911cf53cbd12f12a637c5f2d226f16a2d507a9521a952c1",
    "flux_binding_support_sources.py": "217331e363a698babfdd9349fad8b4bc786601eb2d76220a3c733507b26e41a0",
    "flux_common_differential_campaign.py": "8b399ee217e75588a7949f8d1efd955cb0728e6daa1f4484bf281e22172afd3d",
    "flux_centered_write_campaign.py": "bf4608c2c732d713c532a0edf81c4aef32d4d7dcc0a6ddf2867d3a9c1421e04a",
    "flux_raw_to_image.py": "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "7f13836ae94c84b7ed0f7df67b1f7b1d28a7e534a83d509fa7320a09a6d845df",
    "flux_image_tail_microscope.py": "b9efb61874005c8b11307c197a2ea5299078b42d5b4ab4b5cc719f7a4a4a32bb",
    "flux_source_receiver.py": "d483e41bfac018c8225662e8417bcef46f6916fa2a14d910f16e9a12beb5250c",
}
SCOPE = dict(kind="architectural", origin="observed-paired-joint0-inputs-and-outputs",
    raw_prompt_forecast=False, source_writer_forecast=False, independent_kernel_prediction=False,
    scope="Own-state original shared-primitive denoiser/scheduler/VAE forecasts; residual versus finite net-write endpoints on original S, fixed Q nontext")


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


def frozen_helpers():
    for name, expected in HELPERS.items():
        require(sha(HERE / name) == expected, "frozen helper changed: " + name)
    return runpy.run_path(str(HERE / "flux_lifecycle_campaign.py")), runpy.run_path(str(HERE / "flux_binding_residual_sources.py"))


def common_helpers():
    lifecycle, _ = frozen_helpers()
    return lifecycle["common_helpers"]()


def contract():
    return dict(contexts=2, image_futures=8, known_replay_conditions=4, new_modified_conditions=4,
        functional_transitions=432, native_target_transitions=432, native_blocks=392,
        functional_vae_calls=8, native_vae_reference_calls=8, acts_per_lane=16,
        first_consumer_port_sets_per_lane=8, first_consumer_ports_per_lane=48, native_source_restores=2,
        component_hydrations=2, cuda_placements=4, cpu_offloads=4, max_gpu_components=1,
        native_qwen_source_calls=0, native_project_source_transitions=0, native_joint0_source_transitions=0,
        noise_draws=0, functional_module_forwards=0, persistent_weight_banks=0)


def make_config(parent):
    remove = {"schema", "output_name", "contexts", "arms", "binding_arms", "job_id", "request_sha256",
              "worker_sha256", "output", "_request_record", "hypotheses"}
    config = {key: value for key, value in parent.items() if key not in remove and not key.startswith("cd_")}
    config.update(schema=SCHEMA, output_name=RESULT, contexts=list(CONTEXTS), arms=list(ARMS), source=dict(SOURCE),
        helper_source_sha256=dict(HELPERS), source_frontier=dict(SCOPE), component_hydrations=2,
        cuda_placements=4, offloads=4, max_gpu_components=1, cpu_threads=4, max_wall_seconds=900,
        forecast_transaction_bytes=3 << 30, transport_limits=dict(max_expanded_bytes=8 << 30, max_member_bytes=134221824),
        scientific_gates=[], endpoint_contract="Qout clone; on S direct Torch CPU FP64.to(BF16) of Qout+DR, Qout+DW, Qout+(DR+DW); outside S exact Q bits",
        payload_rule="DR=Pin-Qin; DW=(Pout-Pin)-(Qout-Qin); full finite net write, no equal-norm, no refit",
        finite_factorial="RW-R-W+Q includes measured endpoint cast remainder and full finite native consumer",
        decoder_contract="8 functional complete-image forecasts and8 original native-final decodes; four historical known endpoint controls",
        first_consumer="Original step2/joint1 image/text normalization, R_attention,x,M_base,gate_ff; complete next state",
        retention="All source FP64 arrays/deltas/cast landings; exact typed SET receipts, full first-consumer and carrier landmarks, all boundary hashes, complete images",
        hypotheses=dict(residual="Residual-only and write-only can each transfer, partly transfer or fail; retain every outcome",
                        write="DW is net finite joint0 difference including both attention and MLP additions; no identified writer claim",
                        controls="Q and RW reproduce the exact source whole_Q/support_P payloads and supply known continuation controls",
                        numerical="All eight full own-state futures and images sealed before any new native target; shared primitive profile"))
    return config


def validate_config(config):
    for key, value in make_config({}).items():
        require(config.get(key) == value, "bounded campaign field differs: " + key)
    require(config.get("height") == config.get("width") == 512 and config.get("steps") == 4
            and config.get("batch_size") == 1 and config.get("device") == "cuda"
            and config.get("dtype") == "bfloat16" and config.get("guidance_scale", 1.) == 1., "qualified geometry differs")
    require(not config.get("retry") and not config.get("nested_mrun") and not config.get("reference_image"), "finite single lease required")
    _, _, old = common_helpers()
    for key in old["_source_keys"]():
        require(config.get(key) and sha(config[key]) == config.get(key + "_sha256"), "original metadata differs: " + key)
    require(config["binding_plan"]["schema"] == "mdb-binding-support-complement-next-cut-v1", "original source plan differs")
    return contract()


def fresh_binding_adapter(flux, vae, config, lifecycle, *, device):
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    # Original native pipeline loader changes exactly this configuration field.
    recipe_config = dict(flux.config)
    require(recipe_config.get("_name_or_path") in {
        "/raid/yiyi/klein-4b-distilled-diffusers/transformer",
        str(Path(config["flux_parameter_root"]) / "transformer")}, "unqualified transformer loader path")
    recipe_config["_name_or_path"] = "/raid/yiyi/klein-4b-distilled-diffusers/transformer"
    profile = lifecycle["transformer_profiles"](config, recipe_config)["binding"]
    flux.register_to_config(**profile)
    require(dict(flux.config) == profile, "historical binding loader profile differs")
    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"])),
        vae=vae, transformer=flux, text_encoder=None, tokenizer=None, is_distilled=True)
    return Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", device=device)


def run(config, *, progress=None):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from saturn_pub.core import Session, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    declared = validate_config(config)
    lifecycle, source_helper = frozen_helpers()
    common, base, old = common_helpers()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    save(output / "request.json", config["_request_record"])
    started = time.monotonic()
    counts = {key: 0 for key in declared if "per_lane" not in key and key != "max_gpu_components"}
    counts.update(contexts=2, image_futures=8, known_replay_conditions=4, new_modified_conditions=4)
    phase, released, observed = dict(kind="source-loading"), dict(native=False), {}
    events, placements, emissions = [], [], []

    def event(kind, **details):
        row = dict(number=len(events), kind=kind, seconds=time.monotonic() - started, **details)
        require(row["seconds"] <= config["max_wall_seconds"], "finite wall budget exceeded")
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        if progress:
            progress(row)

    def emit(kind, key, lane, receipt):
        row = dict(number=len(emissions), kind=kind, case=key, lane=lane, receipt=receipt.to_dict())
        emissions.append(row)
        with (output / "receipt-emissions.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")

    event("source_setup", expected=declared)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "qualified CUDA admission required")
    modules = old["_helper_modules"]({key: value for key, value in config.items() if key != "worker_sha256"})
    qw, graph, decoder_graph = (modules[old[key]] for key in ("QWEN_WORKER", "DENOISER_HELPER", "DECODER_HELPER"))
    runtime = qw["_runtime_contract"](config, torch)
    runtime["native_diffusers_sources"] = old["_native_source_contract"](config)["sources"]
    save(output / "runtime-contract.json", runtime)
    sources = {}
    for key in old["_source_keys"]():
        path = Path(config[key])
        target = output / "source" / key / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[key] = dict(path=str(target.relative_to(output)), sha256=sha(target), bytes=target.stat().st_size)
    manifest = copy.deepcopy(read(config["flux_manifest"]))
    for entry in read(config["vae_manifest"])["artifacts"]:
        entry = copy.deepcopy(entry)
        entry["path"] = "vae-" + entry["path"]
        entry["source"]["file"] = "vae/" + entry["source"]["file"]
        manifest["artifacts"].append(entry)
    manifest["defaults"] = dict(artifact_root="parameter-cache", checkpoint_root=config["flux_parameter_root"], ssh_host="beast", python="python3")
    manifest["provenance"] = dict(schema=SCHEMA, serving_parameters=311, scope="169 transformer and142 decoder/BN original recipes; full VAE identity; no payload exports")
    runpy.run_path(config["flux_retrieval_script"])["validate_manifest"](manifest)
    save(output / "parameter-artifacts.json", manifest)
    shutil.copyfile(config["flux_retrieval_script"], output / "retrieve_weights.py")
    flux, _, fm = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae = AutoencoderKLFlux2.from_pretrained(config["vae_parameter_root"], torch_dtype=torch.bfloat16,
                                           local_files_only=True).eval().requires_grad_(False)
    vm = read(config["vae_manifest"])
    require(len(vm["artifacts"]) == 142, "decoder recipe set differs")
    models, prefixes = dict(flux=flux, vae=vae), dict(flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
    identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    full_vae_before = old["_component_signature"](vae)
    prior = source_helper["Evidence"](config["source"])
    original_identities = prior.read("source.json")["parameter_identities"]
    for name in models:
        old["_assert_signature_equal"](original_identities[name], identities[name], name)
    graph["validate_model"](flux)
    counts["component_hydrations"] = 2

    def place(name, device):
        model = models[name]
        if name == "vae":
            model.post_quant_conv.to(device)
            model.decoder.to(device)
        else:
            model.to(device)
        signature = old["_component_signature"](model, prefixes=prefixes[name])
        old["_assert_signature_equal"](identities[name], signature, name)
        placements.append(dict(component=name, device=device, signature=signature))
        counts["cuda_placements" if device == "cuda" else "cpu_offloads"] += 1
        event("placement", component=name, device=device)
        if device == "cpu":
            torch.cuda.empty_cache()

    store = LocalStore(output / "state-store")

    def retain(cut):
        store.save(cut)
        return base["cut_record"](cut)

    def component(adapter, state, parent):
        return retain(StateCut(adapter.model_identity, {**dict(adapter.execution), "composition": SCHEMA,
            "provenance": "component equations, not a native advance"}, adapter.boundary(state), parent,
            state, adapter.point(state).to_dict(), adapter.surface(state).to_dict()))

    descriptors = describe(flux.state_dict())
    adapter = fresh_binding_adapter(flux, vae, config, lifecycle, device="cpu")
    binding, source_table = source_helper["prepare_sources"](config, output, store, adapter, base, old)
    require(describe(flux.state_dict()) == descriptors, "loader reconstruction changed parameter descriptors")
    save(output / "source-identity-qualification.json", dict(schema="mdb-historical-loader-identity-v1",
        family="binding", transformer_config=dict(flux.config), model_identity=adapter.model_identity,
        historical_binding_cuts_matched=True, parameter_content_and_layout_unchanged=True,
        complete_parameter_descriptors=descriptors, model_identity_override=False, learned_calls=0, cuda_placements=0))
    del adapter
    roots, payloads = {}, {}
    for cid, info in binding.items():
        metadata = info["metadata"]
        roots["binding-" + cid] = dict(context=cid, arms=list(ARMS), source=retain(info["qcut"]), transitions=54,
            baseline_arm="whole_Q", support=metadata["support"], base_conditioning=metadata["base_conditioning"])
        payloads["binding-" + cid] = metadata["text_payloads"]
    save(output / "source-frontiers.json", dict(scope=SCOPE, roots=roots))
    save(output / "payload-index.json", payloads)
    event("all_source_states_ready", roots=list(roots), no_new_model_calls=True)
    processor = Flux2ImageProcessor(vae_scale_factor=2**(len(vae.config.block_out_channels) - 1)*2)
    paths, ports_by_lane = {}, {"forecast": {}, "native": {}}

    def patch(session, slot, record, key, lane, purpose):
        before, after, receipt = base["apply_slot_edit"](session, slot, base["load_tensor"](output, record, "cuda"), mode="replace")
        unchanged = describe({name: value for name, value in before.payload.items() if name != slot}) == describe({name: value for name, value in after.payload.items() if name != slot})
        require(unchanged, "Act changed undeclared slots")
        emit("act", key, lane, receipt)
        field = "acts_per_lane_" + lane
        counts[field] = counts.get(field, 0) + 1
        return dict(slot=slot, mode="replace", purpose=purpose, payload=record, before=retain(before), after=retain(after),
                    receipt=receipt.to_dict(), all_other_slots_unchanged=unchanged,
                    displacement=base["displacement_account"](before.payload[slot], after.payload[slot]))

    def run_paths(lane):
        active = fresh_binding_adapter(flux, vae, config, lifecycle, device="cuda")
        for rid, root in roots.items():
            parent = store.load(root["source"]["cut"], device="cuda")
            master = Session(active, parent.payload, parent=parent.parent) if lane == "native" else None
            if master is not None:
                require(master.capture(retain=False).fingerprint == parent.fingerprint, "native source closure differs")
            for arm in ARMS:
                key = rid + "/" + arm
                branch = output / "cases" / rid / arm
                branch.mkdir(parents=True, exist_ok=True)
                if lane == "native":
                    session = master.fork(parent)
                    save(branch / "native-fork.json", dict(common=root["source"], fork=retain(session.capture(retain=False))))
                else:
                    start = component(active, parent.payload, parent.fingerprint)
                    session = Session(active, parent.payload, parent=start["cut"])
                acts = [patch(session, "text", payloads[rid][arm], key, lane, "source-residual-write-endpoint")]
                state = session.capture(retain=False).payload
                start = retain(session.capture(retain=False))
                rows = [dict(number=0, observed_source=True, retained=start, fields=base["tensor_tree"](state, old))]
                previous_cut, ports = start["cut"], {}
                for number in range(1, 55):
                    clock = {name: state[name] for name in ("step", "phase", "cursor")}
                    reset = None
                    if clock == dict(step=3, phase="project", cursor=0):
                        if lane == "forecast":
                            session = Session(active, state, parent=previous_cut)
                        reset = patch(session, "conditioning", root["base_conditioning"], key, lane, "base-conditioning-project3-reset")
                        acts.append(reset)
                        state = session.capture(retain=False).payload
                        previous_cut = reset["after"]["cut"]
                    receipt = None
                    if lane == "forecast":
                        if number == 1:
                            state, values = common["consumer_equations"](graph, flux, state)
                            ports = {name: old["_save_tensor"](output, f"cases/{rid}/{arm}/first-consumer/forecast/{name}.npy", value) for name, value in values.items()}
                        else:
                            state = base["step_equations"](graph, flux, state)
                        counts["functional_transitions"] += 1
                    else:
                        if number == 1:
                            def port(name, value):
                                ports[name] = old["_save_tensor"](output, f"cases/{rid}/{arm}/first-consumer/native/{name}.npy", value)
                            port("gate_ff", graph["_mods"](state["context"]["img_mod"], 2)[1][2])
                            with common["passive_consumer_ports"](flux, port):
                                result = session.continue_()
                        else:
                            result = session.continue_()
                        emit("native", key, lane, result)
                        counts["native_target_transitions"] += 1
                        state, receipt = session.capture(retain=False).payload, result.fingerprint
                    row = dict(number=number, event=dict(step=clock["step"], phase=clock["phase"], index=clock["cursor"]), fields=base["tensor_tree"](state, old))
                    if receipt is not None:
                        row["receipt"] = receipt
                    if reset is not None:
                        row.update(before_project3_reset=reset["before"], after_project3_reset=reset["after"])
                    if lifecycle["selected_landmark"](clock, number, "binding"):
                        row["retained"] = component(active, state, previous_cut) if lane == "forecast" else retain(session.capture(retain=False))
                        previous_cut = row["retained"]["cut"]
                    rows.append(row)
                require(state["step"] == 4 and state["phase"] == "project" and state["cursor"] == 0, "suffix did not close")
                require(set(ports) == {"image_ln", "text_ln", "R_attention", "x", "M_base", "gate_ff"}, "actual first-consumer port set differs")
                save(branch / (lane + "-rows.json"), rows)
                save(branch / (lane + "-acts.json"), acts)
                ports_by_lane[lane][key] = ports
                for name, increment in (("first_consumer_port_sets_per_lane_", 1), ("first_consumer_ports_per_lane_", len(ports))):
                    counts[name + lane] = counts.get(name + lane, 0) + increment
                if lane == "forecast":
                    paths[key] = dict(root=str(branch.relative_to(output)), context=root["context"], arm=arm,
                        baseline_case=rid + "/whole_Q", source_cut=root["source"]["cut"], forecast_final=previous_cut,
                        known_reference=binding[root["context"]]["metadata"]["known_endpoints"].get(arm))
                else:
                    paths[key]["native_final"] = previous_cut
                    if arm == "whole_Q":
                        restored = session.restore(parent)
                        emit("restore", key, lane, restored)
                        require(session.capture(retain=False).fingerprint == parent.fingerprint, "complete parent rewind differs")
                        save(branch / "native-rewind.json", dict(after=retain(session.capture(retain=False)), receipt=restored.to_dict()))
                        counts["native_source_restores"] += 1
                event(lane + "_suffix_completed", case=key, final_cut=previous_cut)

    def decode_paths(lane):
        for path in paths.values():
            branch = output / path["root"]
            final = store.load(path[lane + "_final"], device="cpu").payload
            argument = old["_decoder_input"](vae, final, config, {"Flux2KleinPipeline": Flux2KleinPipeline}).to("cuda")
            if lane == "forecast":
                decoded = decoder_graph["functional_decode"](vae, argument, observe=None, counts={}, profile="original")
                record = base["decoder_result"](old, branch, "forecast", decoded, processor)
                record["decoder_input"] = old["_save_tensor"](branch, "decoder/forecast/decoder-input.npy", argument)
                counts["functional_vae_calls"] += 1
                save(branch / "decoder-forecast.json", record)
            else:
                record = old["_native_decoder"](branch, vae, final, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, {}, {})
                counts["native_vae_reference_calls"] += 1
                save(branch / "decoder-native-reference.json", record)

    def signed_accounts(lane):
        responses, interactions = {}, {}
        decoder_name = "decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json"
        for key, path in paths.items():
            if key == path["baseline_case"]:
                continue
            baseline = paths[path["baseline_case"]]
            a, b = (read(output / item["root"] / decoder_name) for item in (path, baseline))
            responses[key] = dict(orientation="arm-minus-own-baseline", decoder={name: base["signed_fields"](
                base["load_tensor"](output / path["root"], a[name], "cpu"), base["load_tensor"](output / baseline["root"], b[name], "cpu"))["/"] for name in ("decoded", "pixels")},
                first_consumer={name: base["signed_fields"](base["load_tensor"](output, record, "cpu"),
                    base["load_tensor"](output, ports_by_lane[lane][path["baseline_case"]][name], "cpu"))["/"] for name, record in ports_by_lane[lane][key].items()})
        for rid in roots:
            order = ("both_RW", "residual_only", "write_only", "whole_Q")
            keys = [rid + "/" + arm for arm in order]
            arrays = {}
            for field in ("decoded", "pixels"):
                values = [base["load_tensor"](output / paths[key]["root"], read(output / paths[key]["root"] / decoder_name)[field], "cpu").double() for key in keys]
                arrays[field] = old["_save_tensor"](output, f"interactions/{lane}/{rid}/{field}.npy", (values[0] - values[1]) - (values[2] - values[3]))
            for field in ports_by_lane[lane][keys[0]]:
                values = [base["load_tensor"](output, ports_by_lane[lane][key][field], "cpu").double() for key in keys]
                arrays["port/" + field] = old["_save_tensor"](output, f"interactions/{lane}/{rid}/port-{field}.npy", (values[0] - values[1]) - (values[2] - values[3]))
            entries = [store.load(read(output / paths[key]["root"] / (lane + "-rows.json"))[0]["retained"]["cut"], device="cpu").payload["text"].double() for key in keys]
            arrays["entry"] = old["_save_tensor"](output, f"interactions/{lane}/{rid}/entry.npy", (entries[0] - entries[1]) - (entries[2] - entries[3]))
            interactions[rid] = dict(orientation=order, arrays=arrays, scope="Entry cast remainder and complete finite downstream interaction, RW-R-W+Q")
        save(output / (lane + "-responses.json"), responses)
        save(output / (lane + "-interactions.json"), interactions)

    def entered(_module, _args):
        require(phase["kind"] == "target-denoiser" and released["native"], "native block before global release")
        counts["native_blocks"] += 1

    handles = [block.register_forward_pre_hook(entered) for block in (*flux.transformer_blocks, *flux.single_transformer_blocks)]
    try:
        with base["instrument"](list(models.values()), phase, released, observed), torch.inference_mode():
            place("flux", "cuda")
            phase["kind"] = "forecast-denoiser"
            run_paths("forecast")
            save(output / "first-consumer-forecasts.json", ports_by_lane["forecast"])
            place("flux", "cpu")
            place("vae", "cuda")
            phase["kind"] = "forecast-vae"
            decode_paths("forecast")
            signed_accounts("forecast")
            place("vae", "cpu")
            inventory = old["_inventory"](output, exclude=("timeline.jsonl", "receipt-emissions.jsonl"))
            tx = ForecastTransaction(output / "forecast-transaction", model_identity=SCHEMA,
                input_identity=hashlib.sha256(canonical(roots)).hexdigest(), source_identity=hashlib.sha256(canonical(config["_request_record"]["source_hashes"])).hexdigest(),
                frontier=dict(SCOPE, external_files=inventory), execution_scope=SCOPE["scope"], array_storage="files-v1",
                budgets=dict(max_predictions=1, max_executions=1, max_native_calls=512, max_files=15000,
                             max_bytes=config["forecast_transaction_bytes"], max_wall_ms=config["max_wall_seconds"] * 1000))
            tx.frontier("observed-parents", dict(roots=roots, source=config["source"], scope=SCOPE))
            tx.predict("all-futures", lambda _ctx: dict(predictions=dict(external_files=inventory, paths=paths,
                targets_at_seal=0, hypotheses=config["hypotheses"])))
            seal = tx.seal()
            event("all_forecasts_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets_at_seal=counts["native_target_transitions"], native_blocks_at_seal=counts["native_blocks"])

            def targets(_context):
                old["_verify_inventory"](output, inventory)
                released["native"] = True
                event("native_targets_released", seal_sha256=seal["seal_sha256"])
                place("flux", "cuda")
                phase["kind"] = "target-denoiser"
                run_paths("native")
                save(output / "first-consumer-native.json", ports_by_lane["native"])
                place("flux", "cpu")
                place("vae", "cuda")
                phase["kind"] = "target-vae"
                decode_paths("native")
                signed_accounts("native")
                place("vae", "cpu")
                return dict(counts=counts, observed=observed, target_paths=len(paths))

            tx.execute("original-native-targets", targets)
            tx.close()
    finally:
        for handle in reversed(handles):
            handle.remove()
    final = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    full_vae_after = old["_component_signature"](vae)
    old["_assert_signature_equal"](full_vae_before, full_vae_after, "full VAE")
    for name in models:
        old["_assert_signature_equal"](identities[name], final[name], name)
    old["_verify_inventory"](output, inventory)
    for key, expected in declared.items():
        if "per_lane" in key:
            for lane in ("forecast", "native"):
                require(counts[key + "_" + lane] == expected, "actual mechanics census differs: " + key + lane)
        elif key != "max_gpu_components":
            require(counts[key] == expected, "actual mechanics census differs: " + key)
    save(output / "state-index.json", paths)
    save(output / "native-contract.json", dict(counts=counts, observed=observed, placements=placements,
        identities_before=identities, identities_after=final, full_vae_identity_before=full_vae_before,
        full_vae_identity_after=full_vae_after, conditional_scope=SCOPE))
    save(output / "source.json", dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"],
        sources=sources, aliases=dict(source=config["source"]), source_provenance=source_table["provenance"],
        parameter_identities=final, parameter_recipe_counts=dict(flux=len(fm["artifacts"]), vae=len(vm["artifacts"])),
        forecast_seal=seal["seal_sha256"], no_persistent_parameter_bank=True))
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], counts=counts, contract=declared,
        observed=observed, forecast_seal=seal["seal_sha256"], transaction=verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="completed", instrument_status="full functional forecasts and native continuations",
        trend_status="uninterpreted observations retained", terminal_status="unassessed", seconds=time.monotonic() - started)
    save(output / "summary.json", result)
    return result


def worker(config_path):
    import faulthandler

    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path)
    validate_config(config)
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    identity = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "request/config changed")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "executed source changed: " + name)
    job = os.environ["MRUN_JOB_ID"]
    require(job.startswith("job-") and "/" not in job, "invalid job identity")
    derived = dict(config, job_id=job, request_sha256=identity, _request_record=request,
                   output=str(Path(config["output_scratch"]) / job / RESULT))
    result = run(derived, progress=lambda row: print("MDB_BINDING_RESIDUAL_WRITE_PHASE " + canonical(row).decode(), flush=True))
    ready = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))["finalize"](
        derived["output"], job_id=result["job_id"], request_sha256=identity, **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    faulthandler.cancel_dump_traceback_later()
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
