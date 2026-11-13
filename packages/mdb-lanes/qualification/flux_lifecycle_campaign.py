"""Sealed binding-carrier forecasts and fixed-payload receiver contrasts.

Observed states and original parameters are declared inputs. No new encoder,
noise, priming prefix, or native target is permitted before the global seal.
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
SCHEMA = "mdb-flux-lifecycle-v1"
RESULT = "flux-lifecycle-result"
COMMON = "flux_common_differential_campaign.py"
COMMON_SHA256 = "8b399ee217e75588a7949f8d1efd955cb0728e6daa1f4484bf281e22172afd3d"
CD_CONTEXTS = ("fresh0s613", "foxcat0s821")
CD_ARMS = ("baseline", "sham", "common-only", "differential-only", "both")
BINDING_ARMS = ("whole_Q", "support_P", "complement_P", "whole_P")
SOURCE_CD = dict(
    root="/mnt/big/mdb/research/job-14809ca3fbdc/flux-common-differential-result",
    manifest_sha256="4e986376dcb826a0418e62eeba26dc129ee52cc1141bbecf9caefe3e10775436",
    request_sha256="716052fccbb86872ee7cdc41ffa9b6fbb845e44b25047ad21101dea2bb0c088e",
    job_id="job-14809ca3fbdc",
)
SCOPE = dict(
    kind="architectural", observed="Saved paired binding states and cone/fox joint0 exits; inherited 613 payloads",
    raw_prompt_forecast=False, source_writer_forecast=False, independent_kernel_prediction=False,
    scope="Own-state functional denoiser/scheduler/VAE forecasts using original shared primitive kernels; all new native targets withheld until seal",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def common_helpers():
    require(sha(HERE / COMMON) == COMMON_SHA256, "qualified common helper changed")
    common = runpy.run_path(str(HERE / COMMON))
    base = common["previous"]()
    return common, base, base["helpers"]()


def contract():
    return dict(
        contexts=6, image_futures=26, known_replay_conditions=15, new_modified_conditions=11,
        functional_transitions=1964, native_target_transitions=1964,
        native_blocks=1774, functional_vae_calls=26, native_vae_reference_calls=26,
        acts_per_lane=40, first_consumer_port_sets_per_lane=26, native_source_restores=6,
        component_hydrations=2, cuda_placements=4, cpu_offloads=4,
        native_qwen_source_calls=0, native_project_source_transitions=0,
        native_joint0_source_transitions=0, noise_draws=0, functional_module_forwards=0,
        persistent_weight_banks=0, max_gpu_components=1,
    )


def make_config(parent, binding_plan):
    config = {key: value for key, value in parent.items() if key not in (
        "job_id", "request_sha256", "worker_sha256", "output", "_request_record",
        "new_context", "new_selector_context", "context_ids", "arms", "source_frontier",
    )}
    config.update(
        schema=SCHEMA, output_name=RESULT, cd_source=dict(SOURCE_CD), cd_contexts=list(CD_CONTEXTS),
        cd_arms=list(CD_ARMS), binding_arms=list(BINDING_ARMS), binding_plan=binding_plan,
        binding_source_root="/mnt/big/mdb/research/job-c557d492df0c/recovered-native",
        binding_source_manifest_sha256="afd8ac5c32ae0cfcafc0caeccffbb6fee99427bccaf10334620e92fc2ea3953f",
        binding_source_request_sha256="5cbe9ee947692b0f262acb2017f82668c23e6a76d9d98ced133dacdc07a27e26",
        binding_route_root="/mnt/big/mdb/research/job-3d4bdd4210d8/binding-minimal-result",
        binding_route_manifest_sha256="20ba3a1fc8298e30177f1c42cec8df38b3455c6983d8043bcbe077937f28a2d3",
        binding_route_request_sha256="4baa6bd41c8cadd32a541088f9c2f57e94cc4f9fffabe811ff5e834d7c4bd714",
        source_frontier=dict(SCOPE), component_hydrations=2, cuda_placements=4, offloads=4,
        max_wall_seconds=1500, forecast_transaction_bytes=6 << 30,
        transport_limits=dict(max_expanded_bytes=12 << 30, max_member_bytes=134221824),
        scientific_gates=[], cpu_threads=4,
        hypotheses=dict(
            binding="Support-only, complement-only, both, and neither transferring are retained alternatives; source norm does not select a winner.",
            binding_controls="Whole Q retains base assignment; whole P follows historical I.Q_T.P, including partial paraphrase.",
            fixed_payload="Identical supplied C/D/B bytes isolate source-payload changes; actual rounded ADD displacements may still differ by receiver.",
            numerical="Every new full own-state future and image is forecast before native targets under the pinned primitive profile.",
        ),
        decoder_contract="26 functional full-image forecasts,26 actual native-final decodes; known endpoints and own shams supply replay controls",
        endpoint_contract="Binding exact BF16 row SET; C/D identical inherited613 C/D/once-rounded B ADD into each own receiver",
        first_consumer="Original joint1 at each root clock; image/text normalization and image FF operands/return plus full next state",
        payload_rule="No current C/D coefficient recomputation; binding support/complement are exact source-row selection",
        finite_factorial="Binding entry parallelogram is zero; C/D entry remainder is retained; later full signed tensors include finite consumer arithmetic",
        direction_geometry="Inherited613 payload field unchanged on cone613 and fox821; no new directions, scaling, or eigensearch",
        retention="Full source/Act/first-consumer/commit/carrier-landmark states, all boundary hashes, complete pixels and signed interactions; no pretrained parameter banks",
    )
    return config


def validate_config(config):
    require(config.get("schema") == SCHEMA and config.get("cd_source") == SOURCE_CD, "source identity changed")
    require(config.get("cd_contexts") == list(CD_CONTEXTS) and config.get("cd_arms") == list(CD_ARMS)
            and config.get("binding_arms") == list(BINDING_ARMS), "finite panel changed")
    plan = config["binding_plan"]
    require(plan["schema"] == "mdb-binding-support-complement-next-cut-v1"
            and set(plan["cases"]) == {"anchor", "seed", "wording", "objects"}, "binding source plan differs")
    require(config.get("height") == config.get("width") == 512 and config.get("steps") == 4
            and config.get("batch_size") == 1 and config.get("device") == "cuda"
            and config.get("dtype") == "bfloat16", "qualified geometry changed")
    require(not config.get("retry") and not config.get("nested_mrun") and not config.get("reference_image"), "finite single lease required")
    _, base, old = common_helpers()
    for key in old["_source_keys"]():
        require(config.get(key) and sha(config[key]) == config.get(key + "_sha256"), "metadata changed: " + key)
    return contract()


def selected_landmark(clock, number, family):
    return (number == 1 or clock["phase"] in ("readout", "scheduler")
            or (family == "binding" and clock["step"] == 3
                and (clock["phase"] == "project" or (clock["phase"] == "joint" and clock["cursor"] == 0))))


def authenticated_cd_source(config):
    source = config["cd_source"]
    root = Path(source["root"])
    manifest_path = root / "files.json"
    require(sha(manifest_path) == source["manifest_sha256"], "C/D source manifest changed")
    manifest = read(manifest_path)
    require(manifest["job_id"] == source["job_id"] and manifest["request_sha256"] == source["request_sha256"], "C/D source custody differs")
    consumed = {}

    def member(name):
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.resolve().is_relative_to(root), "escaped C/D source")
        row = manifest["files"][name]
        require(not path.is_symlink() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "C/D source member changed: " + name)
        consumed[name] = row
        return path

    request = read(member("request.json"))
    require(request["request_sha256"] == source["request_sha256"] and request["source_hashes"].get(COMMON) == COMMON_SHA256, "C/D executed source differs")
    return root, member, consumed


def transformer_profiles(config, current):
    """Reconstruct both truthful historical Diffusers loader configurations."""
    cd = copy.deepcopy(dict(current))
    binding = copy.deepcopy(cd)
    binding["_name_or_path"] = str(Path(config["flux_parameter_root"]) / "transformer")
    require(cd.get("_name_or_path") == "/raid/yiyi/klein-4b-distilled-diffusers/transformer",
            "qualified recipe loader path differs")
    require([key for key in cd if cd[key] != binding[key]] == ["_name_or_path"],
            "historical loader reconstruction changed more than its path")
    return {"fixed-cd": cd, "binding": binding}


def run(config, *, progress=None):
    declared = validate_config(config)
    common, base, old = common_helpers()
    import torch
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.core import Session, StateCut
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {k: v for k, v in config.items() if k != "_request_record"})
    save(output / "request.json", config["_request_record"])
    started = time.monotonic()
    counts = {key: 0 for key in declared if "per_lane" not in key and key != "max_gpu_components"}
    counts.update(contexts=6, image_futures=26, known_replay_conditions=15, new_modified_conditions=11)
    phase, released, observed = dict(kind="source-loading"), dict(native=False), {}
    events, emissions, placements = [], [], []

    def event(kind, **details):
        row = dict(number=len(events), kind=kind, seconds=time.monotonic()-started, **details)
        require(row["seconds"] <= config["max_wall_seconds"], "finite wall budget exceeded")
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        if progress:
            progress(row)

    def emit(kind, key, lane, value):
        row = dict(number=len(emissions), kind=kind, case=key, lane=lane, receipt=value.to_dict())
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
    modules = old["_helper_modules"]({k: v for k, v in config.items() if k != "worker_sha256"})
    qw, graph, decoder_graph = (modules[old[k]] for k in ("QWEN_WORKER", "DENOISER_HELPER", "DECODER_HELPER"))
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
    # Reuse the qualified stdlib retriever; keep constants in their checkpoint.
    # The combined manifest has one common checkpoint root and distinct output
    # namespaces, while retaining every original named tensor/header/byte hash.
    parameter_manifest = read(config["flux_manifest"])
    parameter_manifest = copy.deepcopy(parameter_manifest)
    for entry in read(config["vae_manifest"])["artifacts"]:
        entry = copy.deepcopy(entry)
        entry["path"] = "vae-" + entry["path"]
        entry["source"]["file"] = "vae/" + entry["source"]["file"]
        parameter_manifest["artifacts"].append(entry)
    parameter_manifest["defaults"] = dict(artifact_root="parameter-cache", checkpoint_root=config["flux_parameter_root"], ssh_host="beast", python="python3")
    parameter_manifest["provenance"] = dict(schema=SCHEMA, flux_manifest_sha256=config["flux_manifest_sha256"],
        vae_manifest_sha256=config["vae_manifest_sha256"], serving_parameters=311,
        scope="169 transformer and142 decoder/BN original recipes; complete VAE checkpoint also bound for historical adapter identity; no parameter payload exported")
    retriever = runpy.run_path(config["flux_retrieval_script"])
    retriever["validate_manifest"](parameter_manifest)
    save(output / "parameter-artifacts.json", parameter_manifest)
    shutil.copyfile(config["flux_retrieval_script"], output / "retrieve_weights.py")
    recipe = runpy.run_path(config["flux_recipe_helper"])
    flux, _, fm = old["_load_transformer"](config, recipe)
    # Binding's historical adapter includes the complete native VAE identity.
    # Hydrate that same original model, while serving only its decoder subset.
    vae = AutoencoderKLFlux2.from_pretrained(config["vae_parameter_root"],
        torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False)
    vm = read(config["vae_manifest"])
    require(len(vm["artifacts"]) == 142, "original decoder recipe set differs")
    models = dict(flux=flux, vae=vae)
    prefixes = dict(flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
    identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    full_vae_identity_before = old["_component_signature"](vae)
    cd_root, member, cd_consumed = authenticated_cd_source(config)
    original_identities = read(member("source.json"))["parameter_identities"]
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
        cut = StateCut(adapter.model_identity, {**dict(adapter.execution), "composition": SCHEMA,
            "provenance": "component equations, not a native advance"}, adapter.boundary(state), parent,
            state, adapter.point(state).to_dict(), adapter.surface(state).to_dict())
        return retain(cut)

    profiles = transformer_profiles(config, flux.config)
    profile_events = []

    def fresh_adapter(family, device):
        # Adapters seal configuration as well as parameters. Never keep one
        # active across this loader-equivalent configuration reconstruction.
        flux.register_to_config(**profiles[family])
        require(dict(flux.config) == profiles[family], "historical loader profile differs")
        if family == "fixed-cd":
            result = common["fresh_adapter"](flux, read(config["flux_scheduler_config"]), device=device)
        else:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
            pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, transformer=flux,
                text_encoder=None, tokenizer=None, is_distilled=True)
            result = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", device=device)
        profile_events.append(dict(family=family, device=device, model_identity=result.model_identity,
            transformer_config=copy.deepcopy(dict(flux.config))))
        return result

    parameter_descriptors = describe(flux.state_dict())
    source_cd_adapter = fresh_adapter("fixed-cd", "cpu")
    source_cd_identity = source_cd_adapter.model_identity
    del source_cd_adapter
    adapter = fresh_adapter("binding", "cpu")
    helper = runpy.run_path(str(HERE / "flux_binding_support_sources.py"))
    binding = helper["prepare_binding_sources"](config, output, store, adapter)
    binding_records, roots, payloads = {}, {}, {}
    for cid, info in binding.items():
        rid = "binding-" + cid
        qcut, pcut = info["qcut"], info["pcut"]
        require(qcut.model_identity == adapter.model_identity == pcut.model_identity, "binding model identity differs")
        root = dict(family="binding", context=cid, arms=list(BINDING_ARMS), source=retain(qcut),
                    transitions=54, baseline_arm="whole_Q", support=info["support"],
                    base_conditioning=old["_save_tensor"](output, f"binding/{cid}/base-conditioning.npy", info["base_conditioning"]))
        roots[rid] = root
        payloads[rid] = {arm: old["_save_tensor"](output, f"binding/{cid}/payloads/{arm}.npy", value)
                         for arm, value in info["text_payloads"].items()}
        binding_records[cid] = dict(P=retain(pcut), Q=root["source"], support=info["support"],
            text_payloads=payloads[rid], base_conditioning=root["base_conditioning"],
            known_endpoints=info["known_endpoints"], witnesses=info["witnesses"],
            analysis=info["analysis"], provenance=info["provenance"])
        event("binding_source_account", context=cid, analysis=info["analysis"])
    save(output / "binding-sources.json", binding_records)
    del binding, adapter
    source_cd_adapter = fresh_adapter("fixed-cd", "cpu")
    require(source_cd_adapter.model_identity == source_cd_identity, "restored recipe adapter identity differs")
    require(describe(flux.state_dict()) == parameter_descriptors,
            "loader profile reconstruction changed parameter content/layout")
    cd_store = LocalStore(cd_root / "state-store")

    def cd_cut(fingerprint):
        meta = read(member("state-store/cuts/" + fingerprint + ".json"))
        for page in meta["slots"].values():
            member("state-store/pages/" + page + ".json")
        for blob in meta["physical_blobs"]:
            member("state-store/blobs/" + blob + ".safetensors")
        cut = cd_store.load(fingerprint, device="cpu")
        retain(cut)
        return cut

    account = read(member("writer/fresh0s613/account.json"))
    fixed = {}
    for arm in CD_ARMS[2:]:
        record = account["payloads"][arm]
        member(record["path"])
        fixed[arm] = old["_save_tensor"](output, "fixed-payloads/" + arm + ".npy", base["load_tensor"](cd_root, record, "cpu"))
    save(output / "fixed-payloads.json", dict(source=dict(SOURCE_CD, context="fresh0s613"), payloads=fixed,
        numerical_scope="Identical inherited BF16 C/D/once-rounded B bytes; no coefficient recomputation, scaling, normalization or refit"))
    cd_known = {}
    source_index = read(member("state-index.json"))
    for cid in CD_CONTEXTS:
        rid = "cd-" + cid
        fork = read(member(f"cases/{cid}/baseline/native-fork.json"))
        cut = cd_cut(fork["common"]["cut"])
        require(cut.model_identity == source_cd_adapter.model_identity and cut.payload["step"] == 0
                and cut.payload["phase"] == "joint" and cut.payload["cursor"] == 1, "C/D source model/clock differs")
        roots[rid] = dict(family="fixed-cd", context=cid, arms=list(CD_ARMS), source=retain(cut), transitions=110, baseline_arm="baseline")
        payloads[rid] = fixed
        for arm in CD_ARMS:
            if cid != "fresh0s613" and arm not in ("baseline", "sham"):
                continue
            rel = f"cases/{cid}/{arm}"
            dec = read(member(rel + "/decoder-native-reference.json"))
            png = dec["png"]
            member(rel + "/" + png["path"])
            cd_known[rid + "/" + arm] = dict(source_root=str(cd_root), source_manifest_sha256=SOURCE_CD["manifest_sha256"],
                branch_root=rel, png=png, final_cut=source_index[cid + "/" + arm]["native_final"])
    save(output / "cd-source-members.json", cd_consumed)
    save(output / "source-frontiers.json", dict(scope=SCOPE, roots=roots))
    save(output / "payload-index.json", payloads)
    del source_cd_adapter
    save(output / "source-identity-qualification.json", dict(schema="mdb-historical-loader-identity-v1",
        profiles=profiles, transitions=profile_events[:], complete_parameter_descriptors=parameter_descriptors,
        parameter_content_and_layout_unchanged=True, historical_binding_cuts_matched=True,
        historical_cd_cuts_matched=True, restored_recipe_identity_matched=True,
        model_identity_override=False, learned_calls=0, cuda_placements=0))
    event("all_source_states_ready", roots=list(roots), no_new_model_calls=True)
    processor = Flux2ImageProcessor(vae_scale_factor=2**(len(vae.config.block_out_channels)-1)*2)
    paths, row_records, port_records = {}, {}, {"forecast": {}, "native": {}}

    def patch(session, slot, record, *, mode, key, lane, purpose):
        value = base["load_tensor"](output, record, "cuda")
        before, after, receipt = base["apply_slot_edit"](session, slot, value, mode=mode)
        unchanged = describe({k: v for k, v in before.payload.items() if k != slot}) == describe({k: v for k, v in after.payload.items() if k != slot})
        require(unchanged, "Act changed an undeclared slot")
        emit("act", key, lane, receipt)
        name = "acts_per_lane_" + lane
        counts[name] = counts.get(name, 0) + 1
        return dict(slot=slot, mode=mode, purpose=purpose, payload=record,
            before=retain(before), after=retain(after), receipt=receipt.to_dict(), all_other_slots_unchanged=unchanged,
            displacement=base["displacement_account"](before.payload[slot], after.payload[slot]))

    def run_family(lane, adapter, family):
        for rid, root in roots.items():
            if root["family"] != family:
                continue
            parent = store.load(root["source"]["cut"], device="cuda")
            master = None
            if lane == "native":
                master = Session(adapter, parent.payload, parent=parent.parent)
                require(master.capture(retain=False).fingerprint == parent.fingerprint, "native source session closure differs")
            for arm in root["arms"]:
                key = rid + "/" + arm
                branch_root = output / "cases" / rid / arm
                branch_root.mkdir(parents=True, exist_ok=True)
                if lane == "native":
                    session = master.fork(parent)
                    start = retain(session.capture(retain=False))
                    save(branch_root / "native-fork.json", dict(common=root["source"], fork=start))
                else:
                    start = component(adapter, parent.payload, parent.fingerprint)
                    session = Session(adapter, parent.payload, parent=start["cut"])
                acts = []
                if root["family"] == "binding":
                    acts.append(patch(session, "text", payloads[rid][arm], mode="replace", key=key, lane=lane, purpose="entry-support-partition"))
                elif arm != "baseline":
                    if arm == "sham":
                        ref = old["_save_tensor"](output, f"cases/{rid}/{arm}/{lane}-sham.npy", session.read("image"))
                    else:
                        ref = fixed[arm]
                    acts.append(patch(session, "image", ref, mode="replace" if arm == "sham" else "add", key=key, lane=lane, purpose="fixed-payload-entry"))
                state = session.capture(retain=False).payload
                start = retain(session.capture(retain=False)) if acts or lane == "native" else start
                rows = [dict(number=0, observed_source=True, retained=start, fields=base["tensor_tree"](state, old))]
                previous_cut = start["cut"]
                ports = {}
                for number in range(1, root["transitions"] + 1):
                    clock = {k: state[k] for k in ("step", "phase", "cursor")}
                    reset = None
                    if root["family"] == "binding" and clock == dict(step=3, phase="project", cursor=0):
                        if lane == "forecast":
                            session = Session(adapter, state, parent=previous_cut)
                        reset = patch(session, "conditioning", root["base_conditioning"], mode="replace", key=key, lane=lane, purpose="base-conditioning-project3-reset")
                        acts.append(reset)
                        state = session.capture(retain=False).payload
                        previous_cut = reset["after"]["cut"]
                    if lane == "forecast":
                        if number == 1:
                            state, raw_ports = common["consumer_equations"](graph, flux, state)
                            ports = {name: old["_save_tensor"](output, f"cases/{rid}/{arm}/first-consumer/forecast/{name}.npy", value) for name, value in raw_ports.items()}
                        else:
                            state = base["step_equations"](graph, flux, state)
                        counts["functional_transitions"] += 1
                        receipt = None
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
                        state = session.capture(retain=False).payload
                        receipt = result.fingerprint
                    row = dict(number=number, event=dict(step=clock["step"], phase=clock["phase"], index=clock["cursor"]), fields=base["tensor_tree"](state, old))
                    if receipt is not None:
                        row["receipt"] = receipt
                    if reset is not None:
                        row["before_project3_reset"] = reset["before"]
                        row["after_project3_reset"] = reset["after"]
                    if selected_landmark(clock, number, root["family"]):
                        row["retained"] = component(adapter, state, previous_cut) if lane == "forecast" else retain(session.capture(retain=False))
                        previous_cut = row["retained"]["cut"]
                    rows.append(row)
                require(state["step"] == 4 and state["phase"] == "project" and state["cursor"] == 0, "suffix did not close")
                save(branch_root / (lane + "-rows.json"), rows)
                save(branch_root / (lane + "-acts.json"), acts)
                port_records[lane][key] = ports
                name = "first_consumer_port_sets_per_lane_" + lane
                counts[name] = counts.get(name, 0) + 1
                if lane == "forecast":
                    known = (binding_records[root["context"]]["known_endpoints"].get(arm)
                             if root["family"] == "binding" else cd_known.get(key))
                    paths[key] = dict(root=str(branch_root.relative_to(output)), family=root["family"], context=root["context"], arm=arm,
                        baseline_case=rid + "/" + root["baseline_arm"], source_cut=root["source"]["cut"],
                        forecast_final=previous_cut, known_reference=known)
                    row_records[key] = rows
                else:
                    paths[key]["native_final"] = previous_cut
                    if arm == root["baseline_arm"]:
                        result = session.restore(parent)
                        emit("restore", key, lane, result)
                        require(session.capture(retain=False).fingerprint == parent.fingerprint, "complete parent rewind differs")
                        save(branch_root / "native-rewind.json", dict(after=retain(session.capture(retain=False)), receipt=result.to_dict()))
                        counts["native_source_restores"] += 1
                event(lane + "_suffix_completed", case=key, final_cut=previous_cut)

    def run_paths(lane):
        for family in ("binding", "fixed-cd"):
            adapter = fresh_adapter(family, "cuda")
            run_family(lane, adapter, family)
            del adapter

    def decode_paths(lane):
        for key, path in paths.items():
            branch_root = output / path["root"]
            final = store.load(path[lane + "_final"], device="cpu").payload
            argument = old["_decoder_input"](vae, final, config, {"Flux2KleinPipeline": Flux2KleinPipeline}).to("cuda")
            if lane == "forecast":
                decoded = decoder_graph["functional_decode"](vae, argument, observe=None, counts={}, profile="original")
                record = base["decoder_result"](old, branch_root, "forecast", decoded, processor)
                record["decoder_input"] = old["_save_tensor"](branch_root, "decoder/forecast/decoder-input.npy", argument)
                save(branch_root / "decoder-forecast.json", record)
                counts["functional_vae_calls"] += 1
            else:
                record = old["_native_decoder"](branch_root, vae, final, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, {}, {})
                save(branch_root / "decoder-native-reference.json", record)
                counts["native_vae_reference_calls"] += 1

    def signed_accounts(lane):
        responses, interactions = {}, {}
        for key, path in paths.items():
            if key == path["baseline_case"]:
                continue
            baseline = paths[path["baseline_case"]]
            a, b = (read(output / p["root"] / ("decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json")) for p in (path, baseline))
            responses[key] = dict(orientation="arm-minus-own-baseline", decoder={name: base["signed_fields"](
                base["load_tensor"](output / path["root"], a[name], "cpu"), base["load_tensor"](output / baseline["root"], b[name], "cpu"))["/"] for name in ("decoded", "pixels")},
                first_consumer={name: base["signed_fields"](base["load_tensor"](output, ref, "cpu"), base["load_tensor"](output, port_records[lane][path["baseline_case"]][name], "cpu"))["/"] for name, ref in port_records[lane][key].items()})
        for rid, root in roots.items():
            arms = ("whole_P", "support_P", "complement_P", "whole_Q") if root["family"] == "binding" else ("both", "common-only", "differential-only", "baseline")
            keys = [rid + "/" + arm for arm in arms]
            items = {}
            for field in ("decoded", "pixels"):
                arrays = []
                for key in keys:
                    p = paths[key]
                    record = read(output / p["root"] / ("decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json"))[field]
                    arrays.append(base["load_tensor"](output / p["root"], record, "cpu").double())
                value = arrays[0] - arrays[1] - arrays[2] + arrays[3]
                items[field] = old["_save_tensor"](output, f"interactions/{lane}/{rid}/{field}.npy", value)
            for field in port_records[lane][keys[0]]:
                arrays = [base["load_tensor"](output, port_records[lane][key][field], "cpu").double() for key in keys]
                value = arrays[0] - arrays[1] - arrays[2] + arrays[3]
                items["port/" + field] = old["_save_tensor"](output, f"interactions/{lane}/{rid}/port-{field}.npy", value)
            entries = [store.load(read(output / paths[key]["root"] / (lane + "-rows.json"))[0]["retained"]["cut"], device="cpu").payload for key in keys]
            slot = "text" if root["family"] == "binding" else "image"
            value = entries[0][slot].double() - entries[1][slot].double() - entries[2][slot].double() + entries[3][slot].double()
            items["entry"] = old["_save_tensor"](output, f"interactions/{lane}/{rid}/entry.npy", value)
            interactions[rid] = dict(orientation=arms, family=root["family"], arrays=items,
                scope="Binding entry partition is exactly additive. C/D uses fixed once-rounded B and has a measured entry remainder. Later interactions include the complete finite consumer.")
        save(output / (lane + "-responses.json"), responses)
        save(output / (lane + "-interactions.json"), interactions)

    block_handles = []

    def entered(_module, _args):
        require(phase["kind"] == "target-denoiser" and released["native"], "native block before global release")
        counts["native_blocks"] += 1

    for block in (*flux.transformer_blocks, *flux.single_transformer_blocks):
        block_handles.append(block.register_forward_pre_hook(entered))
    try:
        with base["instrument"](list(models.values()), phase, released, observed), torch.inference_mode():
            place("flux", "cuda")
            phase["kind"] = "forecast-denoiser"
            run_paths("forecast")
            save(output / "first-consumer-forecasts.json", port_records["forecast"])
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
                budgets=dict(max_predictions=1, max_executions=1, max_native_calls=2200, max_files=30000,
                    max_bytes=config["forecast_transaction_bytes"], max_wall_ms=config["max_wall_seconds"]*1000))
            tx.frontier("observed-parents", dict(roots=roots, scope=SCOPE))
            tx.predict("all-futures", lambda _ctx: dict(predictions=dict(external_files=inventory, paths=paths,
                targets_at_seal=0, hypotheses=config["hypotheses"])))
            seal = tx.seal()
            event("all_forecasts_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets_at_seal=counts["native_target_transitions"], native_blocks_at_seal=counts["native_blocks"])

            def targets(_ctx):
                old["_verify_inventory"](output, inventory)
                released["native"] = True
                event("native_targets_released", seal_sha256=seal["seal_sha256"])
                place("flux", "cuda")
                phase["kind"] = "target-denoiser"
                run_paths("native")
                save(output / "first-consumer-native.json", port_records["native"])
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
        for handle in reversed(block_handles):
            handle.remove()
    final = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    full_vae_identity_after = old["_component_signature"](vae)
    old["_assert_signature_equal"](full_vae_identity_before, full_vae_identity_after, "binding full VAE")
    for name in models:
        old["_assert_signature_equal"](identities[name], final[name], name)
    old["_verify_inventory"](output, inventory)
    for key, expected in declared.items():
        if "per_lane" in key:
            for lane in ("forecast", "native"):
                require(counts[key + "_" + lane] == expected, "actual census differs: " + key + lane)
        elif key != "max_gpu_components":
            require(counts[key] == expected, "actual census differs: " + key)
    save(output / "state-index.json", paths)
    save(output / "native-contract.json", dict(counts=counts, observed=observed, placements=placements,
        loader_profile_transitions=profile_events,
        identities_before=identities, identities_after=final, full_vae_identity_before=full_vae_identity_before,
        full_vae_identity_after=full_vae_identity_after, conditional_scope=SCOPE))
    save(output / "source.json", dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"],
        sources=sources, parameter_identities=final, parameter_recipe_counts=dict(flux=len(fm["artifacts"]), vae=len(vm["artifacts"])),
        aliases=dict(cd=SOURCE_CD, binding=dict(root=config["binding_source_root"], manifest_sha256=config["binding_source_manifest_sha256"], request_sha256=config["binding_source_request_sha256"]),
            route=dict(root=config["binding_route_root"], manifest_sha256=config["binding_route_manifest_sha256"], request_sha256=config["binding_route_request_sha256"])),
        forecast_seal=seal["seal_sha256"], no_persistent_parameter_bank=True))
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], counts=counts, contract=declared,
        observed=observed, forecast_seal=seal["seal_sha256"], transaction=verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="completed", instrument_status="full functional forecasts and native continuations", trend_status="uninterpreted observations retained", terminal_status="unassessed", seconds=time.monotonic()-started)
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
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "request/config changed")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "executed source changed: " + name)
    job = os.environ["MRUN_JOB_ID"]
    require(job.startswith("job-") and "/" not in job, "invalid job identity")
    derived = dict(config, job_id=job, request_sha256=identity, _request_record=request,
        output=str(Path(config["output_scratch"]) / job / RESULT))
    result = run(derived, progress=lambda row: print("MDB_LIFECYCLE_PHASE " + canonical(row).decode(), flush=True))
    finalizer = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))
    ready = finalizer["finalize"](derived["output"], job_id=result["job_id"], request_sha256=identity, **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    faulthandler.cancel_dump_traceback_later()
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
