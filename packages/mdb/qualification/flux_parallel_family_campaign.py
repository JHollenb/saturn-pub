"""Prospective parallel-image relation family: source equations to full futures."""
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
SCHEMA = "mdb-flux-parallel-relation-family-v1"
RESULT = "flux-parallel-relation-result"
SOURCE_PORTS = ("R", "A", "M", "gate", "projection_return", "out", "image_in", "text_in", "image_out", "text_out")
CONSUMER_PORTS = ("image_in", "text_in", "A", "M", "image_out", "text_out", "projection_return", "gate")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def run(config):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from saturn_pub.core import Session, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    writer = runpy.run_path(str(HERE / "flux_parallel_family_writer.py"))
    prefix = runpy.run_path(str(HERE / "flux_parallel_family_sources.py"))
    parent = runpy.run_path(str(HERE / "flux_binding_residual_write.py"))
    lifecycle, _ = parent["frozen_helpers"]()
    _, base, old = parent["common_helpers"]()
    for name, digest in config["frozen_helpers"].items():
        require(sha(HERE / name) == digest, "frozen helper changed: " + name)
    require(config["contexts"] == config["family_rule"]["fresh_contexts"] == prefix["contexts"]()
        and config["arms"] == ["Q", "P", "R", "A", "RA"], "declared finite panel differs")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started, phase, released, observed = time.monotonic(), dict(kind="loading"), dict(native=False), {}
    counts = dict(functional_source_blocks=0, native_source_blocks=0, functional_hybrid_projections=0, native_hybrid_projections=0,
        functional_transitions=0, native_target_transitions=0, native_blocks=0, functional_vae=0, native_vae=0,
        acts=0, forecast_acts=0, native_acts=0, rewinds=0)
    save(output / "plan.json", {k: v for k, v in config.items() if k != "_request_record"})
    save(output / "request.json", config["_request_record"])
    shutil.copyfile("family-rule.json", output / "family-rule.json")

    def event(kind, **fields):
        value = dict(kind=kind, seconds=time.monotonic()-started, counts=dict(counts), **fields)
        require(value["seconds"] <= config["max_wall_seconds"], "finite wall allowance exhausted")
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(value) + b"\n")
        print("MDB_PARALLEL_FAMILY " + canonical(value).decode(), flush=True)

    torch.set_num_threads(4)
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
    models = dict(flux=flux, vae=vae)
    parameter_prefix = dict(flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
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
    roots, provenance = prefix["load_sources"](config, output, store, active)
    save(output / "source-roots.json", roots)
    save(output / "source-provenance.json", provenance)
    del active
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels)-1)*2)
    states, records, endpoints, endpoint_meta, paths, consumer_records = {}, {}, {}, {}, {}, dict(forecast={}, native={})

    def save_ports(relative, values):
        return {name: old["_save_tensor"](output, f"{relative}/{name}.npy", value) for name, value in values.items()}

    def source_writers(lane):
        active = adapter("cuda")
        for cid, root in roots.items():
            local = {}
            for role in ("Q", "P"):
                cut = store.load(root["inputs"][role]["cut"], device="cuda")
                if lane == "forecast":
                    result = writer["functional_single"](active, cut.payload, index=4,
                        expected_model_identity=active.model_identity, retain_ports=SOURCE_PORTS)
                    state = component(active, result["state"], cut.fingerprint)
                    counts["functional_source_blocks"] += 1
                else:
                    session = Session(active, cut.payload, parent=cut.parent)
                    with writer["native_single"](active, cut.payload, index=4,
                            expected_model_identity=active.model_identity, retain_ports=SOURCE_PORTS) as result:
                        session.continue_()
                    require(describe(result["state"]) == describe(session.capture(retain=False).payload), "native source port closure differs")
                    state = retain(session.capture(retain=False))
                    counts["native_source_blocks"] += 1
                local[role] = result["ports"]
                states[cid, role, lane] = state
                records[cid, role, lane] = dict(state=state, ports=save_ports(f"writer/{cid}/{role}/{lane}", result["ports"]),
                    descriptors=result["descriptors"], retained_descriptors=result["retained_descriptors"],
                    retention=result["retention"], profile=result["profile"], calls=result.get("native_calls", result.get("operation_counts")))
            q, p = local["Q"], local["P"]
            endpoint_meta[cid, lane] = {}
            for arm in config["arms"]:
                result = writer["construct_endpoint"](q, p, arm, native=lane == "native")
                endpoints[cid, lane, arm] = old["_save_tensor"](output, f"endpoints/{cid}/{lane}/{arm}.npy", result["image"])
                endpoint_meta[cid, lane][arm] = dict(descriptors=result["descriptors"], construction=result["construction"],
                    projection_return=None if arm == "P" else old["_save_tensor"](output, f"endpoints/{cid}/{lane}/{arm}-projection.npy", result["ports"]["projection_return"]))
                if arm != "P":
                    counts[("functional" if lane == "forecast" else "native") + "_hybrid_projections"] += 1
                del result
            values = {arm: base["load_tensor"](output, endpoints[cid, lane, arm], "cuda").double() for arm in ("Q", "R", "A", "RA")}
            remainder = values["RA"]-values["R"]-values["A"]+values["Q"]
            endpoint_meta[cid, lane]["finite_mixed_remainder"] = old["_save_tensor"](output, f"endpoints/{cid}/{lane}/mixed-remainder.npy", remainder)
            del values, remainder, local, q, p
            event(lane + "_writer_complete", context=cid)
        save(output / (lane + "-source-writers.json"), {cid: dict(sources={role: records[cid, role, lane] for role in ("P", "Q")},
            endpoints={arm: endpoints[cid, lane, arm] for arm in config["arms"]}, constructions=endpoint_meta[cid, lane]) for cid in roots})

    def edit(session, value, lane):
        before, after, receipt = base["apply_slot_edit"](session, "image", value, mode="replace")
        require(describe({k: v for k, v in before.payload.items() if k != "image"})
            == describe({k: v for k, v in after.payload.items() if k != "image"}), "image Act changed another slot")
        counts["acts"] += 1
        counts[lane + "_acts"] += 1
        retain(before)
        return retain(after), receipt.to_dict()

    def suffixes(lane):
        active = adapter("cuda")
        for cid in roots:
            q = store.load(states[cid, "Q", lane]["cut"], device="cuda")
            master = Session(active, q.payload, parent=q.fingerprint if lane == "forecast" else q.parent)
            common = master.capture(retain=False)
            retain(common)
            for arm in config["arms"]:
                key = cid + "/" + arm
                branch = output / "cases" / cid / arm
                branch.mkdir(parents=True, exist_ok=True)
                if arm == "P":
                    p = store.load(states[cid, "P", lane]["cut"], device="cuda")
                    session = Session(active, p.payload, parent=p.fingerprint if lane == "forecast" else p.parent)
                    start = retain(session.capture(retain=False))
                    save(branch / (lane + "-entry-act.json"), dict(reference="natural-P", source=states[cid, "P", lane], after=start, receipt=None))
                else:
                    session = master.fork(common)
                    start, receipt = edit(session, base["load_tensor"](output, endpoints[cid, lane, arm], "cuda"), lane)
                    if arm == "Q":
                        require(describe(session.capture(retain=False).payload) == describe(common.payload), "Q self-copy full state differs")
                    save(branch / (lane + "-entry-act.json"), dict(common=base["cut_record"](common), after=start, receipt=receipt))
                state, prior = session.capture(retain=False).payload, start["cut"]
                rows, consumer = [dict(number=0, fields=base["tensor_tree"](state, old))], None
                for number in range(1, 102):
                    clock = {name: state[name] for name in ("step", "phase", "cursor")}
                    if number == 1:
                        if lane == "forecast":
                            result = writer["functional_single"](active, state, index=5,
                                expected_model_identity=active.model_identity, retain_ports=CONSUMER_PORTS)
                            state = result["state"]
                        else:
                            with writer["native_single"](active, state, index=5,
                                    expected_model_identity=active.model_identity, retain_ports=CONSUMER_PORTS) as result:
                                session.continue_()
                            state = session.capture(retain=False).payload
                            require(describe(result["state"]) == describe(state), "native first-consumer closure differs")
                        consumer = dict(ports=save_ports(f"cases/{cid}/{arm}/consumer/{lane}", result["ports"]),
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
                consumer_records[lane][key] = consumer
                paths.setdefault(key, dict(context=cid, arm=arm, root=str(branch.relative_to(output))))[lane + "_final"] = final
                if lane == "native" and arm == "Q":
                    session.restore(common)
                    require(session.capture(retain=False).fingerprint == common.fingerprint, "Q parent rewind differs")
                    counts["rewinds"] += 1
                event(lane + "_suffix_complete", case=key)

    def decodes(lane):
        for path in paths.values():
            branch = output / path["root"]
            state = store.load(path[lane + "_final"]["cut"], device="cpu").payload
            if lane == "forecast":
                argument = old["_decoder_input"](vae, state, config, {"Flux2KleinPipeline": Flux2KleinPipeline}).to("cuda")
                decoded = decoder["functional_decode"](vae, argument, counts={}, profile="original")
                record = base["decoder_result"](old, branch, lane, decoded, processor)
                record["decoder_input"] = old["_save_tensor"](branch, "decoder/forecast/decoder-input.npy", argument)
            else:
                record = old["_native_decoder"](branch, vae, state, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, {}, {})
            save(branch / ("decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json"), record)
            counts["functional_vae" if lane == "forecast" else "native_vae"] += 1

    def entered(_module, _args):
        require(released["native"] and phase["kind"].startswith("target"), "native transformer before global numeric seal")
        counts["native_blocks"] += 1

    handles = [block.register_forward_pre_hook(entered) for block in (*flux.transformer_blocks, *flux.single_transformer_blocks)]
    try:
        with base["instrument"](list(models.values()), phase, released, observed), torch.inference_mode():
            place("flux", "cuda")
            phase["kind"] = "forecast-writer"
            source_writers("forecast")
            phase["kind"] = "forecast-denoiser"
            suffixes("forecast")
            place("flux", "cpu")
            place("vae", "cuda")
            phase["kind"] = "forecast-vae"
            decodes("forecast")
            place("vae", "cpu")
            save(output / "forecast-paths.json", paths)
            save(output / "forecast-consumer-ports.json", consumer_records["forecast"])
            inventory = old["_inventory"](output, exclude=("timeline.jsonl",))
            tx = ForecastTransaction(output / "forecast-transaction", model_identity=SCHEMA,
                input_identity=hashlib.sha256(canonical(roots)).hexdigest(), source_identity=hashlib.sha256(canonical(config["_request_record"]["source_hashes"])).hexdigest(),
                frontier=dict(kind="architectural", scope="Observed natural0single4 inputs; original equations produce selectedwriter and full ownstate futures", external_files=inventory),
                execution_scope="Original-weight shared primitives; no intermediate native refresh or fitted parameters",
                budgets=dict(max_predictions=1, max_executions=1, max_bytes=12 << 30, max_files=30000, max_wall_ms=config["max_wall_seconds"]*1000))
            tx.frontier("frozen-family-and-inputs", dict(roots=roots, rule=config["family_rule"]))
            tx.predict("all-numeric-futures", lambda _: dict(predictions=dict(external_files=inventory, paths=paths,
                native_source_blocks_at_seal=counts["native_source_blocks"], native_hybrid_projections_at_seal=counts["native_hybrid_projections"],
                native_targets_at_seal=counts["native_target_transitions"], native_decodes_at_seal=counts["native_vae"])))
            seal = tx.seal()
            event("all_forecasts_sealed", seal=seal["seal_sha256"])

            def targets(_):
                old["_verify_inventory"](output, inventory)
                released["native"] = True
                place("flux", "cuda")
                phase["kind"] = "target-writer"
                source_writers("native")
                phase["kind"] = "target-denoiser"
                suffixes("native")
                place("flux", "cpu")
                place("vae", "cuda")
                phase["kind"] = "target-vae"
                decodes("native")
                place("vae", "cpu")
                return dict(counts=counts, futures=len(paths))

            tx.execute("native-source-writer-and-futures", targets)
            tx.close()
    finally:
        for handle in handles:
            handle.remove()
    old["_verify_inventory"](output, inventory)
    for name, model in models.items():
        old["_assert_signature_equal"](identities[name], old["_component_signature"](model, prefixes=parameter_prefix[name]), name)
    expected = dict(functional_source_blocks=12, native_source_blocks=12, functional_hybrid_projections=24, native_hybrid_projections=24,
        functional_transitions=3030, native_target_transitions=3030, native_blocks=2712,
        functional_vae=30, native_vae=30, acts=48, forecast_acts=24, native_acts=24, rewinds=6)
    require(counts == expected, "physical campaign census differs")
    save(output / "state-index.json", paths)
    save(output / "consumer-ports.json", consumer_records)
    save(output / "native-contract.json", dict(counts=counts, observed=observed, identities=identities))
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], counts=counts,
        observed=observed, contexts=list(roots), arms=config["arms"], forecast_seal=seal["seal_sha256"],
        transaction=verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="completed", instrument_status="awaiting-independent-saved-audit", terminal_status="unassessed", seconds=time.monotonic()-started)
    save(output / "summary.json", result)
    return result


def worker(config_path):
    config, request = read(config_path), read(os.environ["SATURN_DEBUG_REQUEST"])
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "request/config differs")
    for name, digest in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == digest, "executed payload differs")
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
