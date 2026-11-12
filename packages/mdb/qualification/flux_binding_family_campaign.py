"""Producer-to-image computation-family assays with original primitive equations.

Historical mode separates actual text R/A/M contributions. Fresh mode consumes
new prefix cuts and a separately frozen route rule. All numeric futures are
globally sealed before native source-block or suffix targets execute.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "mdb-flux-binding-family-v1"
RESULT = "flux-binding-family-result"


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


def historical_arms():
    return ["".join(corner) for corner in itertools.product("QP", repeat=3)] + ["QPM", "PQM"]


def helpers():
    parent = runpy.run_path(str(HERE / "flux_binding_residual_write.py"))
    lifecycle, source = parent["frozen_helpers"]()
    common, base, old = parent["common_helpers"]()
    return parent, lifecycle, source, common, base, old


def run(config, *, progress=None):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from saturn_pub.core import Session, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from flux_binding_family_writer import (
        frozen_endpoint, functional_joint0, functional_recomputed_endpoint,
        native_joint0, native_recomputed_endpoint,
    )

    parent, lifecycle, source_helper, common, base, old = helpers()
    for name, digest in config["frozen_helpers"].items():
        require(sha(HERE / name) == digest, "frozen helper differs: " + name)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    phase, released, observed = dict(kind="source-loading"), dict(native=False), {}
    counts = dict(functional_transitions=0, native_target_transitions=0, native_blocks=0,
        functional_source_joint0=0, native_source_joint0=0, functional_source_mlp=0, native_source_mlp=0,
        functional_vae=0, native_vae=0, acts=0, rewinds=0)
    save(output / "plan.json", {k: v for k, v in config.items() if k != "_request_record"})
    save(output / "request.json", config["_request_record"])

    def event(kind, **details):
        row = dict(kind=kind, seconds=time.monotonic()-started, counts=dict(counts), **details)
        require(row["seconds"] <= config["max_wall_seconds"], "declared worker budget exceeded")
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        if progress:
            progress(row)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "qualified CUDA required")
    modules = old["_helper_modules"]({k: v for k, v in config.items() if k != "worker_sha256"})
    qw, graph, decoder_graph = (modules[old[k]] for k in ("QWEN_WORKER", "DENOISER_HELPER", "DECODER_HELPER"))
    runtime = qw["_runtime_contract"](config, torch)
    runtime["native_diffusers_sources"] = old["_native_source_contract"](config)["sources"]
    save(output / "runtime-contract.json", runtime)
    for key in old["_source_keys"]():
        path = Path(config[key])
        require(sha(path) == config[key + "_sha256"], "parameter/runtime metadata differs: " + key)
        target = output / "source" / key / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = read(config["flux_manifest"])
    for item in read(config["vae_manifest"])["artifacts"]:
        item = copy.deepcopy(item)
        item["path"] = "vae-" + item["path"]
        item["source"]["file"] = "vae/" + item["source"]["file"]
        manifest["artifacts"].append(item)
    manifest["defaults"] = dict(artifact_root="parameter-cache", checkpoint_root=config["flux_parameter_root"], ssh_host="beast", python="python3")
    manifest["provenance"] = dict(schema=SCHEMA, originalParametersOnly=True, scope="169transformer and142decoder/BN recipes; no parameter-bank export")
    runpy.run_path(config["flux_retrieval_script"])["validate_manifest"](manifest)
    save(output / "parameter-artifacts.json", manifest)
    shutil.copyfile(config["flux_retrieval_script"], output / "retrieve_weights.py")
    flux, _, _ = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae = AutoencoderKLFlux2.from_pretrained(config["vae_parameter_root"], torch_dtype=torch.bfloat16,
        local_files_only=True).eval().requires_grad_(False)
    graph["validate_model"](flux)
    models = dict(flux=flux, vae=vae)
    prefixes = dict(flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
    identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}

    def place(name, device):
        if name == "flux":
            flux.to(device)
        else:
            vae.post_quant_conv.to(device)
            vae.decoder.to(device)
        old["_assert_signature_equal"](identities[name], old["_component_signature"](models[name], prefixes=prefixes[name]), name)
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
        return retain(StateCut(active.model_identity,
            {**dict(active.execution), "composition": SCHEMA, "provenance": "original component equations"},
            active.boundary(state), prior, state, active.point(state).to_dict(), active.surface(state).to_dict()))

    active = adapter("cpu")
    if config["mode"] == "historical":
        binding, table = source_helper["prepare_sources"](config, output, store, active, base, old)
        roots = {cid: dict(inputs={role: info["metadata"]["joint0_inputs"][role] for role in ("P", "Q")},
            support=info["metadata"]["support"], base_conditioning=info["metadata"]["base_conditioning"],
            known=info["metadata"]["known_endpoints"]) for cid, info in binding.items()}
        save(output / "source-provenance.json", table["provenance"])
    else:
        prepare = runpy.run_path(str(HERE / "flux_binding_family_fresh_sources.py"))["load_sources"]
        roots, provenance = prepare(config, output, store, active, base)
        save(output / "source-provenance.json", provenance)
    save(output / "source-roots.json", roots)
    del active
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels)-1)*2)
    paths, source_ports, source_states, endpoint_records, recomputed_records = {}, {}, {}, {}, {}
    source_comparisons, port_records = [], {"forecast": {}, "native": {}}
    expected_cases = len(roots) * len(config["arms"])

    def save_ports(cid, role, lane, values):
        return {name: old["_save_tensor"](output, f"writer/{cid}/{role}/{lane}/{name}.npy", value)
                for name, value in values.items()}

    def source_writers(lane):
        active = adapter("cuda")
        for cid, root in roots.items():
            local = {}
            for role in ("Q", "P"):
                cut = store.load(root["inputs"][role]["cut"], device="cuda")
                if lane == "forecast":
                    result = functional_joint0(active, cut.payload, expected_model_identity=active.model_identity)
                    counts["functional_source_joint0"] += 1
                else:
                    session = Session(active, cut.payload, parent=cut.parent)
                    with native_joint0(active, cut.payload, expected_model_identity=active.model_identity) as result:
                        session.continue_()
                    require(describe(result["state"]) == describe(session.capture(retain=False).payload), "native source port closure differs")
                    counts["native_source_joint0"] += 1
                values = result["ports"]
                local[role] = values
                record = component(active, result["state"], cut.fingerprint) if lane == "forecast" else retain(session.capture(retain=False))
                source_states[cid, role, lane] = record
                source_ports[cid, role, lane] = save_ports(cid, role, lane, values)
                if lane == "native":
                    forecast = source_ports[cid, role, "forecast"]
                    for name, ref in source_ports[cid, role, lane].items():
                        a, b = (base["load_tensor"](output, item, "cpu") for item in (forecast[name], ref))
                        source_comparisons.append(dict(context=cid, role=role, port=name,
                            bit_exact=bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))),
                            max_abs=float((a.double()-b.double()).abs().max())))
            q, p = local["Q"], local["P"]
            rows = root["support"]
            endpoint_records[cid, lane] = {}
            recomputed_records[cid, lane] = {}
            for arm in config["arms"]:
                if arm in ("QPM", "PQM"):
                    cut = store.load(root["inputs"]["Q"]["cut"], device="cuda")
                    operation = functional_recomputed_endpoint if lane == "forecast" else native_recomputed_endpoint
                    result = operation(active, cut.payload, q, p, rows, arm[0], arm[1], expected_model_identity=active.model_identity)
                    endpoint = result["endpoint"]
                    counts[("functional" if lane == "forecast" else "native") + "_source_mlp"] += 1
                    recomputed_records[cid, lane][arm] = save_ports(cid, arm, lane, result["ports"])
                elif arm == "whole_P":
                    endpoint = p["text_out"].clone()
                elif arm == "complement_P":
                    endpoint = p["text_out"].clone()
                    endpoint[:, rows] = q["text_out"][:, rows]
                elif arm == "wrong_address":
                    endpoint = q["text_out"].clone()
                    endpoint[:, root["wrong_rows"]] = p["text_out"][:, rows]
                else:
                    endpoint = frozen_endpoint(q, p, rows, *arm)
                endpoint_records[cid, lane][arm] = old["_save_tensor"](output, f"endpoints/{cid}/{lane}/{arm}.npy", endpoint)
            event(lane + "_writer_complete", context=cid)
            del local, q, p
        save(output / (lane + "-source-writers.json"), {
            cid: dict(sources={role: dict(state=source_states[cid, role, lane], ports=source_ports[cid, role, lane])
                               for role in ("P", "Q")}, endpoints=endpoint_records[cid, lane],
                      recomputed=recomputed_records[cid, lane]) for cid in roots})

    def edit(session, slot, tensor):
        before, after, receipt = base["apply_slot_edit"](session, slot, tensor, mode="replace")
        require(describe({k: v for k, v in before.payload.items() if k != slot}) == describe({k: v for k, v in after.payload.items() if k != slot}), "Act changed another slot")
        counts["acts"] += 1
        retain(before)
        return retain(after), receipt.to_dict()

    def suffixes(lane):
        active = adapter("cuda")
        for cid, root in roots.items():
            origin = store.load(source_states[cid, "Q", lane]["cut"], device="cuda")
            master = Session(active, origin.payload, parent=origin.fingerprint if lane == "forecast" else origin.parent)
            common = master.capture(retain=False)
            require(describe(common.payload) == describe(origin.payload), "public branch wrapper changed predicted state")
            retain(common)
            for arm in config["arms"]:
                key = cid + "/" + arm
                branch = output / "cases" / cid / arm
                branch.mkdir(parents=True, exist_ok=True)
                session = master.fork(common)
                start, act = edit(session, "text", base["load_tensor"](output, endpoint_records[cid, lane][arm], "cuda"))
                save(branch / (lane + "-entry-act.json"), dict(receipt=act, common=base["cut_record"](common), after=start))
                state, prior = session.capture(retain=False).payload, start["cut"]
                rows, ports = [dict(number=0, fields=base["tensor_tree"](state, old))], {}
                for number in range(1, 55):
                    clock = {name: state[name] for name in ("step", "phase", "cursor")}
                    if clock == dict(step=3, phase="project", cursor=0):
                        if lane == "forecast":
                            session = Session(active, state, parent=prior)
                        reset, receipt = edit(session, "conditioning", base["load_tensor"](output, root["base_conditioning"], "cuda"))
                        state, prior = session.capture(retain=False).payload, reset["cut"]
                        save(branch / (lane + "-reset-act.json"), dict(receipt=receipt, after=reset))
                    if lane == "forecast":
                        if number == 1:
                            state, values = common_helpers_consumer(base, graph, flux, state)
                            ports = {name: old["_save_tensor"](output, f"cases/{cid}/{arm}/consumer/{lane}/{name}.npy", value) for name, value in values.items()}
                        else:
                            state = base["step_equations"](graph, flux, state)
                        counts["functional_transitions"] += 1
                    else:
                        if number == 1:
                            def port(name, value):
                                ports[name] = old["_save_tensor"](output, f"cases/{cid}/{arm}/consumer/{lane}/{name}.npy", value)
                            port("gate_ff", graph["_mods"](state["context"]["img_mod"], 2)[1][2])
                            with common_consumer_hooks(flux, port):
                                session.continue_()
                        else:
                            session.continue_()
                        state = session.capture(retain=False).payload
                        counts["native_target_transitions"] += 1
                    rows.append(dict(number=number, event=clock, fields=base["tensor_tree"](state, old)))
                require((state["step"], state["phase"], state["cursor"]) == (4, "project", 0), "suffix horizon differs")
                final = component(active, state, prior) if lane == "forecast" else retain(session.capture(retain=False))
                save(branch / (lane + "-rows.json"), rows)
                port_records[lane][key] = ports
                paths.setdefault(key, dict(context=cid, arm=arm, root=str(branch.relative_to(output))))[lane + "_final"] = final
                if lane == "native" and arm == config["arms"][0]:
                    session.restore(common)
                    require(session.capture(retain=False).fingerprint == common.fingerprint, "common-parent rewind differs")
                    counts["rewinds"] += 1
                event(lane + "_suffix_complete", case=key)

    common_helpers_consumer = lambda _base, g, m, s: common["consumer_equations"](g, m, s)
    common_consumer_hooks = common["passive_consumer_ports"]

    def decodes(lane):
        for key, path in paths.items():
            branch = output / path["root"]
            state = store.load(path[lane + "_final"]["cut"], device="cpu").payload
            if lane == "forecast":
                argument = old["_decoder_input"](vae, state, config, {"Flux2KleinPipeline": Flux2KleinPipeline}).to("cuda")
                decoded = decoder_graph["functional_decode"](vae, argument, counts={}, profile="original")
                record = base["decoder_result"](old, branch, lane, decoded, processor)
                record["decoder_input"] = old["_save_tensor"](branch, "decoder/forecast/decoder-input.npy", argument)
            else:
                record = old["_native_decoder"](branch, vae, state, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, {}, {})
            save(branch / ("decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json"), record)
            counts[("functional" if lane == "forecast" else "native") + "_vae"] += 1

    def entered(_module, _args):
        require(released["native"] and phase["kind"].startswith("target"), "native learned block before global seal")
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
            save(output / "forecast-consumer-ports.json", port_records["forecast"])
            inventory = old["_inventory"](output, exclude=("timeline.jsonl",))
            tx = ForecastTransaction(output / "forecast-transaction", model_identity=SCHEMA,
                input_identity=hashlib.sha256(canonical(roots)).hexdigest(), source_identity=hashlib.sha256(canonical(config["_request_record"]["source_hashes"])).hexdigest(),
                frontier={"kind": "architectural", "scope": "joint0 inputs through full source writer and own-state image continuation", "external_files": inventory},
                execution_scope="Original-weight shared-primitive equations; no intermediate native refresh",
                budgets=dict(max_predictions=1, max_executions=1, max_bytes=8 << 30, max_files=30000, max_wall_ms=config["max_wall_seconds"]*1000))
            tx.frontier("source-inputs-and-rule", dict(roots=roots, family_rule=config.get("family_rule"), hypotheses=config["hypotheses"]))
            tx.predict("all-numeric-futures", lambda _: {"predictions": {"external_files": inventory, "paths": paths, "native_targets_at_seal": counts["native_target_transitions"], "native_source_blocks_at_seal": counts["native_source_joint0"]}})
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
                return {"counts": counts, "paths": len(paths)}

            tx.execute("native-source-and-futures", targets)
            tx.close()
    finally:
        for handle in handles:
            handle.remove()
    old["_verify_inventory"](output, inventory)
    for name, model in models.items():
        old["_assert_signature_equal"](identities[name], old["_component_signature"](model, prefixes=prefixes[name]), name)
    require(counts["functional_transitions"] == counts["native_target_transitions"] == 54*expected_cases, "transition census differs")
    require(counts["functional_vae"] == counts["native_vae"] == expected_cases, "decoder census differs")
    require(counts["native_blocks"] == 49*expected_cases + 2*len(roots), "native block census differs")
    save(output / "state-index.json", paths)
    save(output / "consumer-ports.json", port_records)
    save(output / "writer-comparisons.json", source_comparisons)
    save(output / "native-contract.json", dict(counts=counts, observed=observed, identities=identities))
    result = dict(schema=SCHEMA, mode=config["mode"], job_id=config["job_id"], request_sha256=config["request_sha256"],
        contexts=list(roots), arms=config["arms"], counts=counts, observed=observed, forecast_seal=seal["seal_sha256"],
        transaction=verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="completed", instrument_status="awaiting-independent-saved-audit", terminal_status="unassessed", seconds=time.monotonic()-started)
    save(output / "summary.json", result)
    return result


def worker(config_path):
    config = read(config_path)
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "request/config differs")
    for name, digest in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == digest, "executed source differs: " + name)
    job = os.environ["MRUN_JOB_ID"]
    config.update(job_id=job, request_sha256=identity, _request_record=request,
        output=str(Path(config["output_scratch"]) / job / RESULT))
    result = run(config, progress=lambda row: print("MDB_FAMILY " + canonical(row).decode(), flush=True))
    ready = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))["finalize"](config["output"], job_id=job,
        request_sha256=identity, max_expanded_bytes=8 << 30, max_member_bytes=134221824)
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
