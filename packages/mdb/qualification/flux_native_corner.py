"""Native same-parent N1/P1 corner; no independent predictor recomputation.

Imported native step1/project StateCuts supply every carried operand. Only the
committed latent is replaced, using the original sealed P1 or an exact N1 sham.
Both source-bound transactions seal before any native denoiser/VAE entry.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

RESULT = "flux-native-corner-result"
CELLS = ["c1-o0", "c0-o1"]
FACTORIAL_SHA256 = "5acf66a6697f096491f4769692f656e480c4c5c59d109374f7397a174002e22c"
PARAMETER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
EXPECTATION_SHA256 = "cbba85f3109006007b02edb4af794feab76dedb553ccba6ae954866b95bc4214"
RESPONSE_CONTRACT_SHA256 = "6a3113d255ed8448e5c20a93c0e7a3a0cb6b1a1ef39ea3d86bfbd1902dda5d2b"
COUNTS = {"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0, "adapter_steps": 336, "acts": 4,
    "receipt_emissions": 340, "images": 4, "vae_calls": 4, "candidate_calls": 0, "physical_batch": 1,
    "mathematical_forwards": 0}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


@lru_cache(maxsize=1)
def support():
    path = Path(__file__).with_name("flux_prompt_factorial.py")
    require(sha(path) == FACTORIAL_SHA256, "immutable original producer helper differs")
    result = runpy.run_path(str(path))
    result["prior"]()
    return result


def target_counts(multiplier=12):
    return {**{str(i): multiplier for i in range(20)}, "norm_out": multiplier, "proj_out": multiplier,
        **{name: multiplier for name in ("x_embedder", "context_embedder", "time_guidance_embed",
            "double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation")},
        "pos_embed": 2 * multiplier, **{"joint" + str(i): multiplier for i in range(5)}}


def fixed_config():
    return {"schema": "mdb-native-corner-plan-v1", "conditions": list(CELLS), "logical_steps": [1, 2, 3],
        "height": 512, "width": 512, "steps": 4, "cpu_threads": 4, "witness_count": 64,
        "branch_order": [cell + "." + role for cell in CELLS for role in ("N1.sham", "P1")],
        "selector": {"logical_step": 1, "phase": "project", "index": None},
        "observation_plan": {"pass1_landmark_locals": [0, 5, 25, 26, 27], "later_landmark_locals": [27],
            "full_landmark_transitions": 28, "descriptor_only_transitions": 308,
            "scope": "All336 receipts/manifests/full field hashes and fixed64 witnesses; only declared28 exits hydratable."},
        "counts": dict(COUNTS), "target_calls": target_counts(), "target_calls_per_condition": target_counts(6),
        "max_wall_seconds": 900, "transaction_max_bytes": 64 << 20,
        "parameter_storage": "checkpoint-recipe-v1", "array_storage": "bf16-bits",
        "expectations_sha256": EXPECTATION_SHA256, "expectations_path": "expectations.json",
        "response_contract_sha256": RESPONSE_CONTRACT_SHA256, "response_contract_path": "response-contract.json",
        "transport_method": "planned-authenticated-ssh-manifest",
        "transport_limits": {"max_expanded_bytes": 4 << 30, "max_member_bytes": 1 << 30}}


def make_config(parent_config=None):
    parent = parent_config or {}
    result = {key: value for key, value in parent.items() if key.startswith("expected_") or key in ("model", "model_path")}
    result.update(fixed_config())
    return json.loads(json.dumps(result))


def validate_config(config):
    require(all(config.get(key) == value for key, value in fixed_config().items()), "native-corner finite contract differs")


def retained(logical, local):
    return local in (0, 5, 25, 26, 27) if logical == 1 else local == 27


def latent_act(cut, value, name):
    """Native project already has reset working slots; invalidates are empty."""
    import torch
    from saturn_pub import Act
    from saturn_pub.values import describe, digest

    require((cut.payload["step"], cut.payload["phase"], cut.payload["cursor"]) == (1, "project", 0)
        and all(cut.payload[key] is None for key in ("image", "text", "context", "noise_prediction")), "requires complete native N1 project cut")
    reference = cut.payload["latent"]
    require(value.shape == reference.shape and value.dtype == reference.dtype and value.device == reference.device
        and bool(torch.isfinite(value).all()), "replacement latent shape/dtype/device/finite differs")
    payload = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
    payload.copy_(value)
    # Base adapter exposes latent/conditioning at project. Session's declared
    # authoritative footprint preserves all other twelve slots, including clock.
    return replace(Act.replace("latent", payload, name=name), preconditions={key: digest(describe(cut.payload[key]))
        for key in ("latent", "conditioning")}, preserve=("conditioning",))


def run(*, adapter, parents, predictions, references, output, config, request, source, counts,
        parameter_manifest, retrieval_source, expectations, response_contract, guard_components, encoder_components=(),
        array=None, progress=None, production_profile=True):
    """Tiny-testable native worker. Never evaluates numerical predictor code."""
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    validate_config(config)
    f = support()
    old = f["prior"]()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def wall():
        require(time.monotonic() - started <= config["max_wall_seconds"], "finite native-corner wall budget exhausted")
    array = array or old["step"]()["make_array_writer"](output, wall)
    store = LocalStore(output / "typed-store")
    save(output / "plan.json", config)
    save(output / "expectations.json", expectations)
    save(output / "response-contract.json", response_contract)
    require(set(parents) == set(predictions) == set(references) == set(CELLS), "two selected source cases required")
    require(sha(retrieval_source) == PARAMETER_SHA256 and parameter_manifest["artifacts"]
        and (not production_profile or len(parameter_manifest["artifacts"]) == 169), "qualified169 recipe/retriever inventory required")
    manifest = json.loads(json.dumps(parameter_manifest))
    manifest["provenance"] = {**manifest.get("provenance", {}), "parent_provenance": manifest.get("provenance", {}),
        "job_id": request["job_id"], "request_sha256": request["request_sha256"], "producer_source_sha256": sha(__file__),
        "scope": "Copied qualified checkpoint recipes; no parameter archives written or hydrated by this native-only worker."}
    save(output / "parameter-artifacts.json", manifest)
    with (output / "retrieve_weights.py").open("xb") as stream:
        stream.write(Path(retrieval_source).read_bytes())
    model = adapter.model
    require(len(model.transformer_blocks) == 5 and len(model.single_transformer_blocks) == 20, "native block geometry differs")
    for cell, cut in parents.items():
        cut.verify()
        require(cut.model_identity == adapter.model_identity and dict(cut.execution) == dict(adapter.execution), "imported model/execution identity differs")
        latent_act(cut, cut.payload["latent"], "preflight")
        latent_act(cut, predictions[cell]["latent"], "preflight-P1")
        if production_profile:
            require(cut.payload["latent"].shape == (1, 1024, 128) and cut.payload["conditioning"].shape == (1, 512, 7680)
                and cut.payload["latent"].device.type == "cuda" and cut.payload["latent"].dtype == torch.bfloat16,
                "qualified production input profile differs")
    parent_identities = {cell: identity(describe(cut.payload)) for cell, cut in parents.items()}
    parameters_before = old["prior"]()["parameter_census"](guard_components)
    state = {"mode": "sealed", "cell": None, "blocked": 0, "target_calls": dict.fromkeys(target_counts(), 0),
        "per_condition": {cell: dict.fromkeys(target_counts(), 0) for cell in CELLS}}
    targets = {**{block: str(i) for i, block in enumerate(model.single_transformer_blocks)},
        **{block: "joint" + str(i) for i, block in enumerate(model.transformer_blocks)},
        **{getattr(model, name): name for name in ("norm_out", "proj_out", "x_embedder", "context_embedder", "time_guidance_embed",
            "double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation", "pos_embed")}}
    vae_ids = {id(m) for component in guard_components if component is not model for m in component.modules()}
    encoder_ids = {id(m) for component in encoder_components for m in component.modules()}
    def guard(module, args):
        if id(module) in encoder_ids or state["mode"] == "sealed" or (state["mode"] == "vae" and id(module) not in vae_ids):
            state["blocked"] += 1
            raise ValueError("component entered outside sealed native/VAE scope; encoding prohibited")
        if module in targets:
            key = targets[module]
            state["target_calls"][key] += 1
            state["per_condition"][state["cell"]][key] += 1
    handles = [m.register_forward_pre_hook(guard) for component in (*guard_components, *encoder_components) for m in component.modules()]
    def emit(kind, **fields):
        body = {"kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields}
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True) + "\n")
        if progress:
            progress(body)
    def capture(cut, keep):
        body = cut.manifest()
        require(identity(body) == cut.fingerprint, "typed StateCut descriptor identity differs")
        if keep:
            store.save(cut)
        return {"cut": cut.fingerprint, "point": dict(cut.execution_point), "manifest": body,
            "retained_full_cut": keep, "scope": "Hydratable snapshot" if keep else "Descriptor/receipt only; no complete replay retained"}
    def accept(receipt, branch):
        body = receipt.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        counts["receipt_emissions"] += 1
        require(counts["adapter_steps"] <= 336 and counts["acts"] <= 4, "physical native-corner bound exceeded")
        store.receipt(receipt)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"] - 1, "branch": branch, "receipt": receipt.fingerprint}) + "\n")
    txs, seals, rows, bindings = {}, {}, {}, {}
    try:
        for cell in CELLS:
            cut = parents[cell]
            store.save(cut)
            wide = predictions[cell]["latent"].detach().float().cpu().numpy().copy()
            require(not np.any(wide.view(np.uint32) & np.uint32(65535)), "sealed P1 is not lossless native BF16")
            require(f["descriptor"](wide) == predictions[cell]["binding"]["record"]["descriptor"],
                "installed P1 differs from its original sealed source descriptor")
            frontier = {"condition_id": cell, "native_N1": capture(cut, True), "source_state": describe(cut.payload),
                "P1_source": predictions[cell]["binding"], "prior_native_reference": references[cell]["binding"],
                "expectations": expectations, "expectations_sha256": config["expectations_sha256"],
                "response_contract": response_contract, "response_contract_sha256": config["response_contract_sha256"],
                "counts_before_seal": dict(counts)}
            bindings[cell] = frontier
            tx = ForecastTransaction(output / "transactions" / cell, model_identity=adapter.model_identity,
                input_identity=identity(frontier), source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": frontier},
                execution_scope="Previously sealed P1 installed into exact native N1 clock; R=F1(P1), native suffix1–3. Retrospective native corner.",
                budgets={"max_native_calls": 1, "max_bytes": config["transaction_max_bytes"], "max_wall_ms": config["max_wall_seconds"] * 1000})
            tx.frontier("source_binding", frontier)
            tx.predict("native_corner", lambda _, value=wide: {"predictions": {"P1": value, "expectations": expectations}})
            seals[cell] = tx.seal()
            txs[cell] = tx
            emit("condition_sealed", condition_id=cell, seal=seals[cell])
        save(output / "binding.json", bindings)
        cohort = {"schema": "mdb-native-corner-cohort-seal-v1", "conditions": CELLS, "seals": seals,
            "binding_sha256": sha(output / "binding.json"), "counts": dict(counts), "target_calls": dict(state["target_calls"])}
        save(output / "cohort-seal.json", {**cohort, "fingerprint": identity(cohort)})
        emit("cohort_sealed", cohort_sha256=sha(output / "cohort-seal.json"))
        for cell in CELLS:
            def execute(context, cell=cell):
                result = {}
                for role in ("N1.sham", "P1"):
                    wall()
                    name = cell + "." + role
                    parent = parents[cell]
                    session = Session.from_cut(adapter, parent)
                    require(session.capture(retain=False).fingerprint == parent.fingerprint, "hydrated source changed")
                    sealed = np.array(context.predictions["native_corner"]["P1"], copy=True)
                    value = parent.payload["latent"] if role == "N1.sham" else torch.from_numpy(sealed).to(parent.payload["latent"])
                    action = latent_act(parent, value, name)
                    applied = session.apply(action)
                    accept(applied, name)
                    entry = session.capture(retain=False)
                    require(describe(entry.payload) == describe({**parent.payload, "latent": action.parameters["value"]}),
                        "latent Act changed other source/clock/reset ports")
                    entry_record = capture(entry, True)
                    state.update(mode="native", cell=cell)
                    baseline = []
                    for position in range(84):
                        wall()
                        receipt = session.continue_()
                        accept(receipt, name)
                        cut = session.capture(retain=False)
                        logical, local = 1 + position // 28, position % 28
                        p = cut.payload
                        fields = ({"image": p["image"], "text": p["text"], **f["context_fields"](p["context"])} if local == 0 else
                            {"image": p["image"], "text": p["text"]} if local <= 25 else
                            {"noise": p["noise_prediction"]} if local == 26 else {"latent": p["latent"]})
                        keep = retained(logical, local)
                        values = {key: f["value_record"](value.detach().float().cpu().numpy(),
                            archive=array(value) if keep else None, witness_count=64) for key, value in fields.items()}
                        matches = {key: value["descriptor"] == references[cell]["transitions"][logical * 28 + local]["outputs"][key]["descriptor"]
                            for key, value in values.items()} if role == "N1.sham" else None
                        if matches is not None:
                            baseline.append({"logical_step": logical, "local_index": local, "field_hash_matches": matches})
                        record = {"branch": name, "position": position, "logical_step": logical, "local_index": local,
                            "receipt": receipt.to_dict(), "capture": capture(cut, keep), "outputs": values,
                            "native_replay_diagnostic": matches}
                        with (output / (name + ".jsonl")).open("a") as stream:
                            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        emit("native_transition", branch=name, position=position, logical_step=logical, local_index=local)
                    final = session.capture(retain=False)
                    require(final.payload["step"] == 4 and all(final.payload[key] is None for key in ("image", "text", "context", "noise_prediction")), "terminal native reset invalid")
                    state["mode"] = "vae"
                    counts["vae_calls"] += 1
                    image = adapter.decode(session, height=config["height"], width=config["width"])
                    image.save(output / (name + ".png"))
                    counts["images"] += 1
                    row = {"id": name, "condition_id": cell, "input_role": role, "source_parent": parent.fingerprint,
                        "entry": entry_record, "acts": [applied.to_dict()], "act": action.manifest(), "final": capture(final, True),
                        "final_latent": array(final.payload["latent"]), "image": {"path": name + ".png", "sha256": sha(output / (name + ".png"))},
                        "native_suffix_steps": 84, "native_replay_diagnostics": baseline,
                        "baseline_png_exact_prior": sha(output / (name + ".png")) == references[cell]["image"]["sha256"] if role == "N1.sham" else None}
                    save(output / "rows" / (name + ".json"), row)
                    rows[name] = row
                    result[name] = {"image": row["image"], "final_cut": final.fingerprint}
                    emit("native_vae", branch=name, image=row["image"])
                    del session
                return result
            txs[cell].execute("native_futures", execute)
            txs[cell].close()
            verify_transaction(txs[cell].root, expected_seal_sha256=seals[cell]["seal_sha256"], max_bytes=config["transaction_max_bytes"])
            save(output / "transactions" / cell / "report.json", old["util"]()["json_value"](txs[cell].report()))
        require(state["blocked"] == 0 and state["target_calls"] == target_counts()
            and all(v == target_counts(6) for v in state["per_condition"].values()), "actual native component census differs")
        for cell, cut in parents.items():
            cut.verify()
            require(identity(describe(cut.payload)) == parent_identities[cell], "immutable original parent payload changed")
        require(old["prior"]()["parameter_census"](guard_components) == parameters_before, "in-memory native parameters changed")
        expected = {**COUNTS, "model_loads": 0} if not production_profile else COUNTS
        require(counts == expected, "actual whole-lease physical census differs")
        source = {**source, "job_id": request["job_id"], "request_sha256": request["request_sha256"],
            "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity, "execution": dict(adapter.execution),
            "source_integrity": {"parent_identities_before_and_after": parent_identities,
                "native_parameter_census_before_and_after": parameters_before, "unchanged": True}}
        save(output / "source.json", source)
        summary = {"schema": "mdb-native-corner-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
            "counts": dict(counts), "target_calls": state["target_calls"], "target_calls_per_condition": state["per_condition"],
            "rows": rows, "seals": seals, "seconds": time.monotonic() - started,
            "mechanics_status": "pending-independent-audit", "instrument_status": "native-replay-diagnostics-retained",
            "scope": "New native F(P1) corner and three native remaining-pass futures. Previous P program was sealed earlier. Replay disagreements are per-branch diagnostics, not reasons to suppress changed futures."}
        save(output / "summary.json", summary)
        return summary
    finally:
        for handle in handles:
            handle.remove()


def worker():
    """Production bootstrap; all imports are installed public APIs/helpers."""
    import faulthandler
    import importlib.metadata

    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    started = time.monotonic()
    config, request = (json.loads(Path(os.environ[key]).read_text()) for key in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    require(request["config"] == config and all(sha(path) == digest for path, digest in request["source_hashes"].items()), "sealed request/source differs")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    helpers = runpy.run_path("audit_flux_prompt_factorial.py")
    collector = runpy.run_path("collect_flux_prompt_factorial.py")
    parent = Path("parent-run")
    parent_request = json.loads((parent / "request.json").read_text())
    parent_job = json.loads((parent / "collection-job.json").read_text())
    remote = Path(config["parent_remote_result"])
    require(str(remote) == collector["validate_job"](parent_job, config["parent_job_id"], parent_request), "parent remote/job binding differs")
    expected = json.loads((parent / "collection-local-inventory.json").read_text())
    require(collector["inventory"](remote) == expected, "authenticated original source inventory differs")
    data = parent / "extracted/flux-prompt-factorial-result"
    helpers["hardlink_inventory"](remote, data, expected)
    require(collector["inventory"](data) == expected, "local source hardlinks differ")
    collector["verify_manifest"](data, expected, parent_job["job_id"], parent_request)
    expectations = json.loads(Path(config["expectations_path"]).read_text())
    require(sha(config["expectations_path"]) == config["expectations_sha256"], "pre-native expectations changed")
    response_contract = json.loads(Path(config["response_contract_path"]).read_text())
    require(sha(config["response_contract_path"]) == config["response_contract_sha256"], "pre-native response contract changed")
    accepted_path = Path("parent-audit/audit.json")
    accepted = json.loads(accepted_path.read_text())
    require(sha(accepted_path) == config["parent_audit_sha256"] and accepted["mechanics_status"] == "valid"
        and accepted["near_data_execution"]["audit_job_id"] == config["parent_audit_job_id"]
        and accepted["source"]["job_id"] == config["parent_job_id"]
        and accepted["source"]["request_sha256"] == config["parent_request_sha256"], "accepted original audit binding differs")
    matched_path = Path(config.get("matched_prior_path", "matched-prior/result.json"))
    matched = json.loads(matched_path.read_text())
    require(sha(matched_path) == config["matched_result_sha256"]
        and matched["near_data_execution"]["job_id"] == config["matched_job_id"]
        and matched["report"]["source_job_id"] == config["parent_job_id"]
        and matched["report"]["source_request_sha256"] == config["parent_request_sha256"], "prior matched replay binding differs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction
    from mdb.tensor_archive import load_tensor_archive

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    require(torch.cuda.is_available(), "native-corner requires admitted CUDA")
    prior_source = json.loads((data / "source.json").read_text())
    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    require(identity(inventory) == config["expected_checkpoint_sha256"] == prior_source["checkpoint_sha256"], "checkpoint identity differs")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True, text_encoder=None, tokenizer=None)
    require(pipe.text_encoder is None and pipe.tokenizer is None, "encoding components must be absent")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = FluxFinalLatentClosureAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    require(adapter.model_identity == prior_source["model_identity"] and dict(adapter.execution) == prior_source["execution"], "native imported execution differs")
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in (type(pipe), type(pipe.transformer), type(pipe.vae),
        type(pipe.scheduler), type(adapter), Flux2KleinAdapter, type(pipe.transformer.single_transformer_blocks[0]),
        type(pipe.transformer.single_transformer_blocks[0].attn.processor), type(pipe.transformer.transformer_blocks[0]),
        type(pipe.transformer.norm_out), type(pipe.transformer.pos_embed), type(pipe.transformer.time_guidance_embed.timestep_embedder))}
    old_names = {Path(name).name: value for name, value in prior_source["implementation_sources"].items()}
    require(all(old_names.get(Path(name).name) == digest for name, digest in implementations.items()), "native implementation bytes differ")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}
    require(versions == prior_source["versions"], "native package versions differ")
    binding = json.loads((data / "binding.json").read_text())
    parent_summary = json.loads((data / "summary.json").read_text())
    parents, predictions, references = {}, {}, {}
    store = LocalStore(data / "typed-store")
    for cell in CELLS:
        native = [json.loads(line) for line in (data / "native" / (cell + ".jsonl")).read_text().splitlines()]
        parents[cell] = store.load(native[27]["capture"]["cut"], device="cuda")
        tx = data / "transactions" / cell
        verification = verify_transaction(tx, expected_seal_sha256=parent_summary["seals"][cell]["seal_sha256"], max_bytes=parent_request["config"]["transaction_max_bytes"])
        program = support()["prior"]()["util"]()["json_value"](load_forecast_tree(tx, json.loads((tx / "forecasts/program.json").read_text())["predictions"]))
        record = program["transitions"][27]["outputs"]["latent"]
        wide = load_tensor_archive(data, record["archive"])
        require(support()["descriptor"](wide) == record["descriptor"] and program["origin_cut"] == binding["origins"][cell]["cut"]
            and not np.any(wide.view(np.uint32) & np.uint32(65535)), "original sealed P1 binding differs")
        value = torch.from_numpy(np.array(wide, copy=True)).to(parents[cell].payload["latent"])
        require(np.array_equal(value.float().cpu().numpy().view(np.uint32), wide.view(np.uint32)), "P1 install conversion changed sealed values")
        predictions[cell] = {"latent": value, "binding": {"original_program_transition": 27, "record": record,
            "original_program_file_sha256": sha(tx / "forecasts/program.json"),
            "original_seal_sha256": parent_summary["seals"][cell]["seal_sha256"],
            "original_continuation_transition_range": [28, 111],
            "original_complete_forecast_trajectory_fingerprint": identity(program["transitions"]),
            "original_terminal_latent_descriptor": program["terminal_latent"]["descriptor"],
            "transaction": verification, "prior_predicted_png": parent_summary["rows"][cell + ".predicted"]["image"]}}
        references[cell] = {"transitions": native, "image": parent_summary["rows"][cell + ".native"]["image"],
            "binding": {"native_N1": native[27]["capture"], "prior_native_final": parent_summary["rows"][cell + ".native"]}}
    numeric_policy = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(model_loads=1, physical_batch=1)
    def progress(row):
        require(time.monotonic() - started <= config["max_wall_seconds"], "whole-worker native-corner budget exhausted")
        print("MDB_NATIVE_CORNER_PHASE " + json.dumps(row, sort_keys=True), flush=True)
    require(time.monotonic() - started <= config["max_wall_seconds"], "source loading exceeded worker budget")
    with torch.inference_mode():
        summary = run(adapter=adapter, parents=parents, predictions=predictions, references=references, output=Path(RESULT),
            config=config, request=request, source={"parent_job_id": parent_job["job_id"], "parent_inventory_sha256": sha(parent / "collection-local-inventory.json"),
                "implementation_sources": implementations, "versions": versions, "numeric_policy": numeric_policy,
                "parent_audit_sha256": sha(accepted_path), "matched_result_sha256": sha(matched_path)}, counts=counts,
            parameter_manifest=json.loads((data / "parameter-artifacts.json").read_text()), retrieval_source=data / "retrieve_weights.py",
            expectations=expectations, response_contract=response_contract, guard_components=(pipe.transformer, pipe.vae),
            progress=progress)
    require(collector["inventory"](remote) == expected == collector["inventory"](data), "original source changed during native work")
    manifest = support()["prior"]()["step"]()["finalize_manifest"](Path(RESULT), job_id=request["job_id"], request_sha256=request["request_sha256"],
        max_expanded_bytes=config["transport_limits"]["max_expanded_bytes"], max_member_bytes=config["transport_limits"]["max_member_bytes"])
    faulthandler.cancel_dump_traceback_later()
    print("MDB_NATIVE_CORNER_COMPLETE " + json.dumps({"summary": summary, "manifest": manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        if Path(RESULT).is_dir() and not (Path(RESULT) / "remote-manifest.json").exists():
            save(Path(RESULT) / "failure.json", {"error": type(error).__name__ + ": " + str(error),
                "scope": "Completed native rows retained; no automatic retry."})
        raise
