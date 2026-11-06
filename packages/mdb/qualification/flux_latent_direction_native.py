"""Native receiver × incoming-error direction cross; no predictor recomputation.

Six CPU-sealed mathematical finals are rendered using qualified derived-terminal
Acts before any denoiser entry. Six native84 futures then consume the exact CPU
input latents in original complete N1 parents. Scaffold derivations are metadata,
never native steps or execution receipts. Render and native-input Acts count.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import time
from functools import lru_cache
from pathlib import Path

RESULT = "latent-direction-native-result"
MATH_RESULT = "latent-direction-math-result"
CELLS = ["c1-o0", "c0-o1"]
BRANCHES = [cell + suffix for cell in CELLS for suffix in (".sham", ".d-c1-o0", ".d-c0-o1")]
CORNER_SHA256 = "cd54f51860149cb3236baa482ea7ff39070c7a8c90639ea38b80210b6dc845e3"
PARAMETER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
COUNTS = {"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0, "adapter_steps": 504, "acts": 12,
    "render_acts": 6, "native_branch_acts": 6, "receipt_emissions": 516, "images": 12, "vae_calls": 12,
    "predicted_images": 6, "native_images": 6, "candidate_calls": 0, "physical_batch": 1, "mathematical_forwards": 0}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@lru_cache(maxsize=1)
def support():
    path = Path(__file__).with_name("flux_native_corner.py")
    require(sha(path) == CORNER_SHA256, "qualified immutable native-corner helper differs")
    result = runpy.run_path(str(path))
    result["support"]()
    return result


def fixed_config():
    return {"schema": "mdb-latent-direction-native-plan-v1", "conditions": list(CELLS), "branch_order": list(BRANCHES),
        "logical_steps": [1, 2, 3], "height": 512, "width": 512, "steps": 4, "cpu_threads": 4, "witness_count": 64,
        "selector": {"logical_step": 1, "phase": "project", "index": None}, "counts": dict(COUNTS),
        "target_calls": support()["target_counts"](18), "target_calls_per_condition": support()["target_counts"](9),
        "observation_plan": {"pass1_landmark_locals": [0, 5, 25, 26, 27], "later_landmark_locals": [27],
            "full_landmark_transitions": 42, "descriptor_only_transitions": 462,
            "scope": "All504 receipt/manifests/field hashes/fixed64 witnesses; only42 declared exits hydratable."},
        "transaction_max_bytes": 64 << 20, "transaction_max_native_calls": 2, "max_wall_seconds": 1200,
        "parameter_storage": "checkpoint-recipe-v1", "array_storage": "bf16-bits",
        "transport_method": "planned-authenticated-ssh-manifest", "transport_limits": {"max_expanded_bytes": 4 << 30, "max_member_bytes": 1 << 30}}


def make_config(parent_config=None):
    result = {k: v for k, v in (parent_config or {}).items() if k.startswith("expected_") or k in ("model", "model_path")}
    result.update(fixed_config())
    return json.loads(json.dumps(result))


def validate_config(config):
    require(all(config.get(k) == v for k, v in fixed_config().items()), "finite latent-direction native contract differs")


def read_array(data, record):
    """Read only the named sealed value, with exact file/value descriptor checks."""
    import numpy as np

    from mdb.tensor_archive import load_tensor_archive

    f = support()["support"]()
    outer_descriptor = record.get("descriptor")
    record = record.get("archive", record)
    path = Path(record["path"])
    require(not path.is_absolute() and ".." not in path.parts and not (Path(data) / path).is_symlink(), "unsafe sealed array path")
    require(sha(Path(data) / path) == record["sha256"], "sealed array file changed")
    value = load_tensor_archive(data, record) if record.get("encoding") else np.load(Path(data) / path, allow_pickle=False)
    require(value.dtype == np.float32 and np.isfinite(value).all() and not np.any(value.view(np.uint32) & np.uint32(65535)),
        "sealed consumer value must be exact finite BF16 widening")
    descriptor = record.get("descriptor")
    if descriptor is not None and "fp32_le_sha256" in descriptor:
        require(f["descriptor"](value) == descriptor, "sealed array value descriptor differs")
    if outer_descriptor is not None and "fp32_le_sha256" in outer_descriptor:
        require(f["descriptor"](value) == outer_descriptor, "sealed output descriptor differs")
    shape = record.get("shape", record.get("stored_shape", descriptor["shape"] if descriptor else None))
    require(list(value.shape) == shape, "sealed array shape differs")
    return value


def load_math(data, *, parent_job_id, parent_request_sha256):
    data = Path(data)
    inputs, summary, cohort = [read(data / name) for name in ("inputs.json", "summary.json", "cohort-seal.json")]
    require(inputs["source"]["parent_job_id"] == parent_job_id and inputs["source"]["parent_request_sha256"] == parent_request_sha256,
        "CPU inputs refer to different original parent")
    require(set(inputs["receivers"]) == set(CELLS) and set(inputs["branches"]) == set(BRANCHES), "CPU input receiver/branch census differs")
    require(summary["math_transitions"] == cohort["math_transitions"] == 504 and summary["native_forwards"] == cohort["native_forwards"] == 0,
        "CPU mathematical/native census differs")
    require(summary["inputs_sha256"] == cohort["inputs_sha256"] == sha(data / "inputs.json")
        and summary["programs"] == cohort["programs"] and set(summary["programs"]) == set(BRANCHES)
        and summary["cohort_seal_sha256"] == sha(data / "cohort-seal.json")
        and cohort["fingerprint"] == identity({k: v for k, v in cohort.items() if k != "fingerprint"}), "CPU cohort seal/program identities differ")
    require(summary["expectations_sha256"] == cohort["expectations_sha256"] == sha(data / "expectations.json"), "pre-native expectations differ")
    programs = {}
    for branch in BRANCHES:
        ref = summary["programs"][branch]
        require(ref["path"] == "programs/" + branch + ".json" and sha(data / ref["path"]) == ref["sha256"], "sealed CPU program bytes differ")
        program = read(data / ref["path"])
        spec = inputs["branches"][branch]
        receiver = branch.split(".")[0]
        direction = None if branch.endswith(".sham") else branch.split(".d-")[1]
        require(spec["receiver"] == program["receiver"] == receiver and spec["direction"] == program["direction"] == direction
            and program["branch_id"] == branch and program["origin_cut"] == inputs["receivers"][receiver]["cut"]["cut"]
            and program["inputs"] == spec and program["chain_length"] == len(program["transitions"]) == 84, "CPU program source/branch/length differs")
        require([(r["logical_step"], r["local_index"]) for r in program["transitions"]] == [(1 + i // 28, i % 28) for i in range(84)],
            "CPU program clocks differ")
        require(program["terminal_latent"] == program["transitions"][-1]["outputs"]["latent"], "CPU terminal must equal actual final mathematical commit")
        programs[branch] = program
    return inputs, programs, {"summary_sha256": sha(data / "summary.json"), "inputs_sha256": sha(data / "inputs.json"),
        "cohort_seal_sha256": sha(data / "cohort-seal.json"), "programs": summary["programs"],
        "expectations_sha256": sha(data / "expectations.json")}


def run(*, adapter, parents, math_data, references, output, config, request, source, counts,
        parameter_manifest, retrieval_source, guard_components, encoder_components=(), progress=None, production_profile=True):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.flux_derived_terminal import derive_terminal
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    validate_config(config)
    corner = support()
    f, old = corner["support"](), corner["support"]()["prior"]()
    inputs, programs, math_binding = load_math(math_data, parent_job_id=config["parent_job_id"], parent_request_sha256=config["parent_request_sha256"])
    output, math_data = Path(output), Path(math_data)
    output.mkdir(parents=True, exist_ok=False)
    (output / "native").mkdir()
    started = time.monotonic()
    def wall():
        require(time.monotonic() - started <= config["max_wall_seconds"], "finite native wall budget exhausted")
    array = old["step"]()["make_array_writer"](output, wall)
    store = LocalStore(output / "typed-store")
    save(output / "plan.json", config)
    (output / "expectations.json").write_bytes((math_data / "expectations.json").read_bytes())
    require(set(parents) == set(references) == set(CELLS), "two original receiver sources required")
    require(sha(retrieval_source) == PARAMETER_SHA256 and parameter_manifest["artifacts"]
        and (not production_profile or len(parameter_manifest["artifacts"]) == 169), "qualified recipe-only constants required")
    recipe = json.loads(json.dumps(parameter_manifest))
    recipe["provenance"] = {"original": recipe.get("provenance"), "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "producer_source_sha256": sha(__file__), "math": math_binding, "parameter_files_written": 0}
    save(output / "parameter-artifacts.json", recipe)
    (output / "retrieve_weights.py").write_bytes(Path(retrieval_source).read_bytes())
    model = adapter.model
    require(len(model.transformer_blocks) == 5 and len(model.single_transformer_blocks) == 20, "native block geometry differs")
    before = {}
    for cell, cut in parents.items():
        cut.verify()
        require(cut.fingerprint == inputs["receivers"][cell]["cut"]["cut"] and cut.model_identity == adapter.model_identity
            and dict(cut.execution) == dict(adapter.execution), "receiver original complete cut/model/execution differs")
        corner["latent_act"](cut, cut.payload["latent"], "receiver-preflight")
        if production_profile:
            require(cut.payload["latent"].shape == (1, 1024, 128) and cut.payload["conditioning"].shape == (1, 512, 7680)
                and cut.payload["latent"].device.type == "cuda", "production receiver geometry/backend differs")
        before[cell] = identity(describe(cut.payload))
        store.save(cut)
    theta = old["prior"]()["parameter_census"](guard_components)
    state = {"mode": "sealed", "cell": None, "blocked": 0, "target_calls": dict.fromkeys(corner["target_counts"](), 0),
        "per_condition": {cell: dict.fromkeys(corner["target_counts"](), 0) for cell in CELLS}}
    targets = {**{m: str(i) for i, m in enumerate(model.single_transformer_blocks)},
        **{m: "joint" + str(i) for i, m in enumerate(model.transformer_blocks)}, **{getattr(model, name): name for name in
            ("norm_out", "proj_out", "x_embedder", "context_embedder", "time_guidance_embed", "double_stream_modulation_img",
             "double_stream_modulation_txt", "single_stream_modulation", "pos_embed")}}
    vae_ids = {id(m) for c in guard_components if c is not model for m in c.modules()}
    encoder_ids = {id(m) for c in encoder_components for m in c.modules()}
    def guard(module, args):
        if id(module) in encoder_ids or state["mode"] == "sealed" or (state["mode"] == "vae" and id(module) not in vae_ids):
            state["blocked"] += 1
            raise ValueError("native component entered outside sealed/native/VAE scope; encoding forbidden")
        if module in targets:
            key = targets[module]
            state["target_calls"][key] += 1
            state["per_condition"][state["cell"]][key] += 1
    handles = [m.register_forward_pre_hook(guard) for c in (*guard_components, *encoder_components) for m in c.modules()]
    def emit(kind, **fields):
        body = {"kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields}
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True) + "\n")
        if progress:
            progress(body)
    def capture(cut, keep):
        body = cut.manifest()
        require(identity(body) == cut.fingerprint, "typed cut descriptor differs")
        if keep:
            store.save(cut)
        return {"cut": cut.fingerprint, "point": dict(cut.execution_point), "manifest": body, "retained_full_cut": keep}
    def accept(receipt, branch, kind):
        body = receipt.to_dict()
        if body["operation"] == "continue":
            counts["adapter_steps"] += body.get("steps", 1)
        else:
            counts["acts"] += 1
            counts[kind] += 1
        counts["receipt_emissions"] += 1
        store.receipt(receipt)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence": counts["receipt_emissions"] - 1, "branch": branch, "kind": kind, "receipt": receipt.fingerprint}) + "\n")
    def tensor(record, parent):
        wide = read_array(math_data, record)
        value = torch.from_numpy(np.array(wide, copy=True)).to(parent.payload["latent"])
        require(np.array_equal(value.float().cpu().numpy().view(np.uint32), wide.view(np.uint32)), "sealed value cast changed bits")
        return value
    txs, seals, rows, predicted_images = {}, {}, {}, {}
    try:
        for cell in CELLS:
            selected = {name: {"program": math_binding["programs"][name], "input": inputs["branches"][name],
                "terminal_latent": programs[name]["terminal_latent"]} for name in BRANCHES if inputs["branches"][name]["receiver"] == cell}
            frontier = {"receiver": cell, "native_N1": capture(parents[cell], True), "source_payload": describe(parents[cell].payload),
                "CPU_math": math_binding, "all_six_programs": math_binding["programs"], "selected": selected,
                "expectations": read(math_data / "expectations.json"),
                "counts_before_seal": dict(counts), "target_calls_before_seal": dict(state["target_calls"])}
            tx = ForecastTransaction(output / "transactions" / cell, model_identity=adapter.model_identity,
                input_identity=identity(frontier), source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": frontier},
                execution_scope="CPU-sealed84 own-state math futures; predicted VAE first, then exact input native84; no predictor recomputation.",
                budgets={"max_native_calls": 2, "max_executions": 2, "max_bytes": config["transaction_max_bytes"], "max_wall_ms": config["max_wall_seconds"] * 1000})
            tx.frontier("source_binding", frontier)
            tx.predict("program_bindings", lambda _, v=selected: {"predictions": v})
            seals[cell] = tx.seal()
            txs[cell] = tx
            emit("receiver_sealed", receiver=cell, seal=seals[cell])
        save(output / "binding.json", {"inputs": inputs, "CPU_math": math_binding})
        cohort = {"schema": "mdb-latent-direction-native-cohort-seal-v1", "seals": seals,
            "binding_sha256": sha(output / "binding.json"), "counts": dict(counts), "target_calls": state["target_calls"]}
        save(output / "cohort-seal.json", {**cohort, "fingerprint": identity(cohort)})
        emit("cohort_sealed", cohort_sha256=sha(output / "cohort-seal.json"))
        for cell in CELLS:
            def render(context, cell=cell):
                result = {}
                for name in BRANCHES:
                    if inputs["branches"][name]["receiver"] != cell:
                        continue
                    wall()
                    parent = parents[cell]
                    derived = derive_terminal(adapter, parent, predicted_latent=tensor(programs[name]["terminal_latent"], parent), name=name + ".render")
                    session = Session.from_cut(adapter, derived.scaffold)
                    store.save(derived.scaffold)
                    accept(session.apply(derived.action), name, "render_acts")
                    final = session.capture(retain=False)
                    require(all(v == 0 for v in state["target_calls"].values()) and counts["adapter_steps"] == 0, "denoiser entered before all math images")
                    state.update(mode="vae", cell=cell)
                    png = adapter.decode(session, height=config["height"], width=config["width"])
                    path = output / (name + ".predicted.png")
                    png.save(path)
                    counts["vae_calls"] += 1
                    counts["images"] += 1
                    counts["predicted_images"] += 1
                    require(all(v == 0 for v in state["target_calls"].values()), "denoiser entered during math render")
                    derivation = {"qualified_derivation": derived.derivation, "CPU_program": math_binding["programs"][name],
                        "CPU_cohort_sha256": math_binding["cohort_seal_sha256"], "source_N1": parent.fingerprint,
                        "sealed_terminal_latent": programs[name]["terminal_latent"], "actual_render_cut": capture(final, True)}
                    save(output / "derivations" / (name + ".json"), derivation)
                    result[name] = {"image": {"path": path.name, "sha256": sha(path)}, "derivation": "derivations/" + name + ".json",
                        "act": derived.action.manifest(), "final": capture(final, False), "native_steps_before_image": 0,
                        "target_calls_after_image": dict(state["target_calls"])}
                    predicted_images[name] = result[name]
                    emit("predicted_image", branch=name, image=result[name]["image"], target_calls=dict(state["target_calls"]))
                    state["mode"] = "sealed"
                return result
            txs[cell].execute("predicted_images", render)
        require(counts["predicted_images"] == counts["render_acts"] == 6 and counts["adapter_steps"] == 0, "all six predicted images must precede denoising")
        emit("all_predicted_images_complete", target_calls=dict(state["target_calls"]))
        for cell in CELLS:
            def native(context, cell=cell):
                result = {}
                for name in BRANCHES:
                    spec = inputs["branches"][name]
                    if spec["receiver"] != cell:
                        continue
                    wall()
                    parent = parents[cell]
                    value = tensor(spec["input_latent"], parent)
                    if name.endswith(".sham"):
                        require(torch.equal(value, parent.payload["latent"]), "CPU sham differs from native N1")
                    session = Session.from_cut(adapter, parent)
                    action = corner["latent_act"](parent, value, name)
                    accept(session.apply(action), name, "native_branch_acts")
                    entry = session.capture(retain=False)
                    require(describe(entry.payload) == describe({**parent.payload, "latent": action.parameters["value"]}), "input Act changed carried/reset/clock ports")
                    entry_record = capture(entry, True)
                    state.update(mode="native", cell=cell)
                    replay = []
                    for position in range(84):
                        wall()
                        receipt = session.continue_()
                        accept(receipt, name, "native_transition")
                        cut = session.capture(retain=False)
                        logical, local = 1 + position // 28, position % 28
                        p = cut.payload
                        fields = ({"image": p["image"], "text": p["text"], **f["context_fields"](p["context"])} if local == 0 else
                            {"image": p["image"], "text": p["text"]} if local <= 25 else
                            {"noise": p["noise_prediction"]} if local == 26 else {"latent": p["latent"]})
                        keep = corner["retained"](logical, local)
                        outputs = {key: f["value_record"](value.detach().float().cpu().numpy(),
                            archive=array(value) if keep else None, witness_count=64) for key, value in fields.items()}
                        diagnostic = None
                        if name.endswith(".sham"):
                            diagnostic = {key: ref["descriptor"] == references[cell]["transitions"][28 + position]["outputs"][key]["descriptor"] for key, ref in outputs.items()}
                            replay.append(diagnostic)
                        row = {"position": position, "logical_step": logical, "local_index": local, "receipt": receipt.to_dict(),
                            "capture": capture(cut, keep), "outputs": outputs, "native_replay_diagnostic": diagnostic}
                        with (output / "native" / (name + ".jsonl")).open("a") as stream:
                            stream.write(json.dumps(row, sort_keys=True) + "\n")
                        emit("native_transition", branch=name, logical_step=logical, local_index=local)
                    final = session.capture(retain=False)
                    require(final.payload["step"] == 4 and all(final.payload[k] is None for k in ("image", "text", "context", "noise_prediction")), "native terminal reset differs")
                    state["mode"] = "vae"
                    image = adapter.decode(session, height=config["height"], width=config["width"])
                    path = output / (name + ".native.png")
                    image.save(path)
                    counts["images"] += 1
                    counts["vae_calls"] += 1
                    counts["native_images"] += 1
                    result[name] = {"id": name, "receiver": cell, "direction": spec["direction"], "source_parent": parent.fingerprint,
                        "act": action.manifest(), "entry": entry_record, "native_suffix_steps": 84, "final": capture(final, True),
                        "final_latent": array(final.payload["latent"]), "image": {"path": path.name, "sha256": sha(path)},
                        "predicted": predicted_images[name], "native_replay_diagnostics": replay,
                        "baseline_png_exact_prior": sha(path) == references[cell]["image"]["sha256"] if name.endswith(".sham") else None}
                    save(output / "rows" / (name + ".json"), result[name])
                    rows[name] = result[name]
                    state["mode"] = "sealed"
                return result
            txs[cell].execute("native_futures", native)
        for cell in CELLS:
            txs[cell].close()
            save(output / "transactions" / cell / "verification.json", verify_transaction(output / "transactions" / cell,
                expected_seal_sha256=seals[cell]["seal_sha256"], max_bytes=config["transaction_max_bytes"]))
            save(output / "transactions" / cell / "report.json", old["util"]()["json_value"](txs[cell].report()))
        require(state["blocked"] == 0 and state["target_calls"] == config["target_calls"]
            and all(v == config["target_calls_per_condition"] for v in state["per_condition"].values()), "actual native component census differs")
        for cell, cut in parents.items():
            cut.verify()
            require(identity(describe(cut.payload)) == before[cell], "immutable original receiver changed")
        require(old["prior"]()["parameter_census"](guard_components) == theta, "native parameter values changed")
        require(counts == (COUNTS if production_profile else {**COUNTS, "model_loads": 0}), "physical504/12Acts/516receipt/12VAE census differs")
        save(output / "source.json", {**source, "job_id": request["job_id"], "request_sha256": request["request_sha256"],
            "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity, "execution": dict(adapter.execution),
            "CPU_math": math_binding, "source_integrity": {"parent_identities_before_and_after": before, "native_parameter_census_before_and_after": theta, "unchanged": True}})
        summary = {"schema": "mdb-latent-direction-native-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
            "counts": dict(counts), "rows": rows, "seals": seals, "CPU_math": math_binding, "seconds": time.monotonic() - started,
            "target_calls": state["target_calls"], "target_calls_per_condition": state["per_condition"], "mechanics_status": "pending-independent-audit",
            "scope": "Six CPU math finals rendered before native0; exact CPU inputs then six native84 futures. Replay differences retained, not scientific gates."}
        save(output / "summary.json", summary)
        return summary
    finally:
        for handle in handles:
            handle.remove()


def finalize_worker_output(output, *, summary, request, config):
    """Production export tail, shared with the tiny completed-run check."""
    manifest = support()["support"]()["prior"]()["step"]()["finalize_manifest"](Path(output),
        job_id=request["job_id"], request_sha256=request["request_sha256"],
        max_expanded_bytes=config["transport_limits"]["max_expanded_bytes"],
        max_member_bytes=config["transport_limits"]["max_member_bytes"])
    print("MDB_LATENT_DIRECTION_NATIVE_RESULT=" + json.dumps({"summary": summary, "manifest": manifest}, sort_keys=True), flush=True)
    return manifest


def worker():
    """Production bootstrap through installed public packages; no encoders/math."""
    import faulthandler
    import importlib.metadata

    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    started = time.monotonic()
    config, request = (read(os.environ[key]) for key in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    require(request["config"] == config and all(sha(path) == digest for path, digest in request["source_hashes"].items()), "sealed source/request differs")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    launcher = runpy.run_path("submit_flux_latent_direction_native.py")
    parent, math_run = Path("parent-run"), Path("math-run")
    metadata = launcher["validate_sources"](parent, math_run, audit=Path("parent-audit"))
    require(all(metadata[key] == config[key] for key in metadata), "staged CPU/original source identities differ")
    c = launcher["C"]
    hardlink = runpy.run_path("audit_flux_prompt_factorial.py")["hardlink_inventory"]
    locations = [(parent, Path(config["parent_remote_result"]), "flux-prompt-factorial-result"),
        (math_run, Path(config["math_remote_result"]), MATH_RESULT)]
    snapshots = []
    for directory, remote, basename in locations:
        expected = read(directory / "collection-local-inventory.json")
        require(c["inventory"](remote) == expected, "authenticated remote source inventory differs")
        local = directory / "extracted" / basename
        hardlink(remote, local, expected)
        require(c["inventory"](local) == expected, "source hardlinks differ")
        snapshots.append((remote, local, expected))
    data, math_data = locations[0][0] / "extracted/flux-prompt-factorial-result", locations[1][0] / "extracted" / MATH_RESULT
    launcher["verify_math_manifest"](math_data, snapshots[1][2], read(math_run / "job.json")["job_id"], read(math_run / "request.json"))
    c["verify_manifest"](data, snapshots[0][2], config["parent_job_id"], read(parent / "request.json"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb import flux_derived_terminal
    from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter

    require(sha(flux_derived_terminal.__file__) == config["expected_derived_terminal_module_sha256"], "qualified derivation source differs")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    require(torch.cuda.is_available(), "admitted CUDA required")
    prior_source = read(data / "source.json")
    root = Path(config["model_path"])
    checkpoint = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    require(identity(checkpoint) == config["expected_checkpoint_sha256"] == prior_source["checkpoint_sha256"], "checkpoint source differs")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True, text_encoder=None, tokenizer=None)
    require(pipe.text_encoder is None and pipe.tokenizer is None, "encoder components must be absent")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = FluxFinalLatentClosureAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    require(adapter.model_identity == prior_source["model_identity"] and dict(adapter.execution) == prior_source["execution"], "original native model/execution differs")
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in (type(pipe), type(pipe.transformer), type(pipe.vae),
        type(pipe.scheduler), type(adapter), Flux2KleinAdapter, type(pipe.transformer.single_transformer_blocks[0]),
        type(pipe.transformer.single_transformer_blocks[0].attn.processor), type(pipe.transformer.transformer_blocks[0]),
        type(pipe.transformer.norm_out), type(pipe.transformer.pos_embed), type(pipe.transformer.time_guidance_embed.timestep_embedder))}
    original_names = {Path(name).name: digest for name, digest in prior_source["implementation_sources"].items()}
    require(all(original_names.get(Path(name).name) == digest for name, digest in implementations.items()), "native implementation source differs")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}
    require(versions == prior_source["versions"], "native package versions differ")
    parents, references = {}, {}
    store = LocalStore(data / "typed-store")
    for cell in CELLS:
        native = [json.loads(line) for line in (data / "native" / (cell + ".jsonl")).read_text().splitlines()]
        parents[cell] = store.load(native[27]["capture"]["cut"], device="cuda")
        references[cell] = {"transitions": native, "image": read(data / "summary.json")["rows"][cell + ".native"]["image"]}
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(model_loads=1, physical_batch=1)
    with torch.inference_mode():
        summary = run(adapter=adapter, parents=parents, math_data=math_data, references=references, output=Path(RESULT), config=config,
            request=request, source={"checkpoint_sha256": identity(checkpoint), "checkpoint_files": checkpoint,
                "implementation_sources": implementations, "versions": versions, "source_custody": metadata,
                "numeric_profile": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}},
            counts=counts, parameter_manifest=read(math_data / "parameter-artifacts.json"), retrieval_source=math_data / "retrieve_weights.py",
            guard_components=(pipe.transformer, pipe.vae), progress=lambda event: (
                require(time.monotonic() - started <= config["max_wall_seconds"], "whole-worker finite wall exhausted"),
                print(json.dumps(event, sort_keys=True), flush=True)))
    for remote, local, expected in snapshots:
        require(c["inventory"](remote) == c["inventory"](local) == expected, "CPU/original source changed during native execution")
    finalize_worker_output(Path(RESULT), summary=summary, request=request, config=config)


if __name__ == "__main__":
    try:
        worker()
    except BaseException as exc:
        output = Path(RESULT)
        if output.is_dir() and not (output / "failure.json").exists():
            save(output / "failure.json", {"error_type": type(exc).__name__, "error": str(exc), "scope": "Failed finite attempt; no retry or numeric result rewriting."})
        raise
