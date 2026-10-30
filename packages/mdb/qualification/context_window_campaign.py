"""Seven fresh first-step context contrasts, with three separately bound prior rows.

Trusted Python transaction callbacks; no native suffix is consulted to forecast.
Staging submits one finite CPU lease. This module never advances an instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path

FRESH = ("old.native", "old.zero", "old.U_swap", "old.U_swap_P2native", "new.zero",
         "old.U_swap_P2other", "new.U_swap_P2other")
PRIOR = ("new.native", "new.U_swap", "new.U_swap_P2native")
HEADS = ("regs", "flags", "dest", "src", "width", "length", "syscall", "imm",
         "store", "store_addr", "store_value")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def support():
    path = Path(__file__).with_name("campaign_support.py")
    return runpy.run_path(str(path if path.exists() else path.with_name("discovery_upgrades.py")))


def validate_plan(plan):
    if (plan["schema"] != "mdb-context-window-plan-v1"
            or tuple(plan["fresh_arms"]) != FRESH or tuple(plan["prior_arms"]) != PRIOR
            or plan["register_words"] != [0, 0, 0, 0, 0, 0, 3, 262]
            or plan["flags"] != [0, 0, 0, 0]
            or plan["execution"] != {"backend": "cpu", "dtype": "float32", "threads": 1}
            or plan["budgets"]["expected_physical_forwards"] != 7
            or plan["budgets"]["max_physical_forwards"] != 12
            or plan["budgets"]["max_replay_operations"] != 16000
            or plan["budgets"]["retries"] != 0
            or plan["budgets"]["timeout_seconds"] != 300):
        raise ValueError("plan does not describe the bounded seven-call assay")
    windows = {key: bytes.fromhex(plan["contexts"][key]["program_hex"])[:15].ljust(15, b"\0")
               for key in ("old", "new")}
    first = bytes.fromhex(plan["first_instruction_hex"])
    if first != bytes.fromhex("4881ef01000000") or any(not value.startswith(first) for value in windows.values()):
        raise ValueError("first SUB identity differs")
    if [i for i in range(15) if windows["old"][i] != windows["new"][i]] != list(range(8, 14)):
        raise ValueError("context contrast differs from the frozen trailing-byte comparison")
    return windows


def array_references(value):
    """Metadata traversal only: this function never loads an array."""
    if isinstance(value, dict):
        if value.get("kind") == "array":
            yield value
        else:
            for child in value.values():
                yield from array_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from array_references(child)


def decode_tree(value, root):
    """Decode a selected prior numeric subtree only after the new global seal."""
    import numpy as np

    kind = value["kind"]
    if kind == "array":
        relative = Path(value["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "arrays":
            raise ValueError("invalid prior array path")
        path = root / relative
        if path.is_symlink() or sha(path.read_bytes()) != value["sha256"]:
            raise ValueError("prior array bytes differ")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
            raise ValueError("prior array geometry differs")
        if sha(array.tobytes()) != value["content_sha256"] or not np.isfinite(array).all():
            raise ValueError("prior array content differs")
        return array.copy()
    if kind == "mapping":
        return {key: decode_tree(child, root) for key, child in value["items"].items()}
    if kind in ("tuple", "list"):
        items = [decode_tree(child, root) for child in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "scalar":
        return value["value"]
    raise ValueError("unsupported prior tree kind")


def stage_prior(root, plan, destination):
    """Verify external pins and select bytes without interpreting numeric arrays."""
    from mdb.forecast_transaction import verify_transaction

    verified = verify_transaction(root / "transaction", expected_seal_sha256=plan["prior"]["forecast_seal"],
                                  expected_event_head=plan["prior"]["event_head"])
    summary = json.loads((root / "summary.json").read_text())
    if summary["job_id"] != plan["prior"]["job_id"]:
        raise ValueError("prior job identity differs")
    source = json.loads((root / "source.json").read_text())
    if summary["request_sha256"] != source["request"]["request_sha256"]:
        raise ValueError("prior request identity differs")
    inventory = json.loads((root / "files.json").read_text())
    observation = json.loads((root / "transaction/observations/native.json").read_text())
    forecasts = json.loads((root / "transaction/forecasts/campaign.json").read_text())
    row_ids = {name: plan["prior"]["case_id"] + "." + name.split(".", 1)[1] + ".0" for name in PRIOR}
    selected = {name: observation["value"]["items"][row] for name, row in row_ids.items()}
    selected_payloads = {name: forecasts["payloads"][plan["prior"]["case_id"] + ".U_swap." + name]
                         for name in ("post0_zero", "U_native", "P2_native")}
    paths = {"summary.json", "source.json", "transaction-report.json", "files.json",
             "transaction/seal.json", "transaction/observations/native.json", "transaction/forecasts/campaign.json"}
    for descriptor in array_references([selected, selected_payloads]):
        paths.add("transaction/" + descriptor["path"])
    for row in row_ids.values():
        paths.update("captures/" + row + "/" + name for name in ("capture.json", "trace.json", "roots.pt"))
    # Bind extracted selected bytes to the externally pinned archive, rather than
    # treating an editable files.json or collector report as archive authority.
    import base64
    import io
    import tarfile

    marker = "MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 "
    log_path = root.parent.parent / "worker.log"
    encoded = [line[len(marker):].strip() for line in log_path.read_text().splitlines() if line.startswith(marker)]
    if len(encoded) != 1:
        raise ValueError("prior collector log must retain exactly one archive")
    archive_raw = base64.b64decode(encoded[0], validate=True)
    if len(archive_raw) > 64 * 1024 * 1024 or sha(archive_raw) != plan["prior"]["archive_sha256"]:
        raise ValueError("prior archive differs from the externally frozen pin")
    archive_hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
        for relative in paths:
            member = archive.getmember(root.name + "/" + relative)
            if not member.isfile() or member.size > 64 * 1024 * 1024:
                raise ValueError("invalid selected prior archive member")
            archive_hashes[relative] = sha(archive.extractfile(member).read())
    files = []
    hashes = {}
    for relative in sorted(paths):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("prior input must be a regular file: " + relative)
        digest = sha(path.read_bytes())
        if digest != archive_hashes[relative]:
            raise ValueError("selected prior bytes differ from pinned archive: " + relative)
        if relative != "files.json" and inventory.get(relative) != digest:
            raise ValueError("prior published inventory differs: " + relative)
        hashes[relative] = digest
        files.append((path, "prior/" + relative))
    binding = {"schema": "mdb-context-window-prior-binding-v1", "pins": plan["prior"],
               "row_ids": row_ids, "request_sha256": summary["request_sha256"],
               "summary_sha256": hashes["summary.json"], "verification": verified,
               "archive_verified": True, "archive_sha256": sha(archive_raw),
               "selected_payloads": selected_payloads, "files": hashes,
               "numeric_read_policy": "selected numeric arrays decoded only after the new global seal"}
    save(destination, binding)
    files.append((destination, "prior/binding.json"))
    return files, binding


def stage_specs():
    specs = {"z": ("module:decoder:return", None), "input": ("module:inp:return", None),
             "query": ("module:query:return", None), "offset_head": ("module:offset_head:return", None)}
    for layer in range(3):
        modules = {f"mixers.{layer}": "M", f"mixers.{layer}.in_proj": "in_projected",
                   f"mixers.{layer}.select": "selected", f"mixers.{layer}.dt_proj": "dt_affine",
                   f"mixers.{layer}.out_proj": "U", f"post.{layer}": "P2",
                   f"post.{layer}.0": "post2_norm", f"post.{layer}.1": "post2_affine",
                   f"post.{layer}.2": "post2_gelu", f"post.{layer}.3": "P2_computed"}
        specs.update({f"layer{layer}." + stage: ("module:" + module + ":return", None)
                      for module, stage in modules.items()})
    specs["layer2.S"] = ("module:mixers.2:input", ["args", 0])
    specs["layer2.h"] = ("module:reg_out:input", ["args", 0])
    return specs


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery_cli import load_capture, save_capture
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.forecast_transaction import ForecastPayload, ForecastTransaction, verify_transaction
    from mdb.forward_ports import NativePortCatalog
    from mdb.forward_recording import record_forward, tensor_descriptor

    started = time.monotonic()
    timings = []

    def phase(name, event, **detail):
        row = {"phase": name, "event": event, "elapsed_seconds": time.monotonic() - started, **detail}
        timings.append(row)
        print("MDB_CONTEXT_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    formulas = runpy.run_path("raw_isa_equations.py")
    raw_plan = Path("plan.json").read_bytes()
    plan = json.loads(raw_plan)
    validate_plan(plan)
    raw_hypothesis = Path("context-prefix-hypothesis.json").read_bytes()
    hypothesis = json.loads(raw_hypothesis)
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if sha(raw_plan) != config["plan_file_sha256"]:
        raise ValueError("plan bytes differ")
    if sha(raw_hypothesis) != config["hypothesis_file_sha256"]:
        raise ValueError("structural prefix hypothesis bytes differ")
    for name, digest in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != digest:
            raise ValueError("staged source differs: " + name)
    distribution = importlib.metadata.distribution("isa-model-comparison")
    for name, digest in formulas["SOURCE_FORMULAS"].items():
        if sha(distribution.locate_file(name).read_bytes()) != digest:
            raise ValueError("installed ISA source differs: " + name)
    torch.set_num_threads(1)
    phase("model_load", "start")
    model, model_sources = load_r3_mamba_model(config["checkpoint"])
    model.eval().requires_grad_(False)
    if (model.arch != "mamba" or not model.aligned or not model.store or model.width != 48
            or len(model.mixers) != 3 or len(model.post) != 3
            or any(mixer.state != 4 or mixer.kernel != 3 for mixer in model.mixers)):
        raise ValueError("native configuration differs from the frozen source")
    phase("model_load", "end")
    output = Path("context-window-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", plan)
    with (output / "context-prefix-hypothesis.json").open("xb") as stream:
        stream.write(raw_hypothesis)
    save(output / "source.json", {"request": request, "model_sources": model_sources})
    binding = json.loads(Path("prior/binding.json").read_text())
    save(output / "prior-binding.json", binding)
    # Retain the selected original evidence bytes for offline independent audit.
    for relative in binding["files"]:
        destination = output / "prior" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((Path("prior") / relative).read_bytes())
    counters = {"physical_forwards": 0, "replay_operations": 0, "acts": 0, "prior_rows_reused": 0}
    transitions, native_events = [], []

    def count_native(module, args):
        counters["physical_forwards"] += 1
        native_events.append({"kind": "native_root_entry", "index": counters["physical_forwards"],
                              "elapsed_seconds": time.monotonic() - started})
        if counters["physical_forwards"] > plan["budgets"]["max_physical_forwards"]:
            raise ValueError("physical forward budget exhausted")

    count_hook = model.register_forward_pre_hook(count_native)

    def predict(context):
        phase("prediction_compute", "start", native_calls=counters["physical_forwards"])
        weights = formulas["prepare_weights"](context.weights)
        rows, payloads = {}, {}
        roots = {key: formulas["roots"](plan["contexts"][key]["program_hex"], plan["register_words"], plan["flags"])
                 for key in ("old", "new")}
        originals = {key: formulas["predict_raw"](*values, weights) for key, values in roots.items()}
        frozen_u = {key: np.asarray(value["stages"]["layer2.U"], dtype=np.float32).copy()
                    for key, value in originals.items()}
        frozen_p2 = {key: np.asarray(value["stages"]["layer2.P2"], dtype=np.float32).copy()
                     for key, value in originals.items()}
        zero = np.zeros((1, 64, 48), dtype=np.float32)
        swaps = {key: formulas["predict_raw"](*values, weights, zero_post0=True, projection_override=frozen_u[key])
                 for key, values in roots.items()}
        donor_p2 = {key: np.asarray(value["stages"]["layer2.P2"], dtype=np.float32).copy()
                    for key, value in swaps.items()}
        payloads["post0_zero"] = ForecastPayload(zero, dtype="float32", provenance={"recipe": "zero post.0 return"})
        for key in ("old", "new"):
            for name, stage in (("U_native", "layer2.U"), ("P2_native", "layer2.P2")):
                payloads[key + "_" + name] = ForecastPayload(originals[key]["stages"][stage], dtype="float32",
                    provenance={"forecast_path": ["rows", key + ".native", "stages", stage],
                                "recipe": "same-context independent native equation endpoint; one FP32 payload cast"})
            payloads[key + "_P2_U"] = ForecastPayload(swaps[key]["stages"]["layer2.P2"], dtype="float32",
                provenance={"forecast_path": ["rows", key + ".U_swap", "stages", "layer2.P2"],
                            "recipe": "donor zero post0 + frozen same-context native U; one FP32 P2 cast"})
        for row_id in (*FRESH, *PRIOR):
            key, arm = row_id.split(".", 1)
            if arm == "native":
                predicted = originals[key]
            elif arm == "U_swap":
                predicted = swaps[key]
            else:
                kwargs = {"zero_post0": True}
                if arm != "zero":
                    kwargs["projection_override"] = frozen_u[key]
                if arm == "U_swap_P2native":
                    kwargs["post_override"] = frozen_p2[key]
                elif arm == "U_swap_P2other":
                    kwargs["post_override"] = donor_p2["new" if key == "old" else "old"]
                predicted = formulas["predict_raw"](*roots[key], weights, **kwargs)
            rows[row_id] = {"heads": predicted["heads"],
                "stages": {name: predicted["stages"][name] for name in stage_specs()},
                "register_terms": predicted["register_terms"], "decoded": predicted["decoded"],
                "roots": dict(zip(("window", "regs", "flags"), roots[key], strict=True)),
                "scope": "fresh prospective" if row_id in FRESH else "retrospective source-bound comparison"}
        phase("prediction_compute", "end", native_calls=counters["physical_forwards"])
        phase("forecast_publication", "start", scope="validation, cloning, encoding, hashing and durable publication combined")
        return {"predictions": {"rows": rows}, "payloads": payloads}

    def advance(runtime, pc, row_id):
        while runtime.pc < pc:
            if counters["replay_operations"] >= plan["budgets"]["max_replay_operations"]:
                raise ValueError("replay operation budget exhausted")
            transition = runtime.advance_one()
            transitions.append({"branch": row_id, "receipt": transition["receipt"]})
            counters["replay_operations"] += 1

    def changes(row_id, payloads):
        key, arm = row_id.split(".", 1)
        result = []
        if arm != "native":
            result.append(("module:post.0:return", "post0_zero"))
        if arm.startswith("U_swap"):
            result.append(("module:mixers.2.out_proj:return", key + "_U_native"))
        if arm == "U_swap_P2native":
            result.append(("module:post.2:return", key + "_P2_native"))
        if arm == "U_swap_P2other":
            result.append(("module:post.2:return", ("new" if key == "old" else "old") + "_P2_U"))
        return [(selector, name, payloads[name]) for selector, name in result]

    def actual_stages(ports, read):
        stages, references = {}, {}
        for name, (selector, path) in stage_specs().items():
            descriptor = ports.resolve(selector, path=path)
            stages[name] = read(descriptor["value_id"]).detach().cpu().numpy().copy()
            references[name] = descriptor
        return stages, references

    def execute(context):
        phase("native_execution", "start")
        payloads = context.payloads["campaign"]
        predicted = context.predictions["campaign"]["rows"]
        # This is the first numeric interpretation of any prior observation/payload.
        prior_metadata = json.loads(Path("prior/transaction/observations/native.json").read_text())
        for name, descriptor in binding["selected_payloads"].items():
            previous = decode_tree(descriptor["value"], Path("prior/transaction"))
            current_name = "post0_zero" if name == "post0_zero" else "new_" + name
            if not np.array_equal(previous, payloads[current_name]):
                raise ValueError("reused current-program recipient payload bytes differ: " + name)
        result = {}
        new_source = None
        for row_id in PRIOR:
            phase("prior_recipe_replay", "start", row=row_id)
            original_id = binding["row_ids"][row_id]
            prior = decode_tree(prior_metadata["value"]["items"][original_id], Path("prior/transaction"))
            runtime = load_capture(Path("prior/captures") / original_id)
            if runtime.trace["model_identity"] != plan["model_identity"]:
                raise ValueError("prior capture model identity differs")
            if not all(np.array_equal(value, predicted[row_id]["roots"][name])
                       for name, value in prior["roots"].items()):
                raise ValueError("prior architectural/input root identity differs: " + row_id)
            if row_id == "new.native":
                new_source = runtime.fork()
            advance(runtime, len(runtime.trace["events"]), row_id)
            replay_heads = runtime.outputs()
            exact = {name: bool(np.array_equal(replay_heads[name].numpy(), prior["heads"][name])) for name in HEADS}
            if not all(exact.values()):
                raise ValueError("prior recipe/head evidence join differs: " + row_id)
            stages, references = actual_stages(NativePortCatalog(runtime.trace), runtime.read)
            for name, value in prior["stages"].items():
                if not np.array_equal(stages[name], value):
                    raise ValueError("prior recipe/stage evidence join differs: " + row_id + ":" + name)
            prior.update({"stages": stages, "actual_ports": references, "evidence_role": "prior",
                          "original_row_id": original_id, "prior_binding": binding["pins"],
                          "prior_recipe_exact": exact, "forecast_root_join": True,
                          "stage_value_authority": "34 original captured arrays checked exactly against root-only replay; S/h replay-derived at original actual native input ports"})
            result[row_id] = prior
            counters["prior_rows_reused"] += 1
            phase("prior_recipe_replay", "end", row=row_id)
        old_source = None
        for row_id in FRESH:
            key, _ = row_id.split(".", 1)
            driver = ISADriver(plan["contexts"][key]["program_hex"], plan["register_words"], plan["flags"],
                               max_instructions=1)
            inputs = driver.inputs()
            raw = tuple(value.detach().cpu().numpy().copy() for value in inputs)
            hooks = []
            phase("native_record", "start", row=row_id)
            try:
                for selector, _, value in changes(row_id, payloads):
                    module_path = selector[len("module:"):-len(":return")]
                    frozen = torch.from_numpy(np.array(value, copy=True))

                    def replace(module, args, old, replacement=frozen):
                        return replacement.clone()

                    hooks.append(model.get_submodule(module_path).register_forward_hook(replace))
                capture = record_forward(model, inputs, model_identity=plan["model_identity"],
                    input_identity=identity({"program": plan["contexts"][key]["program_hex"], "offset": 0,
                                             "words": plan["register_words"], "flags": plan["flags"]}),
                    execution=plan["execution"], capture_ports=True,
                    metadata={"case_id": row_id, "program_step": 0}, max_operations=8192)
            finally:
                for hook in hooks:
                    hook.remove()
            manifest = save_capture(capture, output / "captures" / row_id)
            phase("native_record", "end", row=row_id)
            if row_id == "old.native":
                old_source = ReplayForward(capture)
            source = old_source if key == "old" else new_source
            runtime = source.fork()
            source_ports = NativePortCatalog(runtime.trace)
            acts = []
            phase("typed_replay", "start", row=row_id)
            for selector, name, value in changes(row_id, payloads):
                port = source_ports.resolve(selector)
                advance(runtime, port["available_pc"], row_id)
                act = runtime.apply(port["value_id"], torch.from_numpy(np.array(value, copy=True)))
                acts.append({"payload_name": name, "port": port, "receipt": act})
                counters["acts"] += 1
            advance(runtime, len(runtime.trace["events"]), row_id)
            replay_heads = runtime.outputs()
            exact = {name: bool(torch.equal(replay_heads[name], capture.output[name])) for name in HEADS}
            phase("typed_replay", "end", row=row_id, exact=exact)
            ports = NativePortCatalog(capture.trace)
            stages, references = actual_stages(ports, lambda value: capture.observed_values[value])
            heads = {name: capture.output[name].detach().cpu().numpy().copy() for name in HEADS}
            native_decode = driver.decode(capture.output)
            decoded = {"register_words": [int(word, 16) for word in native_decode["register_words"]],
                       "flags": native_decode["flags_CF_ZF_SF_OF"], "length": native_decode["length"],
                       "store": native_decode["store"], "syscall": native_decode["syscall"]}
            roots = dict(zip(("window", "regs", "flags"), raw, strict=True))
            result[row_id] = {"heads": heads, "stages": stages, "decoded": decoded,
                "roots": roots, "before": {"register_words": plan["register_words"], "flags": plan["flags"], "offset": 0},
                "evidence_role": "fresh", "actual_ports": references, "acts": acts,
                "decode_authority": "independent public MDB ISADriver.decode; no architectural commit",
                "native_input_authority": "independent public MDB ISADriver.inputs; literal program/register/flag packing",
                "stage_value_authority": "actual native capture observed_values at recorded module ports",
                "replay_native_exact": exact, "mechanics_status": "valid" if all(exact.values()) else "instrument-mismatch",
                "forecast_root_join": all(np.array_equal(value, predicted[row_id]["roots"][name]) for name, value in roots.items()),
                "trace_fingerprint": capture.trace["fingerprint"], "input_identity": capture.trace["input_identity"],
                "capture_manifest": manifest,
                "input_roots": {name: tensor_descriptor(value) for name, value in zip(("window", "regs", "flags"), inputs, strict=True)}}
        if counters["physical_forwards"] != 7:
            raise ValueError("fresh physical call count differs from seven")
        phase("native_execution", "end", counts=dict(counters))
        phase("observation_publication", "start", scope="transaction validation/encoding/hash/publication combined")
        return result

    phase("transaction_initialization", "start", scope="declarations/full-weight publication and guard installation")
    transaction = ForecastTransaction(output / "transaction", model=model,
        model_identity=plan["model_identity"], input_identity=sha(raw_plan), source_identity=request["source_hashes"],
        frontier=plan["frontier"], budgets={"max_native_calls": 12, "max_wall_ms": 240000,
                                         "max_bytes": 128 * 1024 * 1024, "max_files": 4096})
    phase("transaction_initialization", "end")
    try:
        with transaction:
            transaction.run_script([{"op": "frontier", "name": "raw", "value": plan},
                                    {"op": "frontier", "name": "prior_binding", "value": binding},
                                    {"op": "frontier", "name": "structural_prefix_hypothesis", "value": hypothesis}])
            transaction.run_script([{"op": "predict", "name": "campaign", "callback": predict}])
            phase("forecast_publication", "end")
            phase("global_seal", "start", physical_forwards=counters["physical_forwards"])
            transaction.run_script([{"op": "seal"}])
            phase("global_seal", "end", seal_sha256=transaction.report()["seal_sha256"],
                  physical_forwards=counters["physical_forwards"])
            transaction.run_script([{"op": "execute", "name": "native", "callback": execute}])
            phase("observation_publication", "end")
    finally:
        count_hook.remove()
        save(output / "transaction-report.json", transaction.report())
        save(output / "phase-timings.json", timings)
        save(output / "native-events.json", native_events)
        save(output / "public-replay-receipts.json", transitions)
    verified = verify_transaction(output / "transaction", expected_seal_sha256=transaction.report()["seal_sha256"],
                                  expected_event_head=transaction.report()["event_head"])
    observed = transaction.observation("native")
    mismatches = [row for row in FRESH if observed[row]["mechanics_status"] != "valid"]
    summary = {"schema": "mdb-context-window-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counters, "fresh_arms": list(FRESH), "prior_arms": list(PRIOR),
        "mechanics_status": "valid" if not mismatches else "instrument-mismatch", "instrument_mismatch_rows": mismatches,
        "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started,
        "transaction_verification": verified, "prior_binding": binding["pins"],
        "horizon": "first instruction only; no successor calls or architectural commits",
        "cross_panel": "two matched transfers with endogenous natural P2 and frozen analytical off-diagonal P2",
        "limits": plan["limits"]}
    save(output / "summary.json", summary)
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path.read_bytes())
                               for path in sorted(output.rglob("*")) if path.is_file()})
    support()["emit"](output, summary)


def stage(run_dir, isa_root, plan_path, prior_root, hypothesis_path=None):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    validate_plan(plan)
    hypothesis_path = hypothesis_path or Path(__file__).with_name("context-prefix-hypothesis-20261030.json")
    raw_hypothesis = hypothesis_path.read_bytes()
    json.loads(raw_hypothesis)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_bytes(raw_plan)
    prior_files, binding = stage_prior(prior_root, plan, run_dir / "prior-binding.json")
    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    sources, origins, _ = upstream["source_inventory"]()
    wheel = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))["build_isa_wheel"](run_dir, sources)
    files = [(path, name) for name, path in sources.items()]
    files += prior_files + [(wheel, wheel.name), (Path(__file__).with_name("discovery_upgrades.py"), "campaign_support.py"),
        (Path(__file__).with_name("raw_isa_equations.py"), "raw_isa_equations.py"),
        (Path(__file__).with_name("explicit_suffix_equations.py"), "explicit_suffix_equations.py"),
        (run_dir / "plan.json", "plan.json"), (hypothesis_path, "context-prefix-hypothesis.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["context-window-result"],
        "hashed_roots": ["artifacts", "direct_effect", "isa_lab", "prior", wheel.name, "context_window_campaign.py",
                         "raw_isa_equations.py", "explicit_suffix_equations.py", "campaign_support.py",
                         "plan.json", "context-prefix-hypothesis.json", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    config = {"schema": "mdb-context-window-config-v1", "checkpoint": "artifacts/model.pt",
              "plan_file_sha256": sha(raw_plan), "hypothesis_file_sha256": sha(raw_hypothesis)}
    request = {"schema": "mdb-context-window-request-v1", "task_family": "mdb-context-window",
        "model_identity": plan["model_identity"], "needs": {"cpu": True}, "config": config,
        "execution_environment": intent(), "source_hashes": {name: sha(path.read_bytes()) for path, name in files},
        "source_origins": origins, "prior_binding": binding,
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=12, logical_branches=10,
            logical_checkpoint_replays=10, max_local_work_items=32,
            extra={"max_replay_native_operations": 16000, "max_physical_model_forwards": 12}),
        "selection_policy": "seven fresh first-step calls; three explicitly reused prior controls; global raw-equation forecast seal"}
    request["source_hashes"]["context_window_campaign.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = identity(request)
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("context_window_campaign.py") + ["--worker"]
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-context-window",
        task="r3-trailing-window-first-step-cross-p2-transfer", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="One CPU1 finite assay:exactly7fresh first-instruction records plus3prior recipes; <=12physical/16k replayops. Raw equations globally sealed beforefresh native. Old/repeatedSUB vsnew/mixedcontext; two matched frozenanalytic crossP2 transfers,no successor/no retries.2304MiB measuredparent exception (prior962MiB telemetry); compact36ports only. Stopafter10rows/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--hypothesis", type=Path, help="defaults to the dated structural-prefix hypothesis beside this script")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("context-window-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-context-window-result-v1", "mechanics_status": "errored",
                          "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
                          "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                support()["emit"](output, report)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect)))
    else:
        if any(value is None for value in (args.run_dir, args.isa_root, args.plan, args.prior_root)):
            parser.error("staging requires run-dir, isa-root, plan and prior-root")
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve(), args.plan.resolve(), args.prior_root.resolve(),
                               None if args.hypothesis is None else args.hypothesis.resolve())))


if __name__ == "__main__":
    main()
