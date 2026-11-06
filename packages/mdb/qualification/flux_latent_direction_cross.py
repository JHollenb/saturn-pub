"""Predict a receiver-by-latent-direction cross before its native targets.

Six independent mathematical suffixes reuse the frozen original equations.
Only original N1/P1 input arrays, conditioning, IDs, schedules and named
checkpoint parameters are numerical inputs. There are no native forwards.
"""
from __future__ import annotations

import base64
import faulthandler
import gc
import gzip
import json
import os
import runpy
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = "latent-direction-math-result"
CELLS = ("c1-o0", "c0-o1")
BRANCHES = tuple(c + "." + role for c in CELLS for role in ("sham", "d-c1-o0", "d-c0-o1"))
REPLAY_SHA256 = "583bb90bf489df8ba14ec118cdae8316855523d033548b2cd62d85be0999705d"


def require(value, message):
    if not value:
        raise ValueError(message)


def helpers():
    replay_path = HERE / "flux_matched_pass_replay.py"
    import hashlib
    require(hashlib.sha256(replay_path.read_bytes()).hexdigest() == REPLAY_SHA256, "original replay helper differs")
    return runpy.run_path(str(replay_path))


def rms(value):
    wide = np.asarray(value, dtype=np.float64)
    require(wide.size and np.isfinite(wide).all(), "finite nonempty operand required")
    return float(np.sqrt(np.mean(wide * wide)))


def bf16(value):
    """One round-to-nearest-even cast after the declared FP32 conversion."""
    value = np.array(value, dtype=np.float32, copy=True, order="C")
    require(np.isfinite(value).all(), "finite pre-BF16 input required")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(32767) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)
    result = rounded.view(np.float32)
    require(np.isfinite(result).all(), "BF16 input overflow")
    return result


def compose(native, direction, requested_rms):
    """No post-round dose tuning. Actual installed changes remain observations."""
    native, direction = np.asarray(native), np.asarray(direction)
    require(native.dtype == np.float32 and direction.shape == native.shape
        and np.isfinite(native).all() and np.isfinite(direction).all(), "input/direction ABI differs")
    require(np.isfinite(requested_rms) and requested_rms > 0, "positive finite nominal RMS required")
    norm = rms(direction)
    require(norm > 0, "zero direction")
    scale = requested_rms / norm
    before = native.astype(np.float64) + scale * direction.astype(np.float64)
    installed = bf16(before.astype(np.float32))
    effective = installed.astype(np.float64) - native.astype(np.float64)
    dot = float(np.sum(effective * direction.astype(np.float64)))
    length = float(np.linalg.norm(effective.ravel()) * np.linalg.norm(direction.astype(np.float64).ravel()))
    return installed, {"requested_rms": requested_rms, "source_direction_rms": norm,
        "scale": scale, "effective_rms": rms(effective),
        "effective_max_abs": float(np.max(np.abs(effective))),
        "changed_values": int(np.count_nonzero(effective)),
        "effective_source_cosine": dot / length if length else None,
        "rounding_rms": rms(installed.astype(np.float64) - before),
        "formula": "BF16_RNE(FP32(N_receiver + (min_source_RMS/RMS(delta_source))*delta_source)); FP64 addition/scaling; no retuning"}


def retained(logical, local):
    return local in (0, 5, 25, 26, 27) if logical == 1 else local == 27


def replay_suffix(reader, binding, origin, installed, *, two, describe, context_fields, kind_at, archive, output, progress=None):
    """Own-state 84-transition recurrence. No native target reader is supplied."""
    values = {name: reader.array(record) for name, record in origin["inputs"].items() if name != "latent"}
    latent = np.array(installed, dtype=np.float32, copy=True)
    conditioning = values["conditioning"]
    transitions = []
    for logical in (1, 2, 3):
        records = binding["pass_schedules"][str(logical)]
        schedule = {name: reader.array(records[name]) for name in ("timestep", "sigma", "sigma_next")}
        require(float(schedule["timestep"]) == float(values["timesteps"][logical])
            and float(schedule["sigma"]) == float(values["sigmas"][logical])
            and float(schedule["sigma_next"]) == float(values["sigmas"][logical + 1])
            and records["next_clock"] == {"step": logical + 1, "phase": "project", "cursor": 0}
            and records["reset"] == {k: None for k in ("context", "image", "text", "noise_prediction")},
            "source schedule/clock/reset differs")
        iterator = two["independent_pass_events"](latent, conditioning, values["img_ids"], values["txt_ids"], schedule,
            binding, lambda kind, spec: two["load_bank"](reader, kind, spec))
        context_hash = None
        previous = None
        local_count = 0
        for local, (kind, index, value) in enumerate(iterator):
            require(local < 28 and (kind, index) == kind_at(local), "suffix operation sequence differs")
            if kind == "project":
                inputs = {"latent": describe(latent), "conditioning": origin["inputs"]["conditioning"],
                    "img_ids": origin["inputs"]["img_ids"], "txt_ids": origin["inputs"]["txt_ids"], "schedule": records}
                image, text = value["image"], value["text"]
                fields = {"image": image, "text": text, **context_fields(value["context"])}
                context_hash = helpers_identity({name: describe(field) for name, field in context_fields(value["context"]).items()})
            elif kind in ("joint", "single"):
                inputs = {"image": describe(image), "text": describe(text), "own_project_context": context_hash}
                if kind == "single":
                    inputs["residual"] = describe(np.concatenate((text, image), axis=1))
                    text, image = value["output"][:, :conditioning.shape[1]], value["output"][:, conditioning.shape[1]:]
                else:
                    image, text = value["image"], value["text"]
                fields = {"image": image, "text": text}
            elif kind == "readout":
                inputs = {"image": describe(image), "own_project_context": context_hash}
                fields = {"noise": value["noise"]}
            else:
                inputs = {"latent": describe(latent), "noise": describe(previous["noise"]),
                    "sigma": records["sigma"], "sigma_next": records["sigma_next"]}
                fields = {"latent": value["latent"]}
            transitions.append({"logical_step": logical, "local_index": local, "kind": kind, "index": index,
                "inputs": inputs, "outputs": {name: {"descriptor": describe(field),
                    "archive": archive(output, field, describe(field)) if retained(logical, local) else None}
                    for name, field in fields.items()}})
            if progress:
                progress({"logical_step": logical, "local_index": local, "kind": kind, "native_forwards": 0})
            previous, local_count = fields, local_count + 1
        require(local_count == 28 and "latent" in previous, "suffix pass did not commit28 own-state events")
        latent = np.array(previous["latent"], copy=True)
    require(len(transitions) == 84, "suffix census differs")
    return transitions


def helpers_identity(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run(data, *, output, parameter_root, checkpoint, expectations, request, progress=None, production_profile=True):
    h = helpers()
    require(h["sha"](HERE / "analyze_flux_prompt_factorial.py") == h["ANALYZER_SHA256"]
        and h["sha"](HERE / "memoize_flux_audit_parameters.py") == h["MEMOIZATION_SHA256"], "frozen numerical sources differ")
    analyzer = runpy.run_path(str(HERE / "analyze_flux_prompt_factorial.py"))
    old = analyzer["independent"]()
    memo = runpy.run_path(str(HERE / "memoize_flux_audit_parameters.py"))["install"](
        analyzer, 18 << 30, storage_codec="float32")
    output = Path(output)
    require(not output.exists(), "fresh mathematical output required")
    output.mkdir()
    (output / "arrays").mkdir()
    (output / "programs").mkdir()
    save, read, sha = (h[k] for k in ("save", "read", "sha"))
    try:
        source, summary, plan, binding = [read(data / n) for n in ("source.json", "summary.json", "plan.json", "binding.json")]
        analyzer["validate_plan"](plan, source, summary, production_profile=production_profile)
        require(source["binding_sha256"] == sha(data / "binding.json"), "original source/binding differs")
        if production_profile:
            analyzer["source_abi"](data, source, plan)
            old["bank_profiles"](binding)
            require(source["checkpoint_files"][analyzer["CHECKPOINT_FILE"]] == analyzer["CHECKPOINT_SHA256"], "checkpoint differs")
        analyzer["parameter_ports"](binding)
        require(binding["parameter_artifacts"]["sha256"] == sha(data / "parameter-artifacts.json")
            and sha(data / "retrieve_weights.py") == analyzer["PARAMETER_SHA256"], "parameter retrieval binding differs")
        recipes, hydration = old["parameter_cache"](data, parameter_root, checkpoint=checkpoint)
        evidence = old["Evidence"](data, parameter_root, recipes)
        from mdb.forecast_transaction import load_forecast_tree, verify_transaction
        origins, native_inputs, directions, receivers, direction_records = {}, {}, {}, {}, {}
        for cell in CELLS:
            origin, native_binding = h["selected_native_origin"](analyzer, old, evidence, binding, data, cell)
            origins[cell] = origin
            native_inputs[cell] = evidence.array(origin["inputs"]["latent"])
            tx = data / "transactions" / cell
            verification = verify_transaction(tx, expected_seal_sha256=summary["seals"][cell]["seal_sha256"], max_bytes=plan["transaction_max_bytes"])
            program = analyzer["plain"](load_forecast_tree(tx, read(tx / "forecasts/program.json")["predictions"]))
            require(program["origin_cut"] == binding["origins"][cell]["cut"] and program["condition_id"] == cell,
                "original P1 source transaction differs")
            p1 = program["transitions"][27]["outputs"]["latent"]
            predicted = evidence.array(p1["archive"])
            require(analyzer["describe_field"](predicted) == p1["descriptor"], "original P1 bytes differ")
            directions[cell] = predicted.astype(np.float64) - native_inputs[cell].astype(np.float64)
            reconstructed = bf16((native_inputs[cell].astype(np.float64) + directions[cell]).astype(np.float32))
            require(np.array_equal(reconstructed.view(np.uint32), predicted.view(np.uint32)), "unit-direction reconstruction changed original P1")
            receivers[cell] = {"cut": native_binding["cut"], "source_transition": 27, "origin": origin}
            # Native loader expects the complete original capture, not just its fingerprint.
            native_rows = (data / "native" / (cell + ".jsonl")).read_text().splitlines()
            receivers[cell]["cut"] = json.loads(native_rows[27])["capture"]
            direction_records[cell] = {"P1": p1, "N1": origin["inputs"]["latent"],
                "rms": rms(directions[cell]), "original_transaction": verification,
                "unit_direction_reconstructs_original_P1_exactly": True,
                "source_scope": "Previously observed input discrepancy only; no later outcome arrays read."}
        nominal = min(rms(directions[c]) for c in CELLS)
        branches, programs = {}, {}
        for cell in CELLS:
            for direction in (None, *CELLS):
                name = cell + (".sham" if direction is None else ".d-" + direction)
                installed, effective = (native_inputs[cell].copy(), {"requested_rms": 0.0, "effective_rms": 0.0,
                    "effective_max_abs": 0.0, "changed_values": 0, "scale": 0.0}) if direction is None else compose(native_inputs[cell], directions[direction], nominal)
                descriptor = analyzer["describe_field"](installed)
                require("bf16_le_sha256" in descriptor, "installed latent not exact BF16")
                input_archive = h["archive"](output, installed, descriptor)
                branch = {"receiver": cell, "direction": direction, "kind": "sham" if direction is None else "direction",
                    "input_latent": input_archive, "input_descriptor": descriptor, "requested_rms": nominal if direction else 0.0,
                    "effective_delta": effective,
                    "known_input_control": cell == direction == "c0-o1"}
                branches[name] = branch
        inputs = {"schema": "mdb-latent-direction-inputs-v1", "receivers": receivers, "directions": direction_records,
            "branches": branches, "requested_common_rms": nominal,
            "source": {"parent_job_id": source["job_id"], "parent_request_sha256": source["request_sha256"]},
            "scope": "Known selected N1/P1 sources. Low own direction at unit scale is an already observed input control; other three edited inputs newly constructed. No semantic correspondence asserted across prompt coordinates."}
        save(output / "inputs.json", inputs)
        save(output / "expectations.json", expectations)
        for name in BRANCHES:
            branch = branches[name]
            origin = origins[branch["receiver"]]
            allowed = [v for k, v in origin["inputs"].items() if k != "latent"]
            allowed += [binding["pass_schedules"][str(s)][k] for s in (1, 2, 3) for k in ("timestep", "sigma", "sigma_next")]
            reader = h["InputOnlyReader"](evidence, allowed)
            installed = h["load_q"](output, branch["input_latent"], analyzer["describe_field"])
            def tick(row, name=name):
                if progress:
                    progress({"branch": name, **row})
            transitions = replay_suffix(reader, binding, origin, installed, two=old["TWO"], describe=analyzer["describe_field"],
                context_fields=analyzer["context_fields"], kind_at=analyzer["kind_at"], archive=h["archive"], output=output, progress=tick)
            program = {"schema": "mdb-latent-direction-program-v1", "branch_id": name, "receiver": branch["receiver"],
                "direction": branch["direction"], "origin_cut": origin["cut"], "inputs": branch,
                "chain_length": 84, "transitions": transitions, "terminal_latent": transitions[-1]["outputs"]["latent"],
                "input_read_proof": reader.reads}
            relative = "programs/" + name + ".json"
            save(output / relative, program)
            programs[name] = {"path": relative, "sha256": sha(output / relative)}
            del transitions, program
            gc.collect()
        seal = {"schema": "mdb-latent-direction-cohort-seal-v1", "programs": programs,
            "inputs_sha256": sha(output / "inputs.json"), "expectations_sha256": sha(output / "expectations.json"),
            "source": inputs["source"], "math_transitions": 504, "native_forwards": 0}
        save(output / "cohort-seal.json", {**seal, "fingerprint": helpers_identity(seal)})
        cache_report = memo.uninstall()
        import shutil
        for name in ("parameter-artifacts.json", "retrieve_weights.py"):
            shutil.copyfile(data / name, output / name)
        result = {"schema": "mdb-latent-direction-math-summary-v1", "source_job_id": request["job_id"],
            "parent_job_id": source["job_id"], "math_transitions": 504, "native_forwards": 0,
            "programs": programs, "cohort_seal_sha256": sha(output / "cohort-seal.json"),
            "inputs_sha256": sha(output / "inputs.json"), "expectations_sha256": sha(output / "expectations.json"),
            "requested_common_rms": nominal, "parameter_hydration": hydration, "parameter_read_memoization": cache_report,
            "scope": inputs["scope"]}
        save(output / "summary.json", result)
        return result
    finally:
        if memo.stats()["active"]:
            memo.uninstall()


def main():
    started = time.monotonic()
    faulthandler.dump_traceback_later(90, repeat=True)
    h = helpers()
    read, sha, save = (h[k] for k in ("read", "sha", "save"))
    config, request = (read(os.environ[k]) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and config["mathematical_transitions"] == 504,
        "declared CPU-only mathematical work differs")
    require(all(sha(HERE / n) == d for n, d in request["source_hashes"].items()), "staged CPU payload differs")
    parent = HERE / "parent-run"
    collector = runpy.run_path(str(HERE / "collect_flux_prompt_factorial.py"))
    audit = runpy.run_path(str(HERE / "audit_flux_prompt_factorial.py"))
    original_request = read(parent / "request.json")
    remote = Path(collector["validate_job"](read(parent / "collection-job.json"), config["parent_job_id"], original_request))
    expected = read(parent / "collection-local-inventory.json")
    require(str(remote) == config["parent_remote_result"] and collector["inventory"](remote) == expected, "original parent inventory differs")
    data = parent / "extracted/flux-prompt-factorial-result"
    audit["hardlink_inventory"](remote, data, expected)
    require(collector["inventory"](data) == expected, "hardlink parent differs")
    require(sha(HERE / "parent-audit/audit.json") == config["parent_audit_sha256"], "accepted numerical audit differs")
    def progress(value):
        require(time.monotonic() - started <= config["max_wall_seconds"], "finite CPU prediction wall budget exhausted")
        print("MDB_LATENT_DIRECTION_PHASE " + json.dumps(value, sort_keys=True), flush=True)
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output, cache = HERE / RESULT, HERE / "audit-parameter-cache"
    require(not cache.exists(), "fresh job-owned temporary parameter cache required")
    summary = run(data, output=output, parameter_root=cache, checkpoint=Path(config["checkpoint_root"]),
        expectations=read(HERE / "expectations.json"), request=request, progress=progress)
    require(collector["inventory"](remote) == expected == collector["inventory"](data), "original parent changed during prediction")
    released = audit["release_cache"](cache, read(data / "parameter-artifacts.json"))
    execution = {"job_id": request["job_id"], "request_sha256": request["request_sha256"], "worker_sha256": sha(__file__),
        "parent_job_id": config["parent_job_id"], "parent_request_sha256": original_request["request_sha256"],
        "mathematical_transitions": 504, "native_framework_forwards": 0, "parent_inventory_unchanged": True,
        "parameter_cache_cleanup": released, "seconds": time.monotonic() - started}
    save(output / "source.json", {**execution, "source_hashes": request["source_hashes"], "parent_inventory_sha256": sha(parent / "collection-local-inventory.json")})
    save(output / "plan.json", config)
    save(output / "execution.json", execution)
    inventory = collector["inventory"](output)
    save(output / "remote-manifest.json", {"schema": "mdb-latent-direction-math-manifest-v1", "job_id": request["job_id"],
        "request_sha256": request["request_sha256"], "files": inventory["files"]})
    capsule = {"schema": "mdb-latent-direction-math-execution-v1", "report": summary,
        "near_data_execution": execution, "remote_manifest_sha256": sha(output / "remote-manifest.json"), "output_directory": RESULT}
    raw = (json.dumps(capsule, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(raw) <= 32 << 20, "capsule report exceeds transport bound")
    compressed = gzip.compress(raw, mtime=0)
    require(len(compressed) <= 8 << 20, "capsule exceeds compressed transport bound")
    import hashlib
    print("MDB_FACTORIAL_AUDIT_CAPSULE=" + json.dumps({"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "gzip_bytes": len(compressed),
        "gzip_sha256": hashlib.sha256(compressed).hexdigest(), "base64_gzip": base64.b64encode(compressed).decode()}), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
