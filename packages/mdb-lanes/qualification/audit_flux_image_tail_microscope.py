"""Near-data full-array response audit and compact MDB equation records; no forwards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_inventory(root, expected):
    root = Path(root)
    if sha(root / "files.json") != expected:
        raise ValueError("native manifest changed")
    manifest = read(root / "files.json")
    for name, record in manifest["files"].items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]):
            raise ValueError("native evidence changed: " + name)
    return manifest


def square_account(a, b, c, d):
    """D−A = head substitution + tail substitution + their signed interaction."""
    import numpy as np
    a, b, c, d = (np.asarray(value, np.float64) for value in (a, b, c, d))
    if len({value.shape for value in (a, b, c, d)}) != 1:
        raise ValueError("square geometry differs")
    total, head, tail, interaction = d - a, b - a, c - a, d - b - c + a
    terms = {"head": head, "tail": tail, "interaction": interaction}
    energies = {name: float(np.vdot(value, value)) for name, value in terms.items()}
    cross = {left + "__" + right: float(2 * np.vdot(terms[left], terms[right]))
             for left, right in (("head", "tail"), ("head", "interaction"), ("tail", "interaction"))}
    return {"total_squared_l2": float(np.vdot(total, total)), "term_squared_l2": energies,
        "signed_cross_terms": cross, "closure_rms": float(np.sqrt(np.mean((total - head - tail - interaction) ** 2))),
        "energy_sum": sum(energies.values()) + sum(cross.values()),
        "scope": "Fixed-coordinate substitution identity; these are not independent causal shares."}


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np
    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations
    h = runpy.run_path(str(HERE / "flux_source_receiver.py"))
    measure = runpy.run_path(str(HERE / "analyze_flux_reader_arithmetic.py"))["measure"]
    load = h["load_array"]
    root, output = Path(root), Path(output)
    manifest = validate_inventory(root, expected_manifest)
    summary, config, source = (read(root / name) for name in ("summary.json", "plan.json", "source.json"))
    if (summary["request_sha256"] != expected_request or manifest["request_sha256"] != expected_request
            or config["request_sha256"] != expected_request or source["request_sha256"] != expected_request
            or manifest["job_id"] != summary["job_id"] or source["job_id"] != summary["job_id"]):
        raise ValueError("native result lineage differs")
    rows = read(root / "cases.json")
    if set(rows) != {"sham", "edit"} or summary["counts"]["native_blocks"] != 2 or summary["counts"]["native_attention"] != 2:
        raise ValueError("two-case capture census differs")
    for role, row in rows.items():
        expected_cut = config["inputs"][role]["cut"]
        origin = source["source_cuts"][role]
        if (row["source_cut"] != expected_cut or origin["cut"] != expected_cut
                or origin["manifest"]["model_identity"] != config["expected_model_identity"]):
            raise ValueError("native source-role/model join differs: " + role)
    output.mkdir(parents=True, exist_ok=False)

    def value(role, lane, port):
        row = rows[role]
        if lane == "observed":
            return load(root, row["actual"][port])
        group, name = lane.split(".", 1)
        record = row[group][name]
        if port == "head" and port not in record["fields"] and "head" in record:
            return load(root, record["head"])
        return load(root, record["fields"][port])

    lanes = {group + "." + name: set(record["fields"]) | ({"head"} if "head" in record else set())
             for group in ("native", "independent") for name, record in rows["sham"][group].items()}
    errors, effects, local_policy, replay = {}, {}, {}, {}

    def change(sham, edit):
        delta = np.asarray(edit, np.float64) - np.asarray(sham, np.float64)
        return {"orientation": "native edit minus native sham", "rms": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.max(np.abs(delta))), "signed_mean": float(delta.mean()),
            "changed_values": int(np.count_nonzero(delta)), "elements": int(delta.size)}

    entry_changes = {port: change(load(root, rows["sham"]["inputs"][port]), load(root, rows["edit"]["inputs"][port]))
                     for port in ("image", "text", "img_mod", "txt_mod")}
    text_rows = rows["sham"]["inputs"]["text"]["shape"][1]
    positioned_changes = {}
    for port in ("q", "k", "v"):
        sham, edit = value("sham", "observed", port), value("edit", "observed", port)
        positioned_changes[port] = {"text_rows": change(sham[:, :text_rows], edit[:, :text_rows]),
            "image_rows": change(sham[:, text_rows:], edit[:, text_rows:])}
    for role, row in rows.items():
        errors[role] = {}
        for lane, ports in lanes.items():
            errors[role][lane] = {port: measure(value(role, "observed", port), value(role, lane, port))
                                  for port in sorted(ports) if port in row["actual"]}
        replay[role] = row["historical_replay"]
    for lane, ports in lanes.items():
        effects[lane] = {}
        for port in sorted(ports):
            if port not in rows["sham"]["actual"]:
                continue
            native = value("edit", "observed", port).astype(np.float64) - value("sham", "observed", port).astype(np.float64)
            predicted = value("edit", lane, port).astype(np.float64) - value("sham", lane, port).astype(np.float64)
            effects[lane][port] = measure(native, predicted)
            del native, predicted
    for port in ("projection", "paired", "mlp"):
        native = value("edit", "native.local-opposite", port).astype(np.float64) - value("sham", "native.local-opposite", port).astype(np.float64)
        predicted = value("edit", "independent.local", port).astype(np.float64) - value("sham", "independent.local", port).astype(np.float64)
        local_policy[port] = measure(native, predicted)
        del native, predicted
    square_lanes = ("native.actual-head", "native.reference-head", "independent.actual-head", "independent.reference-head")
    squares = {role: square_account(*(value(role, lane, "image") for lane in square_lanes)) for role in rows}
    squares["changed-minus-sham"] = square_account(*(value("edit", lane, "image").astype(np.float64)
        - value("sham", lane, "image").astype(np.float64) for lane in square_lanes))
    result = {"schema": "mdb-image-tail-audit-v1", "job": summary["job_id"], "mechanics_status": "verified-saved-array-custody",
        "instrument_status": "historical-and-primitive-replay-observations-retained-per-field", "trend_status": "observed",
        "terminal_status": "unassessed", "scope": config["scope"], "counts": summary["counts"],
        "native_contract": read(root / "native-contract.json"), "historical_replay": replay,
        "errors": errors, "effects": effects, "independent_local_vs_toggled_native": local_policy,
        "entry_changes": entry_changes, "positioned_QKV_changes": positioned_changes,
        "metric_orientation": {"errors_and_effects_mean_signed": "captured native minus candidate lane; effects are edit minus own sham before comparison",
            "local_vs_toggled_mean_signed": "toggled native effect minus independent local effect",
            "worker_comparisons": "candidate minus native, opposite to this audit's error sign"},
        "boundary_substitution_squares": squares,
        "selected_aten_operations": {role: row["selected_aten_operations"] for role, row in rows.items()},
        "source_job": config["source_job"], "source_cuts": {role: row["source_cut"] for role, row in rows.items()},
        "native_manifest_sha256": expected_manifest, "native_request_sha256": expected_request,
        "auditor_sha256": sha(__file__), "measurement_source_sha256": sha(HERE / "analyze_flux_reader_arithmetic.py"),
        "new_model_calls": 0, "new_reference_forward_calls": 0,
        "limitations": ["Two retrospectively selected states, one receiver/clock and observed operands; no prospective generalization.",
            "Observed-QKV reference head excludes upstream QKV/norm/RoPE arithmetic discrepancies.",
            "Whole-state error and changed-minus-sham response error use different native denominators.",
            "Reduction-policy counterfactual is not a fitted correction to original forecasts.",
            "No suffix or VAE executes here; later HH amplification and semantic names remain unassessed."]}
    save(output / "audit.json", result)
    catalog = EquationCatalog(output / "catalog/equations")
    identifiers, queries = [], []
    record_lanes = ("native.actual-head", "native.reference-head", "independent.actual-head", "independent.reference-head", "native.opposite-reduction")

    def record(case, lane, predicted, native, kind):
        captured_contract = result["native_contract"]
        candidate_contract = ({"arithmetic": "native Torch primitives", "dtype": "bfloat16",
            "bf16_reduced_precision_reduction": (not captured_contract["source_reduction_policy"]
                if lane == "native.opposite-reduction" else captured_contract["source_reduction_policy"])}
            if lane.startswith("native.") else {"arithmetic": "independent FP64 equations with declared BF16 boundaries",
                "native_matrix_reduction_policy": None})
        operand_frontier = {role: {"image": row["inputs"]["image"], "img_mod": row["inputs"]["img_mod"],
            "native_head": row["actual"]["head"], "positioned_QKV": {name: row["actual"][name] for name in ("q", "k", "v")}}
            for role, row in rows.items() if case == "paired" or role == case}
        context = {"model_identity": config["expected_model_identity"], "source_job": summary["job_id"],
            "case": case, "lane": lane, "comparison": kind, "clock": {"step": 1, "phase": "joint", "index": 0},
            "source_cuts": result["source_cuts"], "operand_frontier": operand_frontier,
            "numerical": {"captured_native": captured_contract, "candidate": candidate_contract}, "horizon": 1}
        item = catalog.record(law_id="law.flux.image-tail.observed-operands.v1", route_id="route.flux.joint0.image-attention-residual-mlp.v1",
            alias="cmp.flux.image-tail." + case + "." + lane + "." + kind,
            binding={"model_identity": config["expected_model_identity"], "input_identity": context["source_cuts"],
                "source_identity": expected_request, "execution": context["numerical"], "frontier": context["clock"], "horizon": 1},
            report=compare_equations({"image": predicted}, {"image": native}, max_elements=25_000_000),
            conditions=[{"field": key, "op": "eq", "value": value} for key, value in context.items()], tested_context=context,
            readers=["joint0.image-tail"], evidence={"audit_sha256": sha(output / "audit.json"), "role": "retrospective numerical diagnosis",
                "new_learned_parameters": 0, "quality_gate": False,
                "comparison_orientation": ("lane(edit minus own sham) versus native(edit minus own sham)" if kind == "effect"
                    else "lane image versus same-case observed native image"),
                "residual_orientation": "native minus candidate lane"})
        wrong = {**context, "clock": {"step": 2, "phase": "joint", "index": 0}}
        exact, changed = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to=wrong))
        if exact != 1 or changed != 0:
            raise ValueError("supplied-context equation search mechanics differ")
        identifiers.append(item["fingerprint"])
        queries.append({"case": case, "lane": lane, "comparison": kind, "exact": exact, "changed_clock": changed})
        if case == "edit" and lane == "independent.actual-head":
            save(output / "catalog/context-exact.json", context)
            save(output / "catalog/context-wrong-clock.json", wrong)

    for role in ("sham", "edit"):
        for lane in record_lanes:
            record(role, lane, value(role, lane, "image"), value(role, "observed", "image"), "endpoint")
    for lane in record_lanes:
        predicted = value("edit", lane, "image").astype(np.float64) - value("sham", lane, "image").astype(np.float64)
        native = value("edit", "observed", "image").astype(np.float64) - value("sham", "observed", "image").astype(np.float64)
        record("paired", lane, predicted, native, "effect")
    save(output / "catalog/provenance.json", {"records": len(identifiers), "fingerprints": identifiers,
        "queries": queries, "source_native_manifest_sha256": expected_manifest, "new_model_calls": 0,
        "family_induction": False, "applicability": "supplied exact predicates, not learned rules"})
    return result, len(identifiers), manifest


def worker():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    config = read(os.environ["SATURN_RUN_CONFIG"])
    if config != request["config"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU saved-only audit binding differs")
    for name, digest in request["source_hashes"].items():
        if sha(name) != digest:
            raise ValueError("staged audit source changed")
    started = time.monotonic()
    root = Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, count, manifest = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("cases.json", "native-contract.json", "source.json", "summary.json", "files.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for name in ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", {"host": "beast", "roots": {"native": str(root)},
        "manifests": {"native": {name: record["sha256"] for name, record in manifest["files"].items()}},
        "scope": "Original full tensors retained near data; retrieve individual exact files, no new weight bank."})
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", {"schema": "mdb-image-tail-compact-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "source_job": result["job"], "output_root": str(output),
        "seconds": time.monotonic() - started, "records": count, "model_calls": 0, "native_calls": 0,
        "reference_forward_calls": 0, "mechanics_status": "verified-saved-array-custody", "terminal_status": "unassessed"})
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps({"job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "output_root": str(output), "files_manifest_sha256": sha(output / "files.json"), "summary_sha256": sha(output / "summary.json")}, sort_keys=True), flush=True)


def stage(run, study):
    from mdb_lanes.job_support import one_lease_scheduling_contract, scheduler_api
    from mdb_lanes.payload import stage_worker
    parent = read(study / "request.json")
    binder = runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))
    binder["bind"](parent["execution_environment"])
    job_id = read(study / "submission.json")["job_id"]
    status, raw, _ = scheduler_api().request("GET", "/api/jobs/" + job_id, timeout_s=30)
    job = json.loads(raw)
    if status != 200 or job["state"] != "succeeded" or job["result"]["diagnostics"]["command"] != read(study / "admission-command.json")["command"]:
        raise ValueError("completed exact native command required")
    # Reuse the qualified scheduler/request/declared-sealed-executed payload join.
    collector = HERE / "collect_flux_initial_noise.py"
    if sha(collector) != "e59cb6e3c1f4f0b03f0163f4db8544952def504090a3fdded8e61f530880b9ad":
        raise ValueError("qualified actual-job validator changed")
    runpy.run_path(str(collector))["validate_job"](job, job_id, parent)
    status, log, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected_log = job["result"]["artifacts"]["job_log"]
    if (status != 200 or len(log) > 4 << 20 or len(log) != expected_log["bytes"]
            or hashlib.sha256(log).hexdigest() != expected_log["sha256"]):
        raise ValueError("actual native job log differs")
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines()
               if line.startswith("MDB_DIRECTORY_READY ")]
    if len(markers) != 1:
        raise ValueError("exactly one native directory publication required")
    save(study / "job.json", job)
    with (study / "worker.log").open("xb") as stream:
        stream.write(log)
    root = Path(parent["config"]["output_scratch"]) / job_id / "binding-minimal-result"
    script = "import json,hashlib\nfrom pathlib import Path\np=Path(" + repr(str(root)) + ")\nprint(json.dumps(dict(manifest_sha256=hashlib.sha256((p/'files.json').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((p/'summary.json').read_bytes()).hexdigest(),summary=json.loads((p/'summary.json').read_text()))))\n"
    info = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script, text=True, capture_output=True, check=True, timeout=60).stdout)
    if info["summary"]["job_id"] != job_id or info["summary"]["request_sha256"] != parent["request_sha256"]:
        raise ValueError("native directory identity differs")
    marker = markers[0]
    if (marker["job_id"] != job_id or marker["request_sha256"] != parent["request_sha256"]
            or marker["output_root"] != str(root) or marker["files_manifest_sha256"] != info["manifest_sha256"]
            or marker["summary_sha256"] != info["summary_sha256"]):
        raise ValueError("native scheduler publication and current directory disagree")
    run.mkdir(parents=True, exist_ok=False)
    config = {"native_root": str(root), "native_manifest_sha256": info["manifest_sha256"],
        "native_request_sha256": parent["request_sha256"], "source_job": job_id,
        "output_scratch": "/mnt/big/mdb/research", "directory_custody": True,
        "scope": "Full-array saved-only audit, supplied equation records and compact custody; zero model or mathematical recurrence calls."}
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in ("flux_source_receiver.py", "analyze_flux_reader_arithmetic.py", "retrieve_reader_evidence.py", "flux_family_custody.py")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [Path(__file__).name, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": [str(root), config["output_scratch"], "binding-minimal-result"]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-image-tail-audit-request-v1", "config": config, "execution_environment": parent["execution_environment"],
        "needs": {"cpu": True, "host": "beast"}, "task_family": "analysis", "backend": "numpy", "device": "cpu", "dtype": "float64",
        "source_hashes": {name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        "runtime_wheels": parent["runtime_wheels"],
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=15, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=32, extra={"task_family": "analysis"},
            checkpoint_policy="One saved-only CPU audit with lazy full-field reads, fifteen equation records and compact collection; no retries or new forwards."),
        "resource_exception": {"ram_mb": 3072, "vram_mb": 0, "cpu_threads": 2, "disk_gb": 1, "est_wall_s": 120,
            "calibration": "Prior full-array reader audit job-c8a4bc3cc1e4 used1328.5MB for9.44M paired fields. This audit lazily reads two cases with18.87M paired fields, no Torch/model imports;3072MB allows widened effect operands and comparison buffers."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=Path(__file__), config=config, request=request, additional_files=extras)
    for name, digest in request["runtime_wheels"].items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("audit runtime wheel cohort differs")
    return {"request_sha256": request["request_sha256"], "source_job": job_id}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    request = read(run / "request.json")
    runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))["bind"](request["execution_environment"])
    if (run / "submission.json").exists():
        raise ValueError("audit already submitted")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("audit payload changed")
    command = worker_command(Path(__file__).name)
    command.insert(1, "CUDA_VISIBLE_DEVICES=")
    command.insert(command.index("python") + 1, "-B")
    save(run / "admission-command.json", {"command": command, "request_sha256": request["request_sha256"]})
    job = guarded_mrun_launch(command, request=request, experiment="mdb-image-tail-audit", task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=False, prefer_host="beast", timeout_s=360, retry_on_kill=False, preflight=True,
        **{key: request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s")},
        note="Saved-only full-array image-tail audit:2capturedstates/local-carried comparisons, signedresponseandsubstitutioninteraction;15supplied-context equationrecords. CPU2/RAM3GB,120sestimate/360souter,no native/referenceforwards;rawtrace staysBeast,compact reports/retriever only.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.study.resolve())))
