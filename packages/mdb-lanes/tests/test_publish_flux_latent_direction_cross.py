"""Synthetic saved-custody and plot checks; no model or recurrence execution."""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
HERE = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def module():
    return runpy.run_path(str(HERE / "qualification/publish_flux_latent_direction_cross.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def stats(v):
    v = np.asarray(v, np.float64)
    return {"rms": float(np.sqrt(np.mean(v * v))), "max_abs": float(np.abs(v).max()), "signed_mean": float(v.mean()),
        "p95_abs": float(np.percentile(np.abs(v), 95)), "p99_abs": float(np.percentile(np.abs(v), 99)), "changed_values": int(np.count_nonzero(v))}


def account(scale, *, zero=False, undefined=False):
    return {"native_response": {"rms": 0. if zero else scale}, "mathematical_response": {"rms": scale * .8},
        "prediction_error": {"rms": scale * 1.2, "relative_rms": None if undefined else 1.2},
        "cosine": None if undefined else -.3, "dot": -1., "effective_input_rms": .001,
        "native_finite_response_ratio": 0. if zero else scale * 1000, "mathematical_finite_response_ratio": scale * 800}


@pytest.fixture
def fixture(module, tmp_path):
    p, h, _ = module["helpers"]()
    native, math_run, analysis = [tmp_path / name for name in ("native", "math", "analysis")]
    nd, md = native / "extracted/latent-direction-native-result", math_run / "extracted/latent-direction-math-result"
    nd.mkdir(parents=True)
    md.mkdir(parents=True)
    analysis.mkdir()
    for data in (nd, md):
        write(data / "parameter-artifacts.json", {"artifacts": [{"path": "absent-parameter-" + str(i)} for i in range(169)]})
        (data / "retrieve_weights.py").write_bytes((HERE / "src/mdb_lanes/parameter_artifacts.py").read_bytes())
    inputs = {"requested_common_rms": .001, "branches": {}}
    report = {"schema": "mdb-latent-direction-cross-analysis-v1", "mechanics_status": "authenticated-saved-array-reductions",
        "native_execution_status": "succeeded", "manifest_origin": "producer-remote-manifest",
        "native_job_id": "synthetic-native", "math_job_id": "synthetic-math", "model_calls": 0,
        "counts": {"adapter_steps": 504, "acts": 12, "receipt_emissions": 516, "vae_calls": 12},
        "responses": {}, "factorial": [], "RGB_factorial": {"cosine": -.5, "scope": "Synthetic signed fixture"},
        "RGB": {}, "input_geometry": {}, "effective_input_changes": {}, "endpoints": {},
        "native_replay": {c: {"all_field_hashes_match": False, "terminal_png_exact": False} for c in module["CELLS"]},
        "expectations": [{"holds": False}], "input_construction_audit": {"synthetic": True}, "consumer_join_audit": {"synthetic": True}}
    rows, pixels = {}, {}
    for ci, c in enumerate(module["CELLS"]):
        report["input_geometry"][c] = {"effective_input_cosine_across_receivers": .9 - ci * .1}
        for si, suffix in enumerate((".sham", ".d-c1-o0", ".d-c0-o1")):
            b = c + suffix
            inputs["branches"][b] = {"receiver": c, "direction": None if si == 0 else module["CELLS"][si - 1],
                "effective_delta": {"rounding_rms": (ci + 1) * (si + 1) * 1e-5}}
            refs, values = {}, {}
            for arm in ("native", "predicted"):
                value = np.full((512, 512, 3), 50 + ci * 30 + si * 2 + int(arm == "predicted"), dtype=np.uint8)
                path = nd / (b + "." + arm + ".png")
                Image.fromarray(value).save(path)
                refs[arm] = {"path": path.name, "sha256": h["sha"](path)}
                values[arm] = value
            rows[b] = {"image": refs["native"], "predicted": {"image": refs["predicted"]}}
            pixels[b] = values
            report["endpoints"][b] = {"RGB": stats(values["predicted"].astype(float) - values["native"]), "latent": stats(np.array([.1, -.2]))}
            if si:
                branch_rows = []
                for pos, fields in ((0, ("image", "text", "context.temb")), (5, ("image", "text")), (25, ("image", "text")),
                        (26, ("noise",)), (27, ("latent",)), (55, ("latent",)), (83, ("latent",))):
                    values = {name: account((ci + 1) * (si + 1) * (pos + 1) / 10, zero=pos == 0,
                        undefined=pos == 0 or (ci == 0 and si == 2 and pos == 25)) for name in fields}
                    branch_rows.append({"position": pos, "logical_step": 1 + pos // 28, "local_index": pos % 28, "fields": values})
                report["responses"][b] = branch_rows
                report["RGB"][b] = account(.2)
                report["effective_input_changes"][b] = {"rms": .001 + ci * .0001, "max_abs": .005 + si * .001}
    report["factorial"] = [{"position": pos, "fields": {"signed_fixture": {"cosine": -.7, "signed_mean": -3.}}}
        for pos in (0, 5, 25, 26, 27, 55, 83)]
    report["inputs"] = inputs
    write(md / "inputs.json", inputs)
    programs = {}
    for b in module["BRANCHES"]:
        rel = "programs/" + b + ".json"
        write(md / rel, {"scope": "Synthetic program reference only; not a mathematical forecast"})
        programs[b] = {"path": rel, "sha256": h["sha"](md / rel)}
    write(md / "cohort-seal.json", {"programs": programs, "inputs_sha256": h["sha"](md / "inputs.json")})
    summaries = {"math": {"source_job_id": "synthetic-math", "math_transitions": 504, "native_forwards": 0,
        "cohort_seal_sha256": h["sha"](md / "cohort-seal.json"), "inputs_sha256": h["sha"](md / "inputs.json"), "programs": programs},
        "native": {"rows": rows, "counts": report["counts"]}}
    requests = {}
    for lane, directory, data in (("math", math_run, md), ("native", native, nd)):
        config = {"schema": "synthetic-" + lane, "transport_limits": {"max_expanded_bytes": 4 << 30, "max_member_bytes": 1 << 30}}
        if lane == "native":
            config.update(counts=report["counts"], math_job_id="synthetic-math", math_cohort_sha256=summaries["math"]["cohort_seal_sha256"])
        payload = directory / "payload"
        payload.mkdir()
        worker = "flux_latent_direction_cross.py" if lane == "math" else "worker.py"
        (payload / worker).write_text("# Synthetic bytes only; never executed\n")
        req = {"config": config, "source_hashes": {worker: h["sha"](payload / worker)}, "runtime_wheels": {}}
        req["request_sha256"] = hashlib.sha256(json.dumps(req, sort_keys=True, allow_nan=False).encode()).hexdigest()
        requests[lane] = req
        job_id = "synthetic-" + lane
        write(directory / "request.json", req)
        summary = summaries[lane]
        if lane == "native":
            summary.update(job_id=job_id, request_sha256=req["request_sha256"])
        write(data / "summary.json", summary)
        write(data / "source.json", {"job_id": job_id, "request_sha256": req["request_sha256"], "source_hashes": req["source_hashes"]})
        write(data / "plan.json", config)
        inv = h["inventory"](data)
        write(data / "remote-manifest.json", {"schema": "mdb-latent-direction-math-manifest-v1" if lane == "math" else "mdb-step-boundary-remote-manifest-v1",
            "job_id": job_id, "request_sha256": req["request_sha256"], "files": inv["files"]})
        inv = h["inventory"](data)
        for name in ("collection-local-inventory.json", "collection-remote-before.json", "collection-remote-after.json"):
            write(directory / name, inv)
        log = directory / "worker.log"
        capsule = None
        if lane == "math":
            execution = {"job_id": job_id, "request_sha256": req["request_sha256"], "worker_sha256": req["source_hashes"][worker],
                "mathematical_transitions": 504, "native_framework_forwards": 0, "parent_inventory_unchanged": True}
            result = {"schema": "mdb-latent-direction-math-execution-v1", "output_directory": "latent-direction-math-result",
                "near_data_execution": execution, "report": summary, "remote_manifest_sha256": h["sha"](data / "remote-manifest.json")}
            write(directory / "result.json", result)
            raw = (directory / "result.json").read_bytes()
            compressed = gzip.compress(raw, mtime=0)
            capsule = {"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "gzip_bytes": len(compressed), "gzip_sha256": hashlib.sha256(compressed).hexdigest()}
            log.write_text("MDB_FACTORIAL_AUDIT_CAPSULE=" + json.dumps({**capsule, "base64_gzip": base64.b64encode(compressed).decode()}) + "\n")
        else:
            log.write_text("Synthetic fixture only; no scheduler job.\n")
        executed = {"host": "beast", "sha256": "a" * 64, "size_bytes": 17}
        job = {"job_id": job_id, "assigned_host": "beast", "state": "succeeded", "config": req, "submitted_request": {"config": req},
            "payload_custody": {k: executed for k in ("declared", "sealed", "executed")}, "result": {"returncode": 0, "status": "ok", "executed_payload": executed,
                "diagnostics": {"cwd": "/mnt/hdd8tb/mrun-work/" + job_id, "command": ["synthetic"]},
                "artifacts": {"job_log": {"sha256": h["sha"](log), "bytes": log.stat().st_size}}}}
        write(directory / "job.json", job)
        write(directory / "admission-command.json", {"command": ["synthetic"]})
        collection = {"schema": "mdb-latent-direction-math-collection-v1" if lane == "math" else "mdb-latent-direction-native-collection-v1",
            "job_id": job_id, "request_sha256": req["request_sha256"], "archive_sha256": None, "file_count": inv["file_count"],
            "expanded_bytes": inv["expanded_bytes"], "manifest_sha256": h["sha"](data / "remote-manifest.json"),
            "bindings": {name: h["sha"](directory / name) for name in ("request.json", "job.json", "worker.log", "admission-command.json")}}
        if capsule:
            collection["capsule"] = capsule
        write(directory / "collection.json", collection)
    write(analysis / "analysis.json", report)
    write(analysis / "analysis-receipt.json", {"schema": "mdb-latent-direction-analysis-receipt-v1", "source_sha256": h["sha"](HERE / "qualification/analyze_flux_latent_direction_cross.py"),
        "native_execution_status": report["native_execution_status"], "manifest_origin": report["manifest_origin"],
        "report_sha256": h["sha"](analysis / "analysis.json"), "inputs": {str(d / "collection.json"): h["sha"](d / "collection.json") for d in (native, math_run)},
        "model_calls": 0, "independent_predictor_recomputation": False, "original_native_and_math_inventories_unchanged": True})
    return {"module": module, "h": h, "p": p, "native": native, "math_run": math_run, "analysis": analysis, "report": report, "root": tmp_path}


def test_all_five_figures_full_signed_accounts_byte_exact(fixture):
    f, m = fixture, fixture["module"]
    out = f["root"] / "publication"
    result = m["publish"](f["native"], math_run=f["math_run"], analysis=f["analysis"], output=out)
    assert result["actual_original_PNGs"] == 12 and result["summary_is_byte_exact_analysis"] is True
    assert (out / "summary.json").read_bytes() == (f["analysis"] / "analysis.json").read_bytes()
    qa = f["h"]["read"](out / "layout-qa.json")
    assert len(qa) == 5 and all(v["all_visible_text_inside_canvas"] for v in qa.values())
    for name, entry in qa.items():
        with Image.open(out / name) as image:
            assert list(image.size) == entry["canvas_pixels"]
    assert all(p.stat().st_size > 1000 for p in out.glob("*.png"))
    with pytest.raises(ValueError, match="fresh"):
        m["publish"](f["native"], math_run=f["math_run"], analysis=f["analysis"], output=out)


@pytest.mark.parametrize("member", ["worker.log", "result.json", "extracted/latent-direction-math-result/programs/c1-o0.sham.json"])
def test_cpu_log_capsule_and_sealed_program_tampering_refused(fixture, member):
    f, m = fixture, fixture["module"]
    p = f["math_run"] / member
    p.write_bytes(p.read_bytes() + b"changed")
    _, h, q = m["helpers"]()
    with pytest.raises(ValueError):
        m["verify_math"](h, q, f["math_run"])


def test_analysis_receipt_binds_exact_source_and_report(fixture):
    f, m = fixture, fixture["module"]
    p, h, q = m["helpers"]()
    native = p["producer"](h, f["native"], "latent-direction-native-result", "mdb-latent-direction-native-collection-v1", "job.json")
    mathematical = m["verify_math"](h, q, f["math_run"])
    (f["analysis"] / "analysis.json").write_text("{}\n")
    with pytest.raises(ValueError, match="receipt/source"):
        m["verify_analysis"](h, f["analysis"], f["report"], native, mathematical)


def test_zero_undefined_negative_cosine_and_countertrend_survive(fixture):
    f = fixture
    values = [r["fields"]["image"] for rows in f["report"]["responses"].values() for r in rows if "image" in r["fields"]]
    assert any(v["cosine"] is None for v in values) and any(v["cosine"] < 0 for v in values if v["cosine"] is not None)
    assert any(v["native_response"]["rms"] == 0 for v in values)
    original = copy.deepcopy(f["report"])
    out = f["root"] / "undefined"
    out.mkdir()
    f["module"]["post_single19"](original, out, f["p"]["figure_save"])
    assert original == f["report"]


def test_native_provenance_keeps_failed_scheduler_and_collector_manifest(module):
    record = {"job": {"state": "failed"},
        "collection": {"schema": "mdb-latent-direction-native-recovery-collection-v1"}}
    assert module["native_provenance"](record) == {"native_execution_status": "failed",
        "manifest_origin": "collector-derived-source-manifest"}
    assert record["job"]["state"] == "failed"


def test_analysis_cannot_relabel_actual_native_status(fixture):
    f, m = fixture, fixture["module"]
    p, h, q = m["helpers"]()
    native = p["producer"](h, f["native"], "latent-direction-native-result", "mdb-latent-direction-native-collection-v1", "job.json")
    mathematical = m["verify_math"](h, q, f["math_run"])
    wrong = copy.deepcopy(f["report"])
    wrong["native_execution_status"] = "failed"
    write(f["analysis"] / "analysis.json", wrong)
    receipt = h["read"](f["analysis"] / "analysis-receipt.json")
    receipt.update(report_sha256=h["sha"](f["analysis"] / "analysis.json"), native_execution_status="failed")
    write(f["analysis"] / "analysis-receipt.json", receipt)
    with pytest.raises(ValueError, match="status/manifest"):
        m["verify_analysis"](h, f["analysis"], wrong, native, mathematical)


def recover_fixture(f, *, corrupt=None):
    """Use the real recovery helper with synthetic scheduler/SSH responses."""
    m, h = f["module"], f["h"]
    launch = runpy.run_path(str(HERE / "qualification/submit_flux_latent_direction_native.py"))
    custody = launch["custody"]()
    directory = f["native"]
    data = directory / "extracted/latent-direction-native-result"
    request = h["read"](directory / "request.json")
    # Opaque synthetic bytes qualify identity plumbing only, never an import.
    runtime = directory / "payload/synthetic-runtime.whl"
    runtime.write_bytes(b"Synthetic runtime-custody fixture; not an executable wheel.\n")
    request["runtime_wheels"] = {runtime.name: h["sha"](runtime)}
    request["config"].update(conditions=list(m["CELLS"]),
        expected_parameter_artifacts_module_sha256=h["sha"](data / "retrieve_weights.py"))
    counts = {**f["report"]["counts"], "images": 12}
    request["config"]["counts"] = counts
    request["request_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    write(directory / "request.json", request)
    job = h["read"](directory / "job.json")
    job_id = "job-synthetic-recovered"
    job.update(job_id=job_id, state="failed", config=request, submitted_request={"config": request})
    job["result"].update(status="failed", returncode=1)
    job["result"]["diagnostics"]["cwd"] = "/mnt/hdd8tb/mrun-work/" + job_id
    raw_log = b"Synthetic finalizer failure after complete fixture; no scheduler/model.\nKeyError: 'finalize_manifest'\n"
    job["result"]["artifacts"]["job_log"] = {"sha256": hashlib.sha256(raw_log).hexdigest(), "bytes": len(raw_log)}
    summary = h["read"](data / "summary.json")
    summary.update(job_id=job_id, request_sha256=request["request_sha256"], counts=counts)
    source = h["read"](data / "source.json")
    source.update(job_id=job_id, request_sha256=request["request_sha256"], source_integrity={"unchanged": True})
    write(data / "source.json", source)
    write(data / "summary.json", summary)
    write(data / "plan.json", request["config"])
    write(data / "failure.json", {"error": "'finalize_manifest'", "error_type": "KeyError",
        "scope": "Failed finite attempt; no retry or numeric result rewriting."})
    for branch in m["BRANCHES"]:
        (data / "native").mkdir(exist_ok=True)
        (data / "native" / (branch + ".jsonl")).write_text("{}\n" * 84)
        write(data / "rows" / (branch + ".json"), {"native_replay_diagnostics": [{}] * 84 if branch.endswith(".sham") else []})
    (data / "receipt-emissions.jsonl").write_text("{}\n" * 516)
    for cell in m["CELLS"]:
        write(data / "transactions" / cell / "verification.json", {"phase": "closed", "verified": True})
    (data / "remote-manifest.json").unlink()
    if corrupt == "failure":
        write(data / "failure.json", {"error": "not-finalization", "error_type": "RuntimeError", "scope": "Synthetic"})
    if corrupt == "incomplete":
        (data / "native" / (m["BRANCHES"][0] + ".jsonl")).unlink()
    template = f["root"] / "synthetic-remote-tree"
    data.rename(template)
    for name in ("job.json", "worker.log", "collection.json", "collection-local-inventory.json",
            "collection-remote-before.json", "collection-remote-after.json"):
        (directory / name).unlink()

    class API:
        def request(self, method, endpoint, **kwargs):
            assert method == "GET" and endpoint.startswith("/api/jobs/" + job_id)
            value = (b"[]" if endpoint.endswith("/events") else raw_log if "/logs?" in endpoint
                else json.dumps(job, sort_keys=True).encode())
            return 200, value, {}

    def runner(command, **kwargs):
        if "ssh" in command and "rsync" not in command:
            return SimpleNamespace(stdout=json.dumps(custody["inventory"](template)), stderr="", returncode=0)
        assert "rsync" in command
        shutil.copytree(template, data, dirs_exist_ok=True)
        return SimpleNamespace(stdout="Synthetic local fixture copy; no SSH process executed.", stderr="", returncode=0)

    custody["recover"](directory, api=API(), runner=runner, job_id=job_id)
    f["report"].update(native_job_id=job_id, counts=counts, native_execution_status="failed",
        manifest_origin="collector-derived-source-manifest")
    write(f["analysis"] / "analysis.json", f["report"])
    receipt = h["read"](f["analysis"] / "analysis-receipt.json")
    receipt.update(report_sha256=h["sha"](f["analysis"] / "analysis.json"), native_execution_status="failed",
        manifest_origin="collector-derived-source-manifest")
    receipt["inputs"][str(directory / "collection.json")] = h["sha"](directory / "collection.json")
    write(f["analysis"] / "analysis-receipt.json", receipt)
    return data


def test_recovered_publication_preserves_failed_status_and_external_manifest(fixture):
    f = fixture
    data = recover_fixture(f)
    out = f["root"] / "recovered-publication"
    f["module"]["publish"](f["native"], math_run=f["math_run"], analysis=f["analysis"], output=out)
    receipt = f["h"]["read"](out / "publication-receipt.json")
    assert receipt["native_execution_status"] == "failed"
    assert receipt["manifest_origin"] == "collector-derived-source-manifest"
    assert f["h"]["read"](f["native"] / "job.json")["state"] == "failed"
    assert not (data / "remote-manifest.json").exists()
    assert (f["native"] / "collection-source-manifest.json").is_file()
    assert (out / "summary.json").read_bytes() == (f["analysis"] / "analysis.json").read_bytes()


@pytest.mark.parametrize("corrupt", ["failure", "incomplete"])
def test_recovery_refuses_wrong_failure_or_incomplete_job(fixture, corrupt):
    with pytest.raises(ValueError):
        recover_fixture(fixture, corrupt=corrupt)
