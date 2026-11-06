"""Synthetic saved-custody/render checks; no model/runtime imports or targets."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
HERE = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def module():
    return runpy.run_path(str(HERE / "qualification/publish_flux_native_corner.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(value, denominator=None):
    value = np.asarray(value, dtype=np.float64)
    rms = float(np.sqrt(np.mean(value * value)))
    d = None if denominator is None else float(np.sqrt(np.mean(np.square(denominator))))
    return {"elements": value.size, "rms": rms, "squared_l2": float(np.square(value).sum()), "signed_mean": float(value.mean()),
        "max_abs": float(np.abs(value).max()), "mae": float(np.abs(value).mean()), "p95_abs": float(np.percentile(np.abs(value), 95)),
        "p99_abs": float(np.percentile(np.abs(value), 99)), "changed_values": int(np.count_nonzero(value)),
        "denominator_rms": d, "relative_rms": rms / d if d else None}


def account(module, p, r, m, n):
    vectors = dict(zip(module["TERMS"], (p - r, r - m, m - n)))
    pairs = {a + "__" + b: {"twice_dot": float(2 * np.sum(vectors[a] * vectors[b]))}
        for i, a in enumerate(module["TERMS"]) for b in module["TERMS"][i + 1:]}
    total = float(np.square(p - n).sum())
    return {"terms": {key: stats(value, n) for key, value in vectors.items()}, "pairwise_cross_terms": pairs,
        "P_minus_N_old": stats(p - n, n), "P_minus_M_current": stats(p - m, m), "vector_identity_max_abs": 0.,
        "squared_norm_identity": {"total_squared_l2": total, "three_terms_plus_three_cross_terms": total, "residual": 0.}, "scope": "Synthetic signed fixture"}


@pytest.fixture
def fixture(module, tmp_path):
    h, _ = module["helpers"]()
    native, parent, matched, analysis = [tmp_path / name for name in ("native", "parent", "matched", "analysis")]
    data = native / "extracted/flux-native-corner-result"
    old = parent / "extracted/flux-prompt-factorial-result"
    qdata = matched / "extracted/matched-pass-output"
    for directory in (data, old, qdata, analysis):
        directory.mkdir(parents=True)
    configs = {"parent": {**{"max_" + key: h["COUNTS"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")},
            "transport_limits": {"max_expanded_bytes": 8 << 30, "max_member_bytes": 1 << 30}},
        "native": {"counts": {"adapter_steps": 336, "acts": 4, "receipt_emissions": 340, "vae_calls": 4},
            "transport_limits": {"max_expanded_bytes": 4 << 30, "max_member_bytes": 1 << 30}}}
    manifests, summaries = {}, {"parent": {"rows": {}}, "native": {"rows": {}}}
    arrays = {"P": np.array([4., 5.]), "R": np.array([3., 4.]), "M": np.array([2., 3.]), "N": np.array([2.25, 3.25])}
    report = {"schema": "mdb-native-corner-analysis-v1", "job_id": "synthetic-native", "request_sha256": None,
        "mechanics_status": "authenticated-saved-evidence-reduced", "model_calls": 0, "independent_predictor_recomputation": False,
        "counts": configs["native"]["counts"], "responses": {}, "continuation": {}, "input_changes": {}, "native_replay": {},
        "expectations": [{"holds": False, "scope": "Synthetic observation retained"}], "original_expectations": {"synthetic": True},
        "native_transactions": {"synthetic": True}}
    bindings = {}
    for cell in module["CELLS"]:
        rows = []
        for local, fields in ((0, ("image", "text", "temb", "img_mod", "txt_mod", "single_mod", "rope_cos", "rope_sin")),
                (5, ("image", "text")), (25, ("image", "text")), (26, ("noise",)), (27, ("latent",))):
            entries = {}
            for field in fields:
                sources = {}
                for role, directory in (("P", old), ("Q", qdata), ("R", data), ("M", data)):
                    filename = "arrays/" + cell + "-" + str(local) + "-" + field + "-" + role + ".npy"
                    path = directory / filename
                    path.parent.mkdir(exist_ok=True)
                    np.save(path, np.arange(6, dtype=np.float32).reshape(2, 3))
                    sources[role] = {"archive": {"path": filename, "sha256": module["sha"](path)}, "descriptor": {"shape": [2, 3]}}
                dg, df = np.array([1., -2.]), np.array([2., -1.])
                entries[field] = {"sources": sources, "account": {"D_G_P_minus_Q": stats(dg), "D_F_R_minus_M": stats(df),
                    "error_DG_minus_DF": stats(dg - df, df), "cosine_DG_DF": .8, "dot_DG_DF": 4.,
                    "native_finite_response_ratio": 2., "mathematical_finite_response_ratio": 2.,
                    "input_latent_discrepancy_rms": 1., "native_field_rms": 10.}}
            rows.append({"logical_step": 1, "local_index": local, "fields": entries})
        report["responses"][cell] = rows
        png_refs, pngs = [], {}
        for role, directory in (("P", old), ("R", data), ("M", data), ("N", old)):
            pixels = np.full((512, 512, 3), {"P": 104, "R": 103, "M": 102, "N": 105}[role], dtype=np.uint8)
            path = directory / (cell + "-" + role + ".png")
            Image.fromarray(pixels).save(path)
            ref = {"path": path.name, "sha256": module["sha"](path)}
            png_refs.append(ref)
            pngs[role] = pixels.astype(np.float64)
            lane = "parent" if role in ("P", "N") else "native"
            summaries[lane]["rows"][cell + {"P": ".predicted", "R": ".P1", "M": ".N1.sham", "N": ".native"}[role]] = {"image": ref}
        report["continuation"][cell] = {"latent": [{"logical_step": step, "field": "latent", "account": account(module, **{k.lower(): v for k, v in arrays.items()})} for step in (1, 2, 3)],
            "RGB": account(module, **{k.lower(): v for k, v in pngs.items()}), "PNG_sources": png_refs}
        report["input_changes"][cell] = stats(np.array([1., -1.]))
        report["native_replay"][cell] = {"terminal_png_exact": False, "mismatches": [{"position": 0}]}
        tx = old / "transactions" / cell
        write(tx / "forecasts/program.json", {"synthetic": True})
        write(tx / "seal.json", {"synthetic": True})
        bindings[cell] = {"P1_source": {"original_program_file_sha256": module["sha"](tx / "forecasts/program.json"),
            "original_seal_sha256": module["sha"](tx / "seal.json"), "original_continuation_transition_range": [28, 111],
            "original_complete_forecast_trajectory_fingerprint": "a" * 64}}
    write(data / "binding.json", bindings)
    write(data / "response-contract.json", {"DF": "R-M", "scope": "Synthetic"})
    retriever = HERE / "src/mdb/parameter_artifacts.py"
    (data / "retrieve_weights.py").write_bytes(retriever.read_bytes())
    write(data / "parameter-artifacts.json", {"artifacts": [{"path": "absent-weight-" + str(i)} for i in range(169)]})
    write(matched / "result.json", {"synthetic": True})
    write(matched / "job.json", {"job_id": "synthetic-Q"})
    write(matched / "request.json", {"synthetic": True})
    write(matched / "collection.json", {"synthetic": True})
    configs["native"].update(parent_job_id="synthetic-parent", matched_job_id="synthetic-Q",
        matched_result_sha256=module["sha"](matched / "result.json"))
    for lane, directory, output_data, job_file, collection_schema in (("parent", parent, old, "collection-job.json", "mdb-prompt-factorial-ssh-collection-v1"),
            ("native", native, data, "job.json", "mdb-native-corner-collection-v1")):
        payload = directory / "payload"
        payload.mkdir()
        (payload / "worker.py").write_text("# synthetic saved-only fixture, no execution\n")
        wheels = {}
        for wheel in ("mdb.whl", "saturn.whl", "mrun.whl"):
            (payload / wheel).write_text("Synthetic bytes; not an executable wheel\n")
            wheels[wheel] = module["sha"](payload / wheel)
        request = {"config": configs[lane], "source_hashes": {"worker.py": module["sha"](payload / "worker.py")}, "runtime_wheels": wheels}
        if lane == "parent":
            request["schema"] = "mdb-flux-prompt-factorial-request-v1"
            request.pop("runtime_wheels")
        request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
        job_id = "synthetic-" + lane
        log_name = "collection-worker.log" if lane == "parent" else "worker.log"
        log = directory / log_name
        log.write_text("Synthetic fixture only; no scheduler was called.\n")
        executed = {"host": "beast", "sha256": "a" * 64, "size_bytes": 23}
        job = {"assigned_host": "beast", "state": "succeeded", "job_id": job_id, "config": request,
            "submitted_request": {"config": request}, "payload_custody": {key: executed for key in ("declared", "sealed", "executed")},
            "result": {"returncode": 0, "status": "ok", "executed_payload": executed,
                "diagnostics": {"cwd": "/mnt/hdd8tb/mrun-work/" + job_id, "command": ["synthetic-command"]},
                "artifacts": {"job_log": {"sha256": module["sha"](log), "bytes": log.stat().st_size}}}}
        write(directory / "request.json", request)
        write(directory / job_file, job)
        write(directory / "admission-command.json", {"command": ["synthetic-command"]})
        summaries[lane].update(job_id=job_id, request_sha256=request["request_sha256"], counts=h["COUNTS"] if lane == "parent" else configs[lane]["counts"])
        write(output_data / "summary.json", summaries[lane])
        write(output_data / "source.json", {"job_id": job_id, "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"]})
        write(output_data / "plan.json", configs[lane])
        inv = h["inventory"](output_data)
        write(output_data / "remote-manifest.json", {"schema": "mdb-step-boundary-remote-manifest-v1", "job_id": job_id,
            "request_sha256": request["request_sha256"], "files": inv["files"]})
        inv = h["inventory"](output_data)
        for name in ("collection-local-inventory.json", "collection-remote-before.json", "collection-remote-after.json"):
            write(directory / name, inv)
        collection = {"schema": collection_schema, "archive_sha256": None, "job_id": job_id, "request_sha256": request["request_sha256"],
            "file_count": inv["file_count"], "expanded_bytes": inv["expanded_bytes"], "manifest_sha256": module["sha"](output_data / "remote-manifest.json"),
            "bindings": {name: module["sha"](directory / name) for name in ("request.json", job_file, log_name, "admission-command.json")}}
        write(directory / "collection.json", collection)
        manifests[lane] = module["producer"](h, directory, output_data.name, collection_schema, job_file, expected_runtime_wheels=wheels)
        if lane == "parent":
            configs["native"]["parent_request_sha256"] = request["request_sha256"]
    report["request_sha256"] = manifests["native"]["request"]["request_sha256"]
    report["bindings"] = {"native_collection_sha256": module["sha"](native / "collection.json"),
        "native_summary_sha256": module["sha"](data / "summary.json"), "parent_collection_sha256": module["sha"](parent / "collection.json"),
        "matched_collection_sha256": module["sha"](matched / "collection.json"), "analyzer_source_sha256": module["ANALYZER_SHA256"]}
    write(analysis / "analysis.json", report)
    write(analysis / "analysis-receipt.json", {"schema": "mdb-native-corner-analysis-receipt-v1", "report_sha256": module["sha"](analysis / "analysis.json"),
        "analyzer_source_sha256": module["ANALYZER_SHA256"], "inputs": {str(native / "collection.json"): module["sha"](native / "collection.json")},
        "model_calls": 0, "independent_predictor_recomputation": False, "original_native_inventory_unchanged": True})
    return {"h": h, "native": manifests["native"], "parent": manifests["parent"], "matched": matched, "analysis": analysis,
        "report": report, "module": module, "data": data, "root": tmp_path}


def test_complete28_accounts_images_and_all_visible_labels(fixture, monkeypatch):
    m, f = fixture["module"], fixture
    m["verify_analysis"](f["h"], f["analysis"], f["report"], f["native"], f["parent"], f["matched"])
    values, sources = m["images"](f["h"], f["native"], f["parent"], f["report"])
    out = f["root"] / "figures"
    # Q's actual capsule validator is separately frozen/qualified. This fixture
    # exercises the entire new publisher using a clearly synthetic Q join.
    q = {"verify_run": lambda *_: [None, {"parent_job_id": "synthetic-parent",
        "parent_request_sha256": f["parent"]["request"]["request_sha256"]}]}
    monkeypatch.setitem(m["publish"].__globals__, "helpers", lambda: (f["h"], q))
    result = m["publish"](f["native"]["run"], analysis=f["analysis"], parent=f["parent"]["run"], matched=f["matched"], output=out)
    qa = m["read"](out / "layout-qa.json")
    assert len(qa) == 5 and sum(len(r["fields"]) for rows in f["report"]["responses"].values() for r in rows) == 28
    assert all(row["all_visible_text_inside_canvas"] and row["checked_text"] for row in qa.values())
    assert len(sources) == 2 and all(len(v) == 4 for v in sources.values())
    assert result["response_fields"] == 28 and result["continuation_accounts"] == 8
    assert all(p.stat().st_size > 1000 for p in out.glob("*.png"))
    assert m["read"](out / "summary.json")["continuation"] == f["report"]["continuation"]
    with pytest.raises(ValueError, match="fresh"):
        m["publish"](f["native"]["run"], analysis=f["analysis"], parent=f["parent"]["run"], matched=f["matched"], output=out)


@pytest.mark.parametrize("member", ["worker.log", "admission-command.json", "extracted/flux-native-corner-result/source.json"])
def test_actual_log_command_and_tree_tampering_refused(fixture, member):
    f, m = fixture, fixture["module"]
    path = f["native"]["run"] / member
    path.write_bytes(path.read_bytes() + b" \n")
    with pytest.raises(ValueError):
        m["producer"](f["h"], f["native"]["run"], "flux-native-corner-result", "mdb-native-corner-collection-v1", "job.json")


def test_analysis_receipt_hash_and_complete_cross_terms_refused(fixture):
    f, m = fixture, fixture["module"]
    bad = copy.deepcopy(f["report"])
    bad["continuation"][m["CELLS"][0]]["RGB"]["pairwise_cross_terms"].popitem()
    with pytest.raises(ValueError, match="three-term"):
        m["verify_analysis"](f["h"], f["analysis"], bad, f["native"], f["parent"], f["matched"])
    (f["analysis"] / "analysis.json").write_text("{}\n")
    with pytest.raises(ValueError, match="receipt"):
        m["verify_analysis"](f["h"], f["analysis"], f["report"], f["native"], f["parent"], f["matched"])


def test_wrong_future_and_source_program_refused(fixture):
    f, m = fixture, fixture["module"]
    bad = copy.deepcopy(f["report"])
    bad["continuation"][m["CELLS"][0]]["PNG_sources"][1] = bad["continuation"][m["CELLS"][0]]["PNG_sources"][2]
    with pytest.raises(ValueError, match="future PNG"):
        m["images"](f["h"], f["native"], f["parent"], bad)
    path = f["parent"]["data"] / "transactions" / m["CELLS"][0] / "forecasts/program.json"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="program binding"):
        m["verify_analysis"](f["h"], f["analysis"], f["report"], f["native"], f["parent"], f["matched"])


def test_legacy_wheel_pins_must_come_from_new_native_request(fixture):
    f, m = fixture, fixture["module"]
    args = (f["h"], f["parent"]["run"], "flux-prompt-factorial-result", "mdb-prompt-factorial-ssh-collection-v1", "collection-job.json")
    with pytest.raises(ValueError, match="explicit wheel pins"):
        m["producer"](*args)
    bad = dict(f["native"]["request"]["runtime_wheels"])
    bad["mdb.whl"] = "0" * 64
    with pytest.raises(ValueError, match="source/runtime changed"):
        m["producer"](*args, expected_runtime_wheels=bad)


def test_undefined_ratios_and_negative_cross_terms_remain_visible(fixture):
    f, m = fixture, fixture["module"]
    report = copy.deepcopy(f["report"])
    first = report["responses"][m["CELLS"][0]][0]["fields"]["image"]["account"]
    first.update(cosine_DG_DF=None, native_finite_response_ratio=None, mathematical_finite_response_ratio=None)
    assert any(value["twice_dot"] < 0 for value in report["continuation"][m["CELLS"][0]]["RGB"]["pairwise_cross_terms"].values())
    out = f["root"] / "undefined"
    out.mkdir()
    qa = m["plot_responses"](report, out)
    assert len(qa) == 3 and first["cosine_DG_DF"] is None


def test_figure_regression_refuses_text_outside_canvas(module, tmp_path):
    fig = module["plt"].figure(figsize=(2, 2))
    fig.text(-.5, .5, "Deliberately clipped fixture label")
    with pytest.raises(ValueError, match="outside canvas"):
        module["figure_save"](fig, tmp_path / "bad.png")
    module["plt"].close(fig)
