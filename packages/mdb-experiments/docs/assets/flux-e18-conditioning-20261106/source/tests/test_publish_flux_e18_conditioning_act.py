"""Saved scalar/PNG fixture checks; no model, raw-array or tokenizer execution."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/publish_flux_e18_conditioning_act.py"
P = runpy.run_path(str(PATH))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def metric(rms=0., native=1., count=3):
    return dict(rms=rms, relative_rms=None if native == 0 else rms / native, native_rms=native,
                predicted_rms=native + rms, signed_mean=-rms / 2, native_signed_mean=-native / 4,
                predicted_signed_mean=-native / 4 - rms / 2, cosine=None if native == 0 else 1.,
                maxabs=2 * rms, p95=1.5 * rms, p99=1.9 * rms, sample_count=count,
                changed_values=0 if rms == 0 else 1, bitwise_exact=rms == 0, shape=[1, 1, count])


def reseal(pack, *, wrapped=False):
    entries = {path.relative_to(pack).as_posix(): P["sha"](path) for path in pack.rglob("*") if path.is_file() and path != pack / "files.json"}
    summary = P["read"](pack / "summary.json")
    wrapper = dict(schema="mdb-image-tail-files-v1", job_id=summary["job_id"], request_sha256=summary["request_sha256"],
                   files={name: dict(sha256=digest, bytes=(pack / name).stat().st_size) for name, digest in entries.items()}) if wrapped else entries
    put(pack / "files.json", wrapper)
    receipt = pack.parent / "collection.json"
    put(receipt, dict(files_manifest_sha256=P["sha"](pack / "files.json"), job_id=summary["job_id"],
                     request_sha256=summary["request_sha256"], file_count=len(entries) + 1,
                     expanded_bytes=sum(path.stat().st_size for path in pack.rglob("*") if path.is_file())))
    return P["sha"](pack / "files.json"), receipt


def fixture(root, *, effect=.25, miss=.01):
    pack = root / "compact"
    pack.mkdir(parents=True)
    counts = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=2,
                  functional_transitions=224, native_transitions=224, functional_decode_calls=2,
                  native_vae_calls=4, native_qwen_calls=1, native_module_entries=1000,
                  guard_refusals=0, act_receipts=4, receipt_occurrences=228)
    native_config = dict(schema="mdb-flux-e18-conditioning-plan-v1", seed=277, height=20, width=20,
                         branches=list(P["CASES"]), conditioning_columns=[2560, 5120],
                         qwen_manifest_sha256="0" * 64)
    native_request = dict(config=native_config, source_hashes={"flux_e18_conditioning_act.py": P["WORKER_SHA256"]})
    native_request["request_sha256"] = P["identity"](native_request)
    native_id, native_job = native_request["request_sha256"], "job-native-fixture"
    put(pack / "source-request.json", native_request)
    put(pack / "source-plan.json", {**native_config, "request_sha256": native_id, "output": "/fixture/raw"})
    source = dict(job_id=native_job, request_sha256=native_id, worker_sha256=P["WORKER_SHA256"],
                  helper_source_sha256=P["NUMERICAL_PINS"], source_files={})
    put(pack / "source-runtime-contract.json", dict(selected_kernel="unsupported instrument diagnostic"))
    source["runtime_contract_sha256"] = P["sha"](pack / "source-runtime-contract.json")
    put(pack / "source-source.json", source)
    put(pack / "source-summary.json", dict(job_id=native_job, request_sha256=native_id, counts=counts))
    put(pack / "source-native-contract.json", dict(counts=counts))
    events = []
    for index, phase in enumerate(("qwen", "flux", "vae", "qwen", "flux", "vae")):
        if index == 3:
            events.append(dict(kind="forecast_globally_sealed", seconds=6.5, native_module_entries=0))
        events.extend([dict(kind="component_placement", component=phase, direction="cuda", seconds=2. * index + (1 if index >= 3 else 0)),
                       dict(kind="component_placement", component=phase, direction="cpu", seconds=2. * index + 1 + (1 if index >= 3 else 0))])
    (pack / "source-timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    raw = {name[7:]: dict(sha256=P["sha"](path), bytes=path.stat().st_size) for path in pack.iterdir()
           if path.is_file() and (name := path.name).startswith("source-")}
    for case_index, case in enumerate(P["CASES"]):
        for lane_index, lane in enumerate(P["LANES"]):
            name = case + "-" + lane + ".png"
            Image.new("RGB", (20, 20), (30 + case_index * 40, 70 + lane_index * 20, 120)).save(pack / name)
            raw["cases/" + case + "/decoder/" + lane + "/image.png"] = dict(sha256=P["sha"](pack / name), bytes=(pack / name).stat().st_size)
    original = dict(schema="mdb-image-tail-files-v1", job_id=native_job, request_sha256=native_id,
                    files=raw, expanded_bytes=sum(row["bytes"] for row in raw.values()))
    put(pack / "source-files.json", original)
    shutil.copyfile(HERE / "qualification/retrieve_reader_evidence.py", pack / "retrieve_evidence.py")
    put(pack / "external-evidence.json", dict(host="beast", roots=dict(native="/fixture/raw"),
        manifests=dict(native={name: row["sha256"] for name, row in raw.items()})))
    rows, responses = {case: [] for case in P["CASES"]}, []
    routes = {name: {lane: metric(effect if name == "E18-target" else 0.) for lane in ("forecast", "native")}
              for name in ("E9-collateral", "E18-target", "E27-collateral")}

    def response_field(amount=effect):
        error = miss if amount else 0.
        return dict(forecast=metric(amount + error), native=metric(amount), signed_response_error=metric(error, native=amount),
                    prospective_full_hash_verified=True)

    for number in range(-1, 112):
        local = number % 28
        clock = dict(step=0 if number == -1 else number // 28 + int(local == 27),
                     phase="project" if number == -1 or local == 27 else "scheduler" if local == 26 else "single" if local >= 5 else "joint",
                     cursor=0 if number == -1 or local == 27 else local)
        fields = {"latent", "conditioning"}
        if clock["phase"] != "project":
            fields |= {"image", "text"}
        if local == 26:
            fields.add("noise_prediction")
        signed = {name: response_field(0. if number == -1 and name == "latent" else effect) for name in fields}
        signed["conditioning"]["declared_target_and_collateral_routes"] = routes
        responses.append(dict(number=number, clock=clock, fields=signed))
        if number >= 0:
            for case in P["CASES"]:
                rows[case].append(dict(number=number, after=clock, fields={name: metric(miss if case == "e18-zero" else 0.) for name in fields}))
    acts = {lane: {case: dict(changed_values=0 if case == "sham" or effect == 0 else 1, selected_values=1310720,
                             receipt=hashlib.sha256((lane + case).encode()).hexdigest()) for case in P["CASES"]} for lane in ("forecast", "native")}
    audit = dict(schema="mdb-flux-e18-conditioning-saved-audit-v1", mechanics_status="authenticated-genuine-Acts-and-own-state-continuations",
        job_id=native_job, request_sha256=native_id, raw_manifest_sha256=P["sha"](pack / "source-files.json"),
        raw_members=len(raw), raw_expanded_bytes=original["expanded_bytes"], source=source, new_model_calls=0, new_tokenizer_calls=0,
        actual_counts=counts, acts=acts, receipt_account=dict(actual_acts=4, actual_native_advances=224, total_receipt_occurrences=228),
        context_consumers=dict(actual_occurrences=8), qwen=dict(fields={str(index): dict(full=metric()) for index in range(29)}, first_bit_discrepancy=None),
        cases={case: dict(denoiser=dict(boundaries=rows[case], first_bit_discrepancy=None if case == "sham" or miss == 0 else dict(number=0, field="image"),
                                        first_value_discrepancy=None if case == "sham" or miss == 0 else dict(number=0, field="image")),
                          decoder=dict(first_retained_same_latent_discrepancy=None if miss == 0 else "decoded")) for case in P["CASES"]},
        signed_responses=dict(denoiser=responses, decoder={name: response_field(effect) for name in ("decoded", "pixels")}))
    put(pack / "audit.json", audit)
    queries = []
    for index in range(4):
        fingerprint = hashlib.sha256(str(index).encode()).hexdigest()
        put(pack / "catalog/equations" / (fingerprint + ".json"), dict(fingerprint=fingerprint,
            evidence=dict(audit_sha256=P["sha"](pack / "audit.json")), tested_context=dict(branch=P["CASES"][index // 2])))
        queries.append(dict(fingerprint=fingerprint, exact=1, wrong_clock=0))
    put(pack / "catalog/provenance.json", dict(records=4, queries=queries, automatic_family_induction=False))
    request = dict(schema="mdb-saved-flux-audit-request-v1", config=dict(native_manifest_sha256=audit["raw_manifest_sha256"],
                   native_request_sha256=native_id, source_job=native_job),
                   source_hashes={"audit_flux_e18_conditioning_act.py": P["AUDITOR_SHA256"], **P["READER_PINS"]})
    request["request_sha256"] = P["identity"](request)
    put(pack / "request.json", request)
    put(pack / "summary.json", dict(schema="mdb-flux-e18-conditioning-compact-v1", model_calls=0, cases=2, genuine_acts=4,
        native_advances=224, records=4, source_job=native_job, job_id="job-audit-fixture", request_sha256=request["request_sha256"]))
    qualification = root / "qualification.json"
    put(qualification, dict(status="qualified-saved-tiny-mechanics", sources={"qualification/audit_flux_e18_conditioning_act.py": P["AUDITOR_SHA256"]}))
    digest, collection = reseal(pack)
    args = dict(expected_manifest=digest, expected_request=native_id, expected_job=native_job, collection_receipt=collection, qualification=qualification)
    return pack, args


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("e18-publication")
    pack, args = fixture(root)
    output = root / "publication"
    result = P["publish"](pack, output, **args)
    return pack, args, output, result


def test_full_saved_publication_preserves_originals_once_and_misses(completed):
    pack, _args, output, result = completed
    assert result["plots"] == 5
    assert P["sha"](pack / "audit.json") == P["sha"](output / "evidence/audit.json")
    assert not (output / "audit-data.json").exists()
    assert len(list(output.rglob("audit.json"))) == 1
    for case in P["CASES"]:
        for lane in P["LANES"]:
            name = case + "-" + lane + ".png"
            assert P["sha"](pack / name) == P["sha"](output / "evidence" / name)
            assert len(list(output.rglob(name))) == 1
    publication = P["read"](output / "publication.json")
    assert publication["account_summary"]["field_pairs_with_value_misses"] > 0
    assert publication["account_summary"]["first_discrepancies"]["e18-zero"]["bits"]["field"] == "image"
    assert publication["catalog"]["records"] == 4 and publication["catalog"]["dependent"]
    assert publication["actual_counts"]["receipt_occurrences"] == 228
    qa = P["read"](output / "figure-layout.json")
    assert len(qa) == 5 and all(row["within_figure"] for rows in qa.values() for row in rows)
    with pytest.raises(ValueError, match="output exists"):
        P["publish"](pack, output, **_args)


def test_wrong_collection_or_native_identity_refuses_before_output(tmp_path):
    pack, args = fixture(tmp_path)
    put(args["collection_receipt"], {**P["read"](args["collection_receipt"]), "job_id": "wrong"})
    with pytest.raises(ValueError, match="collection"):
        P["publish"](pack, tmp_path / "never", **args)
    assert not (tmp_path / "never").exists()
    with pytest.raises(ValueError, match="expected native"):
        P["authenticate"](pack, **{**args, "expected_job": "wrong"})


def test_compact_png_substitution_cannot_replace_original_consumer(tmp_path):
    pack, args = fixture(tmp_path)
    Image.new("RGB", (20, 20), "red").save(pack / "sham-forecast.png")
    args["expected_manifest"], args["collection_receipt"] = reseal(pack)
    with pytest.raises(ValueError, match="PNG consumer"):
        P["authenticate"](pack, **args)


def test_wrapped_manifest_no_plan_qualification_fallback_and_zero_nulls(tmp_path):
    pack, args = fixture(tmp_path, effect=0., miss=0.)
    put(args["qualification"], dict(returncode=0, source_hashes={"audit_flux_e18_conditioning_act.py": P["AUDITOR_SHA256"]}))
    args["expected_manifest"], args["collection_receipt"] = reseal(pack, wrapped=True)
    data = P["authenticate"](pack, **args)
    plots = P["plot_data"](data["audit"])
    assert not (pack / "plan.json").exists()
    assert plots["decoder"]["pixels"]["signed_response_error"]["relative_rms"] is None
    assert plots["decoder"]["pixels"]["signed_response_error"]["cosine"] is None
    assert data["audit"]["acts"]["native"]["e18-zero"]["changed_values"] == 0
    fig, ax = P["plt"].subplots(figsize=(7, 4), layout="constrained")
    P["line"](ax, plots["responses"], "noise_prediction", "relative_rms", "undefined", "blue", "signed_response_error")
    P["nonnegative"](ax)
    assert all(row["within_figure"] for row in P["finish"](fig, tmp_path / "zero-null.png"))


def test_raw_array_members_and_symlinks_refused(tmp_path):
    pack, args = fixture(tmp_path)
    (pack / "forbidden.npy").write_bytes(b"hostile fixture, not a tensor")
    args["expected_manifest"], args["collection_receipt"] = reseal(pack)
    with pytest.raises(ValueError, match="raw or malformed"):
        P["authenticate"](pack, **args)
    (pack / "forbidden.npy").unlink()
    (pack / "link.json").symlink_to(pack / "audit.json")
    args["expected_manifest"], args["collection_receipt"] = reseal(pack)
    with pytest.raises(ValueError, match="symlink"):
        P["authenticate"](pack, **args)


def test_actual_cli_help_and_manifest_failure_are_model_free(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "--expected-job" in result.stdout and "--qualification" in result.stdout
    pack, args = fixture(tmp_path)
    command = [sys.executable, "-I", "-B", str(PATH), "--pack", str(pack), "--out", str(tmp_path / "never"),
               "--expected-manifest", "0" * 64, "--expected-request", args["expected_request"], "--expected-job", args["expected_job"],
               "--collection-receipt", str(args["collection_receipt"]), "--qualification", str(args["qualification"])]
    failed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert failed.returncode != 0 and "compact manifest differs" in failed.stderr and not (tmp_path / "never").exists()
