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
PATH = HERE / "qualification/publish_flux_h18_state_history.py"
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


def qualification_record():
    return dict(schema="mdb-flux-h18-history-saved-reader-qualification-v1",
        status="qualified-tiny-saved-mechanics-only", reader_sha256=P["AUDITOR_SHA256"],
        source_cohort={"qualification/audit_flux_h18_state_history.py": P["AUDITOR_SHA256"]}, final_attempt=dict(checks=24))


def synthetic_coverage():
    return {key: dict(fields=count, positions=count * 3, bit_exact_fields=count) for key, count in (
        ("primary_qwen_whole_fields", 41), ("primary_denoiser_whole_fields", 1120),
        ("qwen_query_region_diagnostics", 82), ("aligned_first_suffix_port_comparisons", 30),
        ("signed_denoiser_response_fields", 565))}


def synthetic_qwen(miss):
    def odd():
        return dict(oddness={name: dict(elements=3, numerical_changed=1, bit_changed_elements=2, rms=.01) for name in P["PORTS"]},
            plus_finite_addition_remainder=dict(rms=.001), minus_finite_addition_remainder=dict(rms=.002))
    fields = {group: {str(index): dict(full=metric(miss)) for index in range(count)} for group, count in (("common", 28), ("minus", 10), ("conditioning", 3))}
    # Index labels carry the actual native hidden-state convention even in the
    # synthetic authentication envelope; these are not saved tensor descriptors.
    fields["minus"] = {str(index + 18): row for index, row in enumerate(fields["minus"].values())}
    return dict(fields=fields, first_bit_discrepancy=None, genuine_qwen_act=False,
        first_suffix={sign: {name: metric(miss) for name in P["PORTS"]} for sign in ("plus", "minus")},
        first_suffix_first_bit_discrepancy=dict(plus=None, minus="gate" if miss else None),
        oddness={lane: odd() for lane in ("forecast", "native")},
        source_history_response_orientation_note="Actual operands are edited minus baseline; generic raw label preserved.",
        functional_counts=dict(blocks=36), native_counts=dict(baseline_blocks=36, suffix_blocks=9))


def fixture(root, *, effect=.25, miss=.01):
    pack = root / "compact"
    pack.mkdir(parents=True)
    counts = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=2,
                  functional_transitions=224, native_transitions=224, functional_decode_calls=2,
                  native_vae_calls=4, native_qwen_calls=1, native_module_entries=1000,
                  guard_refusals=0, act_receipts=4, receipt_occurrences=228, qwen_functional_blocks=36, qwen_native_blocks=45)
    native_config = dict(schema="mdb-flux-h18-state-history-plan-v1", seed=277, height=20, width=20,
                         branches=list(P["CASES"]), conditioning_columns=[2560, 7680], history=dict(edit_index=18, horizon=27, selected=[9, 18, 27]),
                         qwen_manifest_sha256="0" * 64)
    native_request = dict(config=native_config, source_hashes={"flux_h18_state_history.py": P["WORKER_SHA256"]})
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
    routes = {name: {lane: metric(effect if name == "H27-history-contrast" else 0.) for lane in ("forecast", "native")}
              for name in ("H9-held", "H27-history-contrast", "negative-H18-held")}

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
                rows[case].append(dict(number=number, after=clock, fields={name: metric(miss if case == "coherent" else 0.) for name in fields}))
    acts = {lane: {case: dict(changed_values=0 if case == "direct" or effect == 0 else 1, selected_values=1310720,
                             receipt=hashlib.sha256((lane + case).encode()).hexdigest()) for case in P["CASES"]} for lane in ("forecast", "native")}
    audit = dict(schema="mdb-flux-h18-state-history-saved-audit-v1", mechanics_status="authenticated-genuine-Acts-and-own-state-continuations",
        job_id=native_job, request_sha256=native_id, raw_manifest_sha256=P["sha"](pack / "source-files.json"),
        raw_members=len(raw), raw_expanded_bytes=original["expanded_bytes"], source=source, new_model_calls=0, new_tokenizer_calls=0, qualification_fixture="SYNTHETIC AUTHENTICATION/TRANSPORT FIXTURE · no production outcome",
        actual_counts=counts, acts=acts, receipt_account=dict(actual_acts=4, actual_native_advances=224, total_receipt_occurrences=228),
        context_consumers=dict(actual_occurrences=8), qwen=synthetic_qwen(miss),
        cases={case: dict(denoiser=dict(boundaries=rows[case], first_bit_discrepancy=None if case == "direct" or miss == 0 else dict(number=0, field="image"),
                                        first_value_discrepancy=None if case == "direct" or miss == 0 else dict(number=0, field="image")),
                          decoder=dict(first_retained_same_latent_discrepancy=None if miss == 0 else "decoded")) for case in P["CASES"]},
        signed_responses=dict(denoiser=responses, decoder={name: response_field(effect) for name in ("decoded", "pixels")}))
    audit["coverage"] = synthetic_coverage()
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
                   source_hashes={"audit_flux_h18_state_history.py": P["AUDITOR_SHA256"], **P["READER_PINS"]})
    request["request_sha256"] = P["identity"](request)
    put(pack / "request.json", request)
    put(pack / "summary.json", dict(schema="mdb-flux-h18-state-history-compact-v1", model_calls=0, cases=2, genuine_acts=4,
        native_advances=224, records=4, source_job=native_job, job_id="job-audit-fixture", request_sha256=request["request_sha256"]))
    qualification = root / "qualification.json"
    put(qualification, qualification_record())
    digest, collection = reseal(pack)
    args = dict(expected_manifest=digest, expected_request=native_id, expected_job=native_job, collection_receipt=collection, qualification=qualification)
    return pack, args


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("h18-publication")
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
    assert publication["account_summary"]["qwen_primary_full_field_pairs"] == 41
    assert publication["account_summary"]["first_discrepancies"]["coherent"]["bits"]["field"] == "image"
    assert publication["catalog"]["records"] == 4 and publication["catalog"]["dependent"]
    assert "edited−baseline" in publication["orientation"]
    assert any("no all-nine" in value for value in publication["limits"])
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
    Image.new("RGB", (20, 20), "red").save(pack / "direct-forecast.png")
    args["expected_manifest"], args["collection_receipt"] = reseal(pack)
    with pytest.raises(ValueError, match="PNG consumer"):
        P["authenticate"](pack, **args)


def test_wrapped_manifest_no_plan_qualification_fallback_and_zero_nulls(tmp_path):
    pack, args = fixture(tmp_path, effect=0., miss=0.)
    put(args["qualification"], qualification_record())
    args["expected_manifest"], args["collection_receipt"] = reseal(pack, wrapped=True)
    data = P["authenticate"](pack, **args)
    plots = P["plot_data"](data["audit"])
    assert not (pack / "plan.json").exists()
    assert plots["decoder"]["pixels"]["signed_response_error"]["relative_rms"] is None
    assert plots["decoder"]["pixels"]["signed_response_error"]["cosine"] is None
    assert data["audit"]["acts"]["native"]["coherent"]["changed_values"] == 0
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
    command[command.index("--expected-manifest") + 1] = args["expected_manifest"]
    command[command.index("--out") + 1] = str(tmp_path / "cli-publication")
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["plots"] == 5 and P["sha"](tmp_path / "cli-publication/publication.json") == result["publication_sha256"]


def test_figures_from_actual_tiny_serializers_and_saved_reader_accounts(tmp_path, monkeypatch):
    import os

    from saturn_pub.store import LocalStore

    source = os.environ.get("MDB_H18_SAVED_FIXTURE")
    if not source:
        pytest.skip("Supply the actual producer tiny saved fixture; no models run here.")
    root = Path(source)
    a = runpy.run_path(str(HERE / "qualification/audit_flux_h18_state_history.py"))
    _, old, q, streamer, _ = a["helpers"]()
    spec = a["read"](root / "fixture.json")
    fq, nq = (a["read"](root / (lane + "-qwen-history.json")) for lane in ("forecast", "native"))
    cfg = dict(history=spec["geometry"], max_length=fq["input_ids"]["shape"][1],
        attention_dispatch=fq["profile"]["attention_dispatch"], rope_buffer_policy=fq["profile"]["rope_buffer_policy"])
    qc = dict(num_hidden_layers=nq["counts"]["baseline_blocks"], hidden_size=fq["common"]["0"]["shape"][-1],
        head_dim=nq["sdpa_calls"][0]["QKV"][0]["shape"][-1], num_attention_heads=nq["sdpa_calls"][0]["QKV"][0]["shape"][1],
        num_key_value_heads=fq["first_suffix_ports"]["plus"]["k"]["shape"][1], vocab_size=nq["baseline_logits_shape"][-1])
    qr, own, _ = a["qwen_accounts"](root, fq, nq, cfg, old, q, qc)
    import torch

    own_t = {lane: {name: torch.from_numpy(value.copy()).to(torch.bfloat16) for name, value in records.items()} for lane, records in own.items()}
    acts = a["read"](root / "act-records.json")
    _, cuts = a["act_accounts"](root, acts, own_t, columns=[qc["hidden_size"], 3 * qc["hidden_size"]])
    reports, rows = {}, {}
    for case in P["CASES"]:
        rows[case] = {lane: a["read"](root / "cases" / case / (lane + "-state-rows.json")) for lane in ("forecast", "native")}
        load = a["state_loader"](root / "cases" / case, old, streamer, dict(profile="tiny-cpu"), identity_cut=cuts["forecast"][case])
        trajectory = old["trajectory_accounts"](rows[case]["forecast"][1:], rows[case]["native"][1:], {lane: values[0] for lane, values in rows[case].items()}, load, joints=2, singles=2, steps=4)
        reports[case] = dict(denoiser=trajectory, decoder=dict(first_retained_same_latent_discrepancy=None))
    sealed = a["read"](root / "forecast-responses.json")
    response = dict(denoiser=[], decoder={})
    for index, row in enumerate(sealed["denoiser"]):
        states = {lane: {case: dict(a["tensor_leaves"](LocalStore(root / "cases" / case / (lane + "-state-store")).load(rows[case][lane][index]["cut"]).payload)) for case in P["CASES"]} for lane in ("forecast", "native")}
        fields = {name: a["response_account"]({lane: {case: old["field_to_numpy"](states[lane][case][name]) for case in P["CASES"]} for lane in states}, descriptor, old) for name, descriptor in row["fields"].items()}
        if "conditioning" in fields:
            fields["conditioning"]["declared_target_and_collateral_routes"] = {}
        response["denoiser"].append(dict(number=row["number"], clock=row["clock"], fields=fields))
    compact = tmp_path / "actual-tiny-compact"
    compact.mkdir()
    decoded = {case: a["read"](root / "cases" / case / "decoder-forecast.json") for case in P["CASES"]}
    for name in ("pixels", "decoded"):
        values = {case: old["load_array"](root / "cases" / case, decoded[case][name]) for case in P["CASES"]}
        # Pixel/decoded values are the producer's actual serialization fixture,
        # explicitly no native/functional VAE execution or VAE parity claim.
        response["decoder"][name] = a["response_account"]({lane: values for lane in ("forecast", "native")}, sealed["decoder"][name], old)
    for case in P["CASES"]:
        pixels = old["load_array"](root / "cases" / case, decoded[case]["pixels"])
        for lane in P["LANES"]:
            Image.fromarray(pixels).save(compact / (case + "-" + lane + ".png"))
    counts = dict(qwen_functional_blocks=fq["counts"]["blocks"], qwen_native_blocks=nq["counts"]["baseline_blocks"] + nq["counts"]["suffix_blocks"],
        native_qwen_calls=1, act_receipts=4, functional_transitions=56, native_transitions=56, receipt_occurrences=60, functional_decode_calls=0, native_vae_calls=0)
    audit = dict(qwen=qr, cases=reports, signed_responses=response, actual_counts=counts,
        context_consumers=dict(actual_occurrences=8), coverage=a["coverage_account"](qr, reports, response),
        qualification_fixture="ACTUAL TINY SAVED COMPONENTS · pixel serialization only, no VAE or production result")
    # The tiny writer's timeline is real; the figure needs placement intervals,
    # which only the whole production orchestrator records. Keep this fixture
    # clock separately labeled instead of inventing measured placement seconds.
    (compact / "source-timeline.jsonl").write_text("")
    output = tmp_path / "actual-tiny-figures"
    output.mkdir()
    plots = P["plot_data"](audit)
    qa = P["figures"](compact, output, audit, plots)
    put(output / "plot-data.json", plots)
    put(output / "scope.json", dict(scope=audit["qualification_fixture"], actual_qwen_blocks=[6, 7], actual_native_advances=56,
                                    no_placement_timing_measurement=True, new_model_calls=0))
    assert len(qa) == 5 and all(row["within_figure"] for rows in qa.values() for row in rows)
    assert all(rows[0]["figure_title"].startswith(audit["qualification_fixture"]) for rows in qa.values())
    assert len(plots["qwen_history"]["minus"]) == 3
    assert audit["coverage"]["primary_qwen_whole_fields"]["fields"] == 11
    assert audit["coverage"]["qwen_query_region_diagnostics"]["fields"] == 22
    assert P["summarize"](audit)["qwen_primary_full_field_pairs"] == 11
    assert "first_suffix" in plots and "oddness" in plots
