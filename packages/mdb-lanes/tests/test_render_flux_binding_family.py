"""Metadata/image-only family normalization against an installed public viewer."""
from __future__ import annotations

import base64
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

FILE = Path(__file__).parents[1] / "qualification/render_flux_binding_family.py"
R = runpy.run_path(str(FILE))
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII=")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def record(path):
    return dict(bytes=path.stat().st_size, sha256=R["sha"](path))


def inventory(root):
    return {path.relative_to(root).as_posix(): record(path) for path in root.rglob("*") if path.is_file()}


def collection(run, kind, publication, *, request_sha="d" * 64, job="job-audit-fixture"):
    files = inventory(run / "compact")
    write(run / "collection.json", dict(schema=R["COLLECTION"], kind=kind, job_id=job, request_sha256=request_sha,
        publication=publication, inventory=dict(files=files, root="/remote/" + job),
        payload_custody=dict(required=True, **{key: dict(sha256="f" * 64, size_bytes=123) for key in ("declared", "sealed", "executed")})))


def fixture(tmp_path):
    native_run, audit_run = tmp_path / "native", tmp_path / "audit"
    native, audited = native_run / "compact", audit_run / "compact"
    native.mkdir(parents=True)
    audited.mkdir(parents=True)
    model, job = "fixture-exact-model", "job-native-fixture"
    request = dict(config=dict(mode="historical", arms=["QQQ", "PQP"]), source_hashes={"flux_binding_family_campaign.py": "a" * 64})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    request_sha = request["request_sha256"]
    tx = ForecastTransaction(native / "forecast-transaction", model_identity=model, input_identity="fixture-input",
        source_identity="fixture-original-source", execution_scope="saved JSON fixture, no model",
        frontier=dict(kind="architectural"))
    tx.frontier("source-inputs-and-rule", dict(source="saved fixture"))
    tx.predict("all-numeric-futures", lambda _ctx: dict(predictions=dict(cases=["toy/QQQ", "toy/PQP"])))
    seal = tx.seal()
    tx.execute("native-source-and-futures", lambda _ctx: dict(observations="fixture only"))
    tx.close()
    transaction = verify_transaction(native / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"])
    cut = dict(cut="b" * 64, model_identity=model, execution_point=dict(logical_step=2, phase="joint", index=0, edge="before"),
        execution=dict(dtype="torch.bfloat16", device=dict(type="cuda", index=0)))
    roots = dict(toy=dict(inputs=dict(P=cut, Q=cut), support=[4, 11, 12, 13], base_conditioning=dict(path="remote-only.npy"), known={}))
    states, images, cases = {}, {}, {}
    measured = dict(rms=.25, reference_rms=2., candidate_rms=2.1, cosine=.9, raw_bit_exact=False, changed=4)
    for arm in request["config"]["arms"]:
        key = "toy/" + arm
        states[key] = dict(root="cases/" + key, context="toy", arm=arm, forecast_final=cut, native_final=cut)
        for lane, folder in (("forecast", "forecast"), ("native", "native-reference")):
            path = native / "cases" / key / "decoder" / folder / "image.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(PNG)
            target = audited / "images" / (key.replace("/", "--") + "--" + lane + ".png")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(PNG)
            images[key + "/" + lane] = dict(path=target.relative_to(audited).as_posix(), **record(target))
        cases[key] = dict(decoder=dict(decoded=measured, pixels=measured), consumer_ports=dict(x=measured), entry_endpoint=measured)
    summary = dict(schema="mdb-flux-binding-family-v1", job_id=job, request_sha256=request_sha,
        contexts=["toy"], arms=request["config"]["arms"], forecast_seal=seal["seal_sha256"], transaction=transaction,
        counts=dict(functional_transitions=108, native_target_transitions=108, functional_vae=2, native_vae=2))
    writer = dict(state=dict(cut, execution_point=dict(logical_step=2, phase="joint", index=1, edge="before")), ports={})
    for name, value in (("request.json", request), ("plan.json", request["config"]), ("summary.json", summary),
                        ("source-roots.json", roots), ("state-index.json", states), ("runtime-contract.json", dict(profile="saved fixture")),
                        ("native-source-writers.json", dict(toy=dict(sources=dict(P=writer, Q=writer))))):
        write(native / name, value)
    raw_manifest = dict(schema="mdb-image-tail-files-v1", job_id=job, request_sha256=request_sha, files=inventory(native))
    write(native / "files.json", raw_manifest)
    publication = dict(job_id=job, request_sha256=request_sha, files_manifest_sha256=R["sha"](native / "files.json"),
        summary_sha256=R["sha"](native / "summary.json"))
    collection(native_run, "native", publication, request_sha=request_sha, job=job)
    observed = dict(toy=dict(pixels=dict(responses=dict(PQP=dict(projection=-.125)),
        frozen_route_predictions=dict(support_closer_to_whole_than_Q=False), triple_interaction=dict(rms=.75))))
    audit = dict(schema=R["AUDIT"], native_job_id=job, native_request_sha256=request_sha,
        native_manifest_sha256=publication["files_manifest_sha256"], source_worker_sha256="a" * 64, contexts=["toy"],
        arms=request["config"]["arms"], cases=cases, images=images, transaction=transaction, construction_checks=dict(toy={arm: measured for arm in request["config"]["arms"]}),
        observations=observed, external_sealed_files=4, mechanics_status="verified", instrument_status="fixture saved reader",
        terminal_status="unassessed", limitations=["Synthetic metadata fixture; not a real scientific result"],
        numerical_bit_misses=[dict(case="toy/PQP", field="pixels")], writer_checks=dict(toy=dict(diagnostic="retained")))
    write(audited / "audit.json", audit)
    collection(audit_run, "audit", dict(source_job=job, audit_sha256=R["sha"](audited / "audit.json")))
    return native_run, audit_run


def reseal_audit(run):
    value = R["read"](run / "collection.json")
    publication = dict(value["publication"], audit_sha256=R["sha"](run / "compact/audit.json"))
    collection(run, "audit", publication)


def test_rendered_labels_scopes_misses_and_linked_occurrences_are_explicit(tmp_path):
    native, audit = fixture(tmp_path)
    source_hashes = {path: R["sha"](path) for root in (native, audit) for path in root.rglob("*") if path.is_file()}
    request = R["read"](native / "compact/request.json")
    records = tmp_path / "records.jsonl"
    rows = [dict(schema_version="known-computations.v1", id="exact", model_identity="fixture-exact-model",
        request_sha256=request["request_sha256"], results=[dict(endpoint_rows=[dict(case="toy/PQP")])]),
        dict(schema_version="known-computations.v1", id="foreign", model_identity="another-model",
        request_sha256=request["request_sha256"], results=[dict(endpoint_rows=[dict(case="toy/QQQ")])])]
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    value = R["bundle"](native, audit, records=[records])
    assert [row["label"] for row in value["methods"]] == ["Native reference", "Original-weight equations"]
    assert value["comparisons"]["endpoint"][0]["rms"] == .25
    assert value["comparisons"]["endpoint"][0]["relativeRms"] == .125
    assert value["comparisons"]["response"] == value["comparisons"]["interaction"] == []
    assert isinstance(value["evidence"]["producer_native_witnesses"], list)
    observed = value["evidence"]["source_audit"]["observations"]["toy"]["pixels"]
    assert observed["responses"]["PQP"]["projection"] < 0 and observed["frozen_route_predictions"]["support_closer_to_whole_than_Q"] is False
    assert value["records"][0]["linked_cases"] == ["toy/PQP"] and value["records"][1]["linked_cases"] == []
    receipt = R["render"](native, audit, tmp_path / "view", records=[records])
    html = (tmp_path / "view/index.html").read_text()
    assert "Original-weight equations" in html and "Native reference" in html
    assert "frozen predictor adds a fitted correction" not in html and 'arm=data.cases[0]?.arm' in html
    assert "Previous-velocity cut" not in html and "Historical measured reference" not in html
    assert "Observed state and the velocity actually consumed" not in html
    assert "short(caseRow.inputs.source_inputs.P.cut)" in html and "Paired joint0 source inputs" in html
    assert receipt["model_executions"] == receipt["tensor_array_reads"] == 0
    assert all(R["sha"](path) == digest for path, digest in source_hashes.items())


@pytest.mark.parametrize("mutation", ("audit-bytes", "image-bytes", "foreign-job", "mixed-model", "case-census"))
def test_custody_source_model_and_census_drift_refuse(tmp_path, mutation):
    native, audit = fixture(tmp_path)
    if mutation == "image-bytes":
        next((audit / "compact/images").glob("*.png")).write_bytes(PNG + b"changed")
    elif mutation == "audit-bytes":
        with (audit / "compact/audit.json").open("a") as stream:
            stream.write(" ")
    elif mutation in ("foreign-job", "case-census"):
        value = R["read"](audit / "compact/audit.json")
        if mutation == "foreign-job":
            value["native_job_id"] = "job-another"
        else:
            value["cases"].pop("toy/PQP")
        write(audit / "compact/audit.json", value)
        reseal_audit(audit)
    else:
        value = R["read"](native / "compact/source-roots.json")
        value["toy"]["inputs"]["P"] = dict(value["toy"]["inputs"]["P"], model_identity="foreign")
        write(native / "compact/source-roots.json", value)
    with pytest.raises(ValueError, match="hash differs|source job|census differs|identity differs"):
        R["bundle"](native, audit)


def test_missing_local_image_is_unavailable_without_searching_or_inventing_an_outcome(tmp_path):
    native, audit = fixture(tmp_path)
    value = R["read"](audit / "compact/audit.json")
    target = audit / "compact" / value["images"]["toy/PQP/forecast"]["path"]
    target.unlink()
    reseal_audit(audit)
    bundle = R["bundle"](native, audit)
    future = next(row for row in bundle["futures"] if row["id"] == "toy/PQP/equations")
    assert future["image"]["status"] == "unavailable" and future["metrics"]["pngRGB"]["raw_bit_exact"] is False


def test_normalization_and_rendering_use_no_model_frameworks_or_array_loads(tmp_path):
    native, audit = fixture(tmp_path)
    code = "import runpy,sys; r=runpy.run_path(sys.argv[1]); r['render'](sys.argv[2],sys.argv[3],sys.argv[4]); assert not {'torch','numpy','diffusers','transformers'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(FILE), str(native), str(audit), str(tmp_path / "framework-free-view")], check=True)
