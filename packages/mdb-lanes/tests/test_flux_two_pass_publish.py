"""Public-only curation of six dependent two-pass output comparisons."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from mdb_lanes.computation_atlas import ComputationAtlas
from mdb_lanes.equation_comparison import EquationCatalog

P = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/publish_flux_two_pass.py"))


def fixture():
    binding = {"model_identity": {"checkpoint": "1" * 64},
        "execution": {"backend": "synthetic-cpu"}, "input_identity": "synthetic-input",
        "frontier": {"cut": "2" * 64, "step": 2}, "horizon": "synthetic whole-suffix recurrence",
        "source_identity": {"fixture.py": "3" * 64}}
    context = {key: binding[key] for key in ("model_identity", "execution", "source_identity")}
    context.update(input_cut="2" * 64, seed=109, prompt="tiny synthetic fixture")
    provenance = {"run_id": "synthetic-no-model", "artifacts": [{"path": "fixture.json", "sha256": "4" * 64}]}
    fields = {}
    for name in P["FIELD_NAMES"]:
        predicted = np.array([[[1., -2.], [3., -4.]]], np.float32)
        observed = predicted.copy()
        if name.startswith("final_noise"):
            observed[0, 0, 0] += .25
        fields[name] = {"predicted": predicted, "observed": observed,
            "provenance": {"fixture": True, "array_sha256": "5" * 64,
                "native_cut": hashlib.sha256(name.encode()).hexdigest()}}
    endpoints = {kind: {"candidate_branch":"full.predicted","reference_branch":reference,
        "latent":{"signed_mean":-.125},"rgb":{"signed_mean":.5,"mae":1.25}}
        for kind,reference in P["REFERENCES"].items()}
    return dict(fields=fields, binding=binding, tested_context=context, provenance=provenance, endpoints=endpoints)


def test_public_catalog_retains_both_references_and_misses_without_decisions(tmp_path):
    output = tmp_path / "publication"
    result = P["curate"](output, **fixture())
    assert result["curated_records"] == result["distinct_named_field_comparisons"] == 6
    assert result["distinct_sealed_output_fields"] == 4
    assert result["native_histories"] == 1 and result["family_candidates_created"] == 0
    catalog, atlas = EquationCatalog(output / "equations"), ComputationAtlas(output / "atlas")
    equations = [catalog.show(item["equation_id"]) for item in result["records"]]
    assert len(catalog.find(law_id="flux.denoising.own-state-suffix")) == 6
    assert {value["evidence"]["reference_kind"] for value in equations} == {"native", "matched_native_suffix"}
    assert all(value["report"]["decision_fields"] == 0 for value in equations)
    assert all("dependent" in value["evidence"]["independence"] for value in equations)
    assert sum(bool(atlas.show(identifier)["counterexamples"]) for identifier in result["atlas_observation_ids"]) == 2
    for value in equations:
        field = value["report"]["fields"][value["evidence"]["output_role"]+"."+value["evidence"]["reference_kind"]]
        if value["evidence"]["output_role"] == "final_noise":
            assert field["residual"]["signed_mean"] == .0625 and field["residual"]["max_abs"] == .25


def test_exact_query_does_not_expand_seed_or_missing_cut(tmp_path):
    result = P["curate"](tmp_path / "publication", **fixture())
    assert result["queries"]["exact_context_matches"] == 6
    assert result["queries"]["changed_seed_matches"] == result["queries"]["missing_input_cut_matches"] == 0
    assert result["queries"]["missing_input_cut_applicability"] == "unknown"
    assert len(result["query_contexts"]) == 6
    assert all(row["exact_context_matches"] == 1 for row in result["queries"]["occurrences"])
    assert {item["context"]["horizon"] for item in result["query_contexts"]} == {28,56}
    assert all(item["context"]["total_program_horizon"] == 56 for item in result["query_contexts"])
    for item in result["query_contexts"]:
        assert {predicate["field"] for predicate in item["conditions"]} >= {
            "output_role","reference_kind","reference_cut","input_cut","source_identity"}


def test_futures_distinguish_matched_bridge_and_native_reference(tmp_path):
    result = P["curate"](tmp_path / "publication", **fixture())
    atlas = ComputationAtlas(tmp_path / "publication/atlas")
    values = [atlas.show(identifier) for identifier in result["atlas_observation_ids"]]
    assert {value["continuation"]["reference_branch"] for value in values} == {"native", "bridge.predicted"}
    assert all([item["value"] for item in value["signed_effects"]] == [-.125, .5] for value in values)
    assert all("no isolated effect" in value["signed_effects"][0]["scope"] for value in values)
    assert all(value["rule_status"] == "observation; no validated rule" for value in values)
    for value in values:
        future = value["continuation"]
        assert future["branch"] == "full.predicted"
        assert future["status"] == "observed-native-VAE-only" and future["denoiser_calls_after_installation"] == 0
        assert "No independently predicted RGB or isolated field intervention" in future["scope"]
    assert sorted(value["continuation"]["remaining_mathematical_stages_after_named_field"] for value in values) == [0,0,1,1,28,29]


@pytest.mark.parametrize("kind",["native","matched_native_suffix"])
def test_wrong_endpoint_arm_cannot_be_attached_to_composition(tmp_path,kind):
    args = fixture()
    args["endpoints"][kind]["candidate_branch"] = "bridge.predicted"
    with pytest.raises(ValueError,match="whole-program VAE future mapping differs"):
        P["curate"](tmp_path/"publication",**args)


def test_inventory_freshness_and_shape_are_integrity_requirements(tmp_path):
    args = fixture()
    P["curate"](tmp_path / "publication", **args)
    with pytest.raises(FileExistsError):
        P["curate"](tmp_path / "publication", **args)
    del args["fields"]["final_noise.native"]
    with pytest.raises(ValueError, match="inventory differs"):
        P["curate"](tmp_path / "incomplete", **args)
    assert not (tmp_path / "incomplete").exists()
    args = fixture()
    args["fields"]["final_noise.native"]["observed"] = np.ones((1, 1, 4), np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        P["curate"](tmp_path / "shape", **args)


def test_failed_job_refuses_before_output_creation(tmp_path):
    request = {"config": {}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    audit = {"request_sha256": request["request_sha256"], "job_id": "job-synthetic",
        "mechanics_status": "valid", "models_executed": False}
    for name, value in (("request.json", request), ("job.json", {"job_id": "job-synthetic", "state": "failed"}),
        ("collection.json", {"request_sha256": request["request_sha256"], "job_id": "job-synthetic"}), ("audit.json", audit)):
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="accepted audit/job/request differs"):
        P["publish"](tmp_path, tmp_path / "audit.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_preoutcome_auditor_must_match_actual_and_frozen_sources(tmp_path):
    path = tmp_path / "independent-math-freeze/analyze_flux_two_pass.py"
    path.parent.mkdir()
    path.write_text("preoutcome independent equations")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    request = {"independent_math_sources": {"analyze_flux_two_pass.py": digest}}
    audit = {"auditor_source_sha256": digest, "independent_math_sources": request["independent_math_sources"]}
    result = P["audit_authority"](tmp_path, audit, request, current_auditor=path)
    assert result["status"] == "pre-outcome-auditor" and result["accepted_auditor_sha256"] == digest
    other = tmp_path / "other.py"
    other.write_text("unaccepted alternative equations")
    with pytest.raises(ValueError, match="frozen mathematical source"):
        P["audit_authority"](tmp_path, audit, request, current_auditor=other)


def test_signed_endpoint_diagnostic_has_no_quality_gate():
    actual = np.array([2., 0.], np.float32)
    baseline = np.array([1., 1.], np.float32)
    result = P["difference"](actual, baseline)
    assert result == {"signed_mean": 0., "mae": 1., "rms": 1., "l2": np.sqrt(2.), "max_abs": 1.,
        "changed_value_count": 2, "sample_count": 2}
    assert P["MAX_ELEMENTS"] >= 1024 * 3072


@pytest.mark.parametrize("name", ["../absolute", "/absolute", "a/../b"])
def test_member_paths_are_canonical_and_contained(tmp_path, name):
    with pytest.raises(ValueError, match="evidence path"):
        P["member"](tmp_path, name)

def saved_run(tmp_path,monkeypatch):
    """Small source/transaction/inventory fixture; all native claims are synthetic."""
    import torch
    from PIL import Image

    from mdb_lanes.forecast_transaction import ForecastTransaction
    from mdb_lanes.tensor_archive import write_tensor_archive

    root = tmp_path/"run"
    data = root/"extracted/flux-two-pass-result"
    data.mkdir(parents=True)
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    current = root/"current/analyze_flux_two_pass.py"
    current.parent.mkdir()
    current.write_text("synthetic pre-outcome reader fixture")
    frozen = root/"independent-math-freeze/analyze_flux_two_pass.py"
    frozen.parent.mkdir()
    frozen.write_bytes(current.read_bytes())
    payload = root/"payload/worker.py"
    payload.parent.mkdir()
    payload.write_text("synthetic producer fixture")
    monkeypatch.setitem(P["audit_authority"].__globals__,"HERE",current.parent)
    source_hashes = {"worker.py":P["sha"](payload)}
    independent = {"analyze_flux_two_pass.py":P["sha"](frozen)}
    plan = {"context":{"seed":109,"prompt":"synthetic fixture"}}
    request = {"config":plan,"source_hashes":source_hashes,"independent_math_sources":independent}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    job = {"state":"succeeded","job_id":"job-synthetic-two-pass","result":{"returncode":0}}
    source = {"context":plan["context"],"source_hashes":source_hashes,"request_sha256":request["request_sha256"],
        "job_id":job["job_id"],"model_identity":{"checkpoint":"synthetic"},"execution":{"backend":"synthetic"},
        "implementation_sources":{}}
    binding = {"input_cut":"a"*64,"point":{"logical_step":2,"phase":"project"}}
    base = np.array([[[1.,2.],[3.,4.]]],np.float32)
    passes = [{"readout":{"noise":base+i},"commit":{"latent":base+2+i}} for i in range(2)]
    tx = ForecastTransaction(data/"transaction",model_identity=source["model_identity"],input_identity="synthetic",
        source_identity="fixture",frontier={"kind":"architectural"},execution_scope="Synthetic publisher test only",
        budgets={"max_native_calls":1,"max_wall_ms":5000,"max_bytes":1<<20})
    with tx:
        tx.predict("two_pass",lambda _:{"predictions":{"passes":passes,"chain_length":56}})
        seal = tx.seal()
        tx.execute("synthetic",lambda _:{"scope":"No model calls"})
    def archive(name,value):
        path = data/(name+".npy")
        return {**write_tensor_archive(path,torch.tensor(value).to(torch.bfloat16)),"path":path.name}
    records = {key:archive(key,value) for key,value in {"noise0":base+.25,"latent0":base+2.25,
        "noise1":base+1.25,"native":base+3.25,"bridge.predicted":base+3.5,"full.predicted":base+3}.items()}
    def boundary(name,logical,phase,latent="native",noise=None):
        return {"cut":hashlib.sha256(name.encode()).hexdigest(),"point":{"logical_step":logical,"phase":phase},
            "clock":{"step":logical,"phase":phase,"cursor":0},
            "closure":{"latent":records[latent],"context":None,"image":None,"text":None,
                "noise_prediction":None if noise is None else records[noise]}}
    native = [boundary(str(i),2,"joint") for i in range(56)]
    native[26] = boundary("noise0",2,"scheduler",noise="noise0")
    native[27] = boundary("latent0",3,"project",latent="latent0")
    native[54] = boundary("noise1",3,"scheduler",noise="noise1")
    native[55] = boundary("native-terminal",4,"project")
    bridge = [boundary("bridge"+str(i),3,"single") for i in range(22)]
    bridge[20] = boundary("bridge-noise",3,"scheduler",noise="noise1")
    bridge[21] = boundary("bridge-terminal",4,"project",latent="bridge.predicted")
    rows = {}
    for index,name in enumerate(("native","sham","bridge.predicted","full.predicted")):
        terminal = native[55] if name in ("native","sham") else bridge[21] if name == "bridge.predicted" else boundary(name,4,"project",latent=name)
        Image.fromarray(np.full((2,2,3),index,np.uint8)).save(data/(name+".png"))
        rows[name] = {"id":name,"final_cut":terminal["cut"],"terminal_final":terminal,"continuation_receipts":[],
            "bridge_receipts":[{}]*22 if name == "bridge.predicted" else [],
            "bridge_boundaries":bridge if name == "bridge.predicted" else [],
            "bridge_records":[{"after":value} for value in bridge] if name == "bridge.predicted" else [],
            "installed":{"cut":"b"*64},"final_latent":terminal["closure"]["latent"],
            "image":{"path":name+".png","sha256":P["sha"](data/(name+".png"))}}
        save(data/"rows"/(name+".json"),rows[name])
    derived = {"native_steps":0,"acts_applied":0,"execution_receipt_emissions":0,"source_cut":binding["input_cut"]}
    save(data/"derived-terminal.json",derived)
    first = {"image":rows["full.predicted"]["image"],"counts":{"adapter_steps":56,"acts":1,"vae_calls":1,"images":1},
        "target_entries":{"single":0},"source_cut":binding["input_cut"],
        "derivation_sha256":P["sha"](data/"derived-terminal.json")}
    summary = {"request_sha256":request["request_sha256"],"job_id":job["job_id"],"counts":P["COUNTS"],"seal":seal}
    for name,value in (("source.json",source),("summary.json",summary),("plan.json",plan),("binding.json",binding),
            ("native-segment.json",{"boundaries":native,"receipts":[{}]*56,"native_final":native[55]}),
            ("first-predicted-image.json",first)):
        save(data/name,value)
    files = {p.relative_to(data).as_posix():{"sha256":P["sha"](p),"bytes":p.stat().st_size,"filetype":"regular"}
        for p in data.rglob("*") if p.is_file()}
    save(data/"remote-manifest.json",{"files":files,"job_id":job["job_id"],"request_sha256":request["request_sha256"]})
    files["remote-manifest.json"] = {"sha256":P["sha"](data/"remote-manifest.json"),
        "bytes":(data/"remote-manifest.json").stat().st_size,"filetype":"regular"}
    inventory = {"files":files}
    for name in ("collection-local-inventory.json","collection-remote-before.json","collection-remote-after.json"):
        save(root/name,inventory)
    collection = {"request_sha256":request["request_sha256"],"job_id":job["job_id"],"bindings":{},
        "manifest_sha256":P["sha"](data/"remote-manifest.json"),"summary_sha256":P["sha"](data/"summary.json"),
        "seal_sha256":seal["seal_sha256"]}
    for name,value in (("request.json",request),("job.json",job),("collection.json",collection)):
        save(root/name,value)
    audit = {"request_sha256":request["request_sha256"],"job_id":job["job_id"],"mechanics_status":"valid",
        "models_executed":False,"auditor_source_sha256":P["sha"](current),"independent_math_sources":independent,
        "collection":{"sha256":P["sha"](root/"collection.json")},"counts":P["COUNTS"],"seal":seal,
        "first_predicted_image":first,"derived_terminal":{"all13_fields_prefix_derived":True}}
    save(root/"audit.json",audit)
    return root,data


def test_saved_custody_selected_fields_and_terminal_futures(tmp_path,monkeypatch):
    root,data = saved_run(tmp_path,monkeypatch)
    result = P["publish"](root,root/"audit.json",tmp_path/"publication")
    assert result["curated_records"] == 6 and result["distinct_sealed_output_fields"] == 4
    assert result["queries"]["exact_context_matches"] == 6
    for record in result["records"]:
        field = record["report"]["fields"][record["output_role"]+"."+record["reference_kind"]]
        assert field["residual"]["signed_mean"] > 0
        assert record["candidate_branch"] == "full.predicted"
    catalog = EquationCatalog(tmp_path/"publication/equations")
    assert catalog.show(result["records"][0]["equation_id"])["binding"]["horizon"] == 28
    with (data/"full.predicted.png").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError,match="metadata changed"):
        P["publish"](root,root/"audit.json",tmp_path/"tampered-publication")
    assert not (tmp_path/"tampered-publication").exists()
