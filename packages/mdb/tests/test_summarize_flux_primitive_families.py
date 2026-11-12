"""Compact-only fixtures, opaque factor IDs, counterexamples and custody faults."""
from __future__ import annotations

import copy
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.computation_protocol import compile_protocol, evaluate_rule, freeze_rule

FILE = Path(__file__).parents[1] / "qualification/summarize_flux_primitive_families.py"
R = runpy.run_path(str(FILE))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(R["canonical"](value) + b"\n")


def record(name, shape):
    return dict(path=name + ".npy", shape=shape, sha256="a"*64, bytes=128,
        content_sha256="b"*64, logical_dtype="torch.bfloat16", stored_dtype="uint16")


def metric(size, **extra):
    return dict(elements=size, raw_bit_exact=True, changed=0, zero_sign_mismatches=0, rms=0., max_abs=0., **extra)


def plan_fixture():
    return compile_protocol(dict(schema="mdb-computation-family-spec-v2", family_id="fixture-opaque-factors",
        model_identity="fixture-model", endpoint_adapter="metadata-only-fixture", source_policy="supplied synthetic fixture",
        ports={name: dict(shape=[2], dtype="fixture", stage="supplied operand") for name in ("x", "y")},
        factors=[dict(id="left_part", ports=["x"]), dict(id="right_part", ports=["y"])],
        nodes=[dict(id="product", op="multiply", inputs=["x", "y"])], output="product",
        receiver=dict(R="P", conditioning="Q"), clock=dict(step=0, phase="fixture", index=1), max_contexts=2,
        continuation=dict(consumer="supplied fixture consumer", transitions=1, transformer_blocks=1, vae_decodes=1),
        metric=dict(observable="signed fixture axis", units="dimensionless", reference="natural P zero / Q one", scope="fixture only", direction="minimize")))


def context(name, seed):
    return dict(id=name, seed=seed, panel="fixture", prompts=dict(P="left "+name, Q="right "+name))


def packet(plan, row, values):
    return dict(plan_sha256=plan["plan_sha256"], model_identity=plan["specification"]["model_identity"],
        metric=plan["specification"]["metric"], contexts=[row], provenance=dict(fixture=True),
        rows=[dict(context=row["id"], action=action["id"], value=values[action["id"]]) for action in plan["actions"]])


def seal_collection(run, kind, publication, *, remote=None):
    compact = run / "compact"
    files = {p.relative_to(compact).as_posix(): dict(bytes=p.stat().st_size, sha256=R["sha"](p))
        for p in compact.rglob("*") if p.is_file()}
    custody = dict(required=True, declared=dict(sha256="c"*64), executed=dict(sha256="c"*64), sealed=dict(sha256="c"*64))
    write(run / "collection.json", dict(kind=kind, publication=publication, payload_custody=custody,
        inventory=dict(files=files, rawBytes=100000+sum(row["bytes"] for row in files.values()), rawFileCount=len(files)+7,
            root="/unopened/remote/"+kind, retained_remote_members=remote or {}), raw_retained_on_durable_volume=True))


def fixture(tmp_path, *, legacy_reader=False):
    plan, ctx = plan_fixture(), context("fresh-context", 29)
    actions, cid = plan["actions"], ctx["id"]
    native_run, audit_run = tmp_path / "native", tmp_path / "audit"
    nroot, aroot = native_run / "compact", audit_run / "compact"
    nroot.mkdir(parents=True)
    aroot.mkdir(parents=True)
    config = dict(compiled_protocol=plan, contexts=[ctx], arms=[a["id"] for a in actions], family="fixture-family")
    request = dict(config=config, source_hashes={}, runtime_wheels={})
    request["request_sha256"] = R["request_identity"](request)
    counts = dict(functional_source_blocks=2, native_source_blocks=2, functional_primitive_calls=4, native_primitive_calls=4,
        functional_hybrid_projections=4, native_hybrid_projections=4, functional_transitions=6, native_target_transitions=6,
        functional_vae=6, native_vae=6, forecast_acts=5, native_acts=5, native_blocks=8)
    summary = dict(protocol_sha256=plan["plan_sha256"], job_id="native-fixture-job", request_sha256=request["request_sha256"],
        counts=counts, forecast_seal="d"*64, mechanics_status="completed")
    writers = {cid: dict(sources={role: dict(ports={p: record(role+p, [2]) for p in ("a", "b")}, profile=dict(domain="fixture/no model"))
        for role in ("P", "Q")}, endpoints={a["id"]: record(a["id"], [2]) for a in actions},
        constructions={a["id"]: dict(ports={} if a["role"] == "natural-reference" else
            dict(primitive_return=record(a["id"]+"-primitive", [2]), projection_return=record(a["id"]+"-projection", [1]))) for a in actions})}
    keys = [cid+"/"+a["id"] for a in actions]
    index = {key: dict(root="cases/"+key) for key in keys}
    consumer = {key: dict(ports=dict(retained=record(key+"-consumer", [3]))) for key in keys}
    program = dict(commands=[dict(operation=command["operation"]) for command in plan["program"]])
    for name, value in {
        "plan.json": config, "request.json": request, "summary.json": summary,
        "native-contract.json": dict(counts=counts, observed={}), "state-index.json": index,
        "program.json": program, "runtime-contract.json": dict(profile="fixture/no model"),
        "source-provenance.json": dict(source_policy_sha256="e"*64),
        "forecast-source-writers.json": writers, "native-source-writers.json": writers,
        "consumer-ports.json": dict(forecast=consumer, native=consumer),
    }.items():
        write(nroot / name, value)
    rawfiles = {p.relative_to(nroot).as_posix(): dict(bytes=p.stat().st_size, sha256=R["sha"](p)) for p in nroot.iterdir()}
    write(nroot / "files.json", dict(files=rawfiles))
    manifest = R["sha"](nroot / "files.json")
    values = {a["id"]: .5 for a in actions}
    for action in actions:
        if action["role"] == "natural-reference":
            values[action["id"]] = 0. if action["source"] == "P" else 1.
    first, second = plan["candidate_menu"]
    values[first], values[second] = .8, .4
    dev_values = dict(values, **{first: .1, second: .2})
    rule = freeze_rule(plan, packet(plan, context("development-context", 11), dev_values), [ctx])
    measurements = packet(plan, ctx, values)
    effect = evaluate_rule(plan, rule, measurements)
    source_checks = {cid: {role: dict(comparisons={p: metric(2) for p in ("a", "b")}) for role in ("P", "Q")}}
    # A numeric-exact signed-zero miss must remain separate from numeric errors.
    source_checks[cid]["P"]["comparisons"]["a"].update(raw_bit_exact=False, zero_sign_mismatches=1)
    cases = {key: dict(consumer_ports=dict(retained=metric(3)),
        decoder=dict(latent=metric(2), decoded=metric(1), pixels=metric(3), decoder_input=metric(2)), png_byte_exact=True) for key in keys}
    eq = {key: dict(continuation=dict(fields={name: dict(shape=shape) for name, shape in
        (("latent", [2]), ("decoded", [1]), ("pixels", [1, 1, 3]))})) for key in keys}
    images = {}
    for key in keys:
        for lane in ("forecast", "native"):
            relative = "images/"+key.replace("/", "--")+"--"+lane+".png"
            path = aroot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"PNG placeholder; metadata fixture only")
            images[key+"/"+lane] = dict(path=relative, sha256=R["sha"](path), bytes=path.stat().st_size)
    axes = {cid: {field: dict(status="applicable", source=next(a["id"] for a in actions if a.get("source") == "P"),
        recipient=next(a["id"] for a in actions if a.get("source") == "Q"),
        rows={aid: dict(p=value, orthogonal_rms=.1, source_rms=.2, recipient_rms=.3) for aid, value in values.items()},
        mixed_remainder=dict(rms=.05)) for field in ("latent", "decoded", "pixels")}}
    audit = dict(native_job_id=summary["job_id"], native_request_sha256=request["request_sha256"], native_manifest_sha256=manifest,
        protocol_sha256=plan["plan_sha256"], native_counts=counts, contexts=[cid], writer_checks=source_checks,
        endpoint_checks={cid: {a["id"]: metric(2) for a in actions}}, cases=cases, equation_reports=eq,
        observations=axes, construction_checks={cid: {lane: dict(mixed_remainder=dict(rms=.02)) for lane in ("forecast", "native")}},
        images=images, transaction=dict(event_head="f"*64), boundary_counts=dict(paired_rows=12),
        mechanics_status="verified", terminal_status="unassessed", frozen_effect_evaluation=effect)
    if not legacy_reader:
        audit["construction_return_comparisons"] = {cid: {a["id"]: dict(primitive_return=metric(2), projection_return=metric(1))
            for a in actions if a["role"] != "natural-reference"}}
        audit["construction_return_comparisons"][cid][first]["primitive_return"].update(raw_bit_exact=False, changed=1,
            raw_bit_mismatch_elements=1, raw_bit_mismatch_bits=7, rms=.7)
        j = dict(path="primitive-interactions/remote-native-J.npy", sha256="1"*64, bytes=144, shape=[2])
        audit["native_primitive_interactions"] = {cid: dict(array=j, rms=.25)}
    write(aroot / "audit.json", audit)
    write(aroot / "measurements.json", measurements)
    seal_collection(native_run, "native", dict(job_id=summary["job_id"], request_sha256=request["request_sha256"], files_manifest_sha256=manifest))
    remote = {} if legacy_reader else {j["path"]: dict(sha256=j["sha256"], bytes=j["bytes"])}
    seal_collection(audit_run, "audit", dict(source_job=summary["job_id"], audit_sha256=R["sha"](aroot / "audit.json")), remote=remote)
    return native_run, audit_run, plan, rule


def test_all_pair_counts_signedzeros_nullspace_and_counterexamples_remain_distinct(tmp_path):
    native, audit, plan, rule = fixture(tmp_path)
    result = R["summarize"](native, audit, plan, rule=rule)
    assert result["comparison_totals"]["expected_array_pairs"] == 48
    assert result["comparison_totals"]["compared_array_pairs"] == 48
    assert result["comparison_totals"]["compared_elements"] == 98
    assert result["comparison_totals"]["raw_bit_miss_pairs"] == 2
    assert result["comparison_totals"]["zero_sign_mismatches"] == 1
    assert result["comparisons"]["source"]["miss_pairs_without_bit_count"] == 1
    assert result["comparisons"]["projection"]["raw_bit_miss_pairs"] == 0
    assert result["comparisons"]["primitive"]["known_raw_bit_mismatch_bits"] == 7
    assert result["statuses"]["held_counterexamples"] == 1 and result["statuses"]["audit_mechanics"] == "verified"
    assert result["png"]["compared_pairs"] == result["png"]["exact_pairs"] == 6
    assert result["mixed_interactions"]["fresh-context"]["primitive_remote_ledger_status"] == "hash-and-bytes-joined"
    assert result["geometry"]["joins"] and all(row["equal"] for row in result["geometry"]["joins"])
    assert result["geometry"]["missing_direct_counters"][0]["field"] == "declared_graph_node_invocations"
    assert len(result["per_context_action_rows"]) == 18


def test_old_reader_coverage_is_incomplete_and_legacy_geometry_is_not_fabricated(tmp_path):
    native, audit, plan, _ = fixture(tmp_path, legacy_reader=True)
    result = R["summarize"](native / "compact", audit / "compact", plan)
    assert result["comparisons"]["primitive"]["expected_array_pairs"] == 4
    assert result["comparisons"]["primitive"]["compared_array_pairs"] == 0
    assert result["comparison_totals"]["compared_array_pairs"] == 40
    assert result["statuses"]["evidence"] == "incomplete-comparison-coverage"
    assert result["mixed_interactions"]["fresh-context"]["primitive"] is None
    no_geometry = copy.deepcopy(plan)
    del no_geometry["execution_geometry"]
    function = R["geometry_account"]
    raw_summary = R["read"](native / "compact/summary.json")
    joined = function(no_geometry, {"fresh-context": {}}, raw_summary, R["read"](native / "compact/native-contract.json"),
        R["read"](audit / "compact/audit.json"), R["read"](native / "compact/state-index.json"), R["read"](native / "compact/program.json"))
    assert joined["generated"] is None and joined["status"] == "legacy-plan-without-generated-geometry"
    assert joined["joins"] == [] and joined["actual_counts"]["native_blocks"] == 8


def test_collection_tamper_and_remote_J_metadata_corruption_are_refused(tmp_path):
    native, audit, plan, _ = fixture(tmp_path)
    path = audit / "compact/audit.json"
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="collection bytes"):
        R["summarize"](native, audit, plan)
    path.write_bytes(path.read_bytes()[:-1])
    collection = R["read"](audit / "collection.json")
    next(iter(collection["inventory"]["retained_remote_members"].values()))["sha256"] = "2"*64
    write(audit / "collection.json", collection)
    with pytest.raises(ValueError, match="J ledger"):
        R["summarize"](native, audit, plan)


def test_reporter_never_imports_frameworks_or_opens_remote_tensors(tmp_path):
    native, audit, plan, _ = fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    script = r'''
import pathlib,runpy,sys
r=runpy.run_path(sys.argv[1]); original=pathlib.Path.open
def guarded(path,*args,**kwargs):
    assert path.suffix not in {'.npy','.npyb','.safetensors','.whl'}
    return original(path,*args,**kwargs)
pathlib.Path.open=guarded
r['summarize'](sys.argv[2],sys.argv[3],sys.argv[4])
assert not {'torch','numpy','diffusers','transformers','saturn_pub'} & sys.modules.keys()
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(FILE), str(native), str(audit), str(plan_path)], check=True)
