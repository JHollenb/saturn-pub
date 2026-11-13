"""Saved-only full-cut audit of native single-stack response localization.

Recomputes observed response curves from complete I/T fields. It constructs no
model, forecast, Act, native continuation, or equation-forecast record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLES = ("sham", "HH", "HL")
WORKER_SHA256 = "ef799128d1270bce1fb447f20e08b174bb4c2d70d9bbc7650f9c8cdb0dab85fc"
EXECUTION_ID = "mdb-flux-single-stack-component-v1"
PARTITIONS = {"early": (0, 6), "middle": (7, 13), "late": (14, 19)}
HELPERS = ("audit_flux_image_tail_microscope.py", "flux_image_tail_microscope.py", "flux_source_receiver.py",
           "retrieve_reader_evidence.py", "flux_family_custody.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def metrics(changed, sham):
    """Independent changed-minus-sham full-field reductions, native units."""
    import numpy as np
    a, b = np.asarray(changed, np.float64), np.asarray(sham, np.float64)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), "full field shape/nonfinite")
    d = a - b
    energy, baseline, amplitude = float(np.mean(d * d)), float(np.sqrt(np.mean(b * b))), float(np.sqrt(np.mean(a * a)))
    rms = energy ** .5
    return dict(orientation="changed-minus-sham", rms=rms, sham_rms=baseline, changed_rms=amplitude,
                relative_to_sham_rms=None if baseline == 0 else rms / baseline, signed_mean=float(np.mean(d)),
                maxabs=float(np.max(np.abs(d))), changed_values=int(np.count_nonzero(d)), sample_count=int(d.size))


def responses(fields, previous):
    rows = {role: {stream: metrics(fields[role][stream], fields["sham"][stream]) for stream in ("image", "text")}
            for role in ("HH", "HL")}
    for role in rows:
        for stream, row in rows[role].items():
            before = None if previous is None else previous[role][stream]
            for new, old in (("rms_absolute_increment", "rms"), ("signed_mean_increment", "signed_mean"), ("maxabs_increment", "maxabs")):
                row[new] = None if before is None else row[old] - before[old]
    for stream in ("image", "text"):
        high, low = rows["HH"][stream]["rms"], rows["HL"][stream]["rms"]
        rows["HH"][stream]["HH_over_HL_response_rms"] = None if low == 0 else high / low
        rows["HL"][stream]["HL_over_HH_response_rms"] = None if high == 0 else low / high
    return rows


def summarize(curve, layers):
    summary, ranks, selective = {}, {}, {}
    for role in ("HH", "HL"):
        summary[role], ranks[role] = {}, {}
        for stream in ("image", "text"):
            summary[role][stream] = {}
            values = [dict(after_block=row["after_block"], **row["responses"][role][stream]) for row in curve[1:]]
            ranks[role][stream] = sorted(values, key=lambda row: abs(row["rms_absolute_increment"]), reverse=True)[:5]
            for name, (first, last) in PARTITIONS.items():
                last = min(last, layers - 1)
                if first > last:
                    summary[role][stream][name] = dict(coverage="not-present-in-tiny-fixture", blocks=[])
                    continue
                a, b = curve[first]["responses"][role][stream], curve[last + 1]["responses"][role][stream]
                summary[role][stream][name] = dict(blocks=list(range(first, last + 1)), start_boundary=first, end_boundary=last + 1,
                    start_response_rms=a["rms"], end_response_rms=b["rms"], signed_absolute_rms_change=b["rms"] - a["rms"],
                    start_sham_rms=a["sham_rms"], end_sham_rms=b["sham_rms"], start_changed_rms=a["changed_rms"], end_changed_rms=b["changed_rms"],
                    signed_mean_change=b["signed_mean"] - a["signed_mean"])
    for stream in ("image", "text"):
        selective[stream] = sorted([dict(after_block=row["after_block"],
            HH_increment=row["responses"]["HH"][stream]["rms_absolute_increment"],
            HL_increment=row["responses"]["HL"][stream]["rms_absolute_increment"],
            HH_minus_HL_increment=row["responses"]["HH"][stream]["rms_absolute_increment"] - row["responses"]["HL"][stream]["rms_absolute_increment"])
            for row in curve[1:]], key=lambda row: row["HH_minus_HL_increment"], reverse=True)
    return dict(partitions=summary, top5_by_absolute_size_of_signed_rms_increment=ranks, HH_selective_signed_increment_ordering=selective,
                scope="Exploratory ordering of changes in absolute response RMS, with native amplitudes retained; not causal attention/MLP contributions or independent shares.")


def cut_members(root, store, identifier, inventory):
    """Require all authoritative v2 cut members to be explicitly inventoried."""
    relative = Path(store) / "cuts" / (identifier + ".json")
    require(relative.as_posix() in inventory, "typed cut not inventoried")
    data = read(root / relative)
    require(data["schema"] == "saturn-pub-local-cut-v2", "authoritative v2 cut required")
    names = [relative.as_posix(), *[(Path(store) / "pages" / (value + ".json")).as_posix() for value in data["slots"].values()],
             *[(Path(store) / "blobs" / (value + ".safetensors")).as_posix() for value in data["physical_blobs"]]]
    require(set(names) <= set(inventory), "typed cut pages/blobs not inventoried")
    return names


def audit(root, output, *, expected_manifest, expected_request):
    from mdb_lanes.forward_recording import tensor_descriptor

    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    root, output = Path(root), Path(output)
    inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == set(inventory["files"]) | {"files.json"}, "raw inventory is incomplete or contains extra files")
    m = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))
    config, summary, source = [read(root / name) for name in ("plan.json", "summary.json", "source.json")]
    require(all(row["request_sha256"] == expected_request for row in (config, summary, source, inventory))
            and len({summary["job_id"], source["job_id"], inventory["job_id"]}) == 1, "trace request/job differs")
    require(source["worker_sha256"] == WORKER_SHA256 and source["helper_source_sha256"] == config["helper_source_sha256"], "producer source differs")
    require(source["source_provenance"] == summary["source_provenance"] == config["source_provenance"], "original recovered custody differs")
    tiny = config.get("profile") == "tiny-cpu"
    layers = config.get("block_count", 3) if tiny else 20
    expected_counts = dict(native_blocks=3 * layers, native_attention=3 * layers, linear=6 * layers, layernorm=3 * layers, rmsnorm=6 * layers, swiglu=3 * layers)
    require(summary["counts"] == expected_counts and summary["boundaries_per_role"] == layers + 1
            and summary["complete_typed_cuts"] == 3 * (layers + 1), "native census/coverage differs")
    require(all(summary[name] == 0 for name in ("act_receipts", "session_continue_receipts", "vae_calls", "encodes", "noise_draws", "suffix_steps")), "trace declared forbidden work")
    native_contract = read(root / "native-contract.json")
    require(native_contract["module"] == "Flux2SingleTransformerBlock" and native_contract["module_config"] == config["module_config"]
            and native_contract["module_source_sha256"] == config["native_source_sha256"]
            and native_contract["count_contract"]["counts"] == expected_counts, "native contract differs")
    if not tiny:
        p = config["source_provenance"]
        require(p["job_id"] == "job-46a52ffae499" and p["scheduler_status"] == "failed"
                and p["collection_mode"] == "authenticated-ssh-manifest-recovery" and p["manifest_origin"] == "collector-derived-source-manifest", "original failed exporter status lost")
    for key, name in (("binding", "binding.json"), ("parameter_manifest", "parameter-artifacts.json"),
                      ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
        require(sha(root / name) == config[key + "_sha256"], "copied source metadata differs")
    binding = read(root / "binding.json")
    source_root = Path(config["input_root"])
    for name, record in config["input_files"].items():
        path = source_root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.resolve().is_relative_to(source_root.resolve())
                and not path.is_symlink() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "selected historical source changed")
    require(source["input_files"] == config["input_files"] and source["binding_sha256"] == config["binding_sha256"], "producer source inventory/binding differs")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    require([row["kind"] for row in timeline] == ["source_bound", "checkpoint_validated", *["layer_completed"] * layers]
            and timeline[0]["native_blocks"] == 0, "trace event program differs")
    call_index = {}
    for index in range(layers):
        event = timeline[index + 2]
        require(event["block_index"] == index and event["counts"] == {key: value * (index + 1) // layers for key, value in expected_counts.items()}, "monotonic native census differs")
        calls = read(root / "calls" / f"{index:02d}.json")
        require(calls["block_index"] == index and calls["parameter_spec"] == binding["final_blocks"][str(index)]
                and [row["role"] for row in calls["calls"]] == list(ROLES), "call/parameter occurrence differs")
        call_index[index] = calls
    originals, exits, payloads = {}, {}, {}
    for role in ROLES:
        spec = config["inputs"][role]
        require(spec["ledger"] in config["input_files"], "historical ledger not inventoried")
        ledger = [json.loads(line) for line in (source_root / spec["ledger"]).read_text().splitlines()]
        entry, exit_row = ledger[5], ledger[25]
        require(spec["entry_row"] == 5 and spec["exit_row"] == 25 and entry["capture"]["cut"] == spec["cut"]
                and source["source_cuts"][role] == spec["cut"] and entry["logical_step"] == exit_row["logical_step"] == 1, "historical source role differs")
        for row in (entry, exit_row):
            require(row["capture"]["retained_full_cut"], "historical complete-cut coverage missing")
            cut_members(source_root, spec["store"], row["capture"]["cut"], config["input_files"])
        original = LocalStore(source_root / spec["store"]).load(spec["cut"])
        native_exit = LocalStore(source_root / spec["store"]).load(exit_row["capture"]["cut"])
        originals[role], exits[role], payloads[role] = original, native_exit, original.payload
        require(original.model_identity == native_exit.model_identity == config["expected_model_identity"], "historical model differs")
        require((payloads[role]["step"], payloads[role]["phase"], payloads[role]["cursor"]) == (1, "single", 0), "historical entry clock differs")
        require((native_exit.payload["step"], native_exit.payload["phase"], native_exit.payload["cursor"]) == (1, "readout", layers), "historical exit clock differs")
        for stream in ("image", "text"):
            require(m["historical_descriptor"](payloads[role][stream]) == entry["outputs"][stream]["descriptor"]
                    and m["historical_descriptor"](native_exit.payload[stream]) == exit_row["outputs"][stream]["descriptor"]
                    == spec["historical_" + stream + "_descriptor"], "historical live array/descriptor differs")
    carried = {role: {key: describe(value) for key, value in payload.items() if key not in ("image", "text", "phase", "cursor")} for role, payload in payloads.items()}
    curves, cuts, cut_index, historical = [], {}, {}, {}
    for index in range(layers + 1):
        fields, cut_index[index] = {}, {}
        for role in ROLES:
            row = read(root / "boundaries" / role / f"{index:02d}.json")
            cut_members(root, "typed-store", row["cut"], inventory["files"])
            cut = LocalStore(root / "typed-store").load(row["cut"])
            payload = cut.payload
            require(row["manifest"] == cut.manifest() and row["parent"] == cut.parent and row["point"] == dict(cut.execution_point or {})
                    and cut.model_identity == config["expected_model_identity"], "boundary manifest/point/model differs")
            if index == 0:
                require(cut.fingerprint == originals[role].fingerprint, "entry boundary not original source")
            else:
                require(cut.parent == cuts[role].fingerprint and cut.execution["component_trace"]["id"] == EXECUTION_ID
                        and cut.execution["component_trace"]["native_source_sha256"] == config["native_source_sha256"], "component parent lineage differs")
                point = dict(edge="before", family="flux2-klein", index=None if index == layers else index, local_clock=index,
                             logical_step=1, phase="readout" if index == layers else "single", operator="readout" if index == layers else "single",
                             next_operation="norm_and_projection" if index == layers else f"single_block.{index}")
                require(dict(cut.execution_point) == point, "actual component cursor differs")
                call = call_index[index - 1]["calls"][ROLES.index(role)]
                require(call["input_cut"] == cuts[role].fingerprint and call["output_cut"] == cut.fingerprint
                        and call["input_point"] == dict(cuts[role].execution_point or {}) and call["output_point"] == point
                        and call["operation"] == f"single_transformer_blocks.{index - 1}" and call["component_call"] is True
                        and call["act_receipt"] is None and call["continue_receipt"] is None, "call/boundary sidecar join differs")
            require((payload["step"], payload["phase"], payload["cursor"]) == (1, "readout" if index == layers else "single", index), "live payload cursor differs")
            require({key: describe(value) for key, value in payload.items() if key not in ("image", "text", "phase", "cursor")} == carried[role], "carried source state differs")
            fields[role] = {}
            for stream in ("image", "text"):
                record = row["fields"][stream]
                require(record["storage"] == dict(kind="typed-cut-slot", store="typed-store", cut=cut.fingerprint, slot=stream), "slot storage join differs")
                actual_descriptor = dict(tensor_descriptor(payload[stream]), stride=list(payload[stream].stride()))
                require(record["native_descriptor"] == actual_descriptor and record["descriptor"] == m["historical_descriptor"](payload[stream]), "field descriptor/layout differs")
                fields[role][stream] = payload[stream].float().numpy()
            cuts[role], cut_index[index][role] = cut, row
        previous = None if not curves else curves[-1]["responses"]
        curves.append(dict(boundary=index, after_block=None if index == 0 else index - 1,
                           phase="pre-single0" if index == 0 else "pre-readout" if index == layers else "single", responses=responses(fields, previous)))
    for role in ROLES:
        historical[role] = {}
        for stream in ("image", "text"):
            actual, expected = cuts[role].payload[stream], exits[role].payload[stream]
            comparison = metrics(actual.float().numpy(), expected.float().numpy())
            comparison["orientation"] = "traced-minus-original-historical-exit"
            for old, new in (("sham_rms", "historical_rms"), ("changed_rms", "traced_rms"), ("relative_to_sham_rms", "relative_to_historical_rms")):
                comparison[new] = comparison.pop(old)
            historical[role][stream] = dict(comparison=comparison,
                exact_descriptor=m["historical_descriptor"](actual) == m["historical_descriptor"](expected),
                historical_cut=exits[role].fingerprint, historical_descriptor=m["historical_descriptor"](expected),
                traced_descriptor=m["historical_descriptor"](actual))
    producer = read(root / "accounts.json")
    observations = dict(producer_full_account_rows_exact=producer["boundaries"] == curves,
        producer_historical_boolean_agreement={role: {stream: summary["native_replay"][role][stream]["exact"] == historical[role][stream]["exact_descriptor"]
                                                     for stream in ("image", "text")} for role in ROLES})
    result = dict(schema="mdb-single-stack-audit-v1", job=summary["job_id"], native_request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, worker_sha256=source["worker_sha256"], auditor_sha256=sha(__file__),
        source_provenance=config["source_provenance"], source_cuts=source["source_cuts"], expected_model_identity=config["expected_model_identity"],
        counts=expected_counts, full_cut_count=3 * (layers + 1), full_boundary_count=layers + 1, curves=curves,
        localization=summarize(curves, layers), historical_replay=historical, producer_diagnostics=observations,
        native_contract=native_contract, selected_backend=read(root / "selected-backend.json"),
        metric_orientation="HH-minus-sham and HL-minus-sham; each native field's sham RMS denominator, absolute native amplitudes retained",
        upstream_custody_scope="Original recovery receipt and full provenance were source-bound by native launcher; this audit independently verifies all selected original cut/ledger files listed by that request, not the complete original 887-file recovery.",
        new_model_calls=0, new_reference_forward_calls=0, new_forecasts=0, equation_forecast_records=0,
        mechanics_status="authenticated-full-cut-saved-array-audit", trend_status="observed", terminal_status="unassessed",
        limitations=["One high receiver with two historical directions, three native stack replays.",
            "Response changes localize boundaries; they do not distinguish attention, MLP, gate, residual or numerical causes.",
            "Different native amplitudes and layer units prevent interpreting raw RMS changes alone as causal amplification.",
            "Replay discrepancies remain per-field observations rather than scientific nulls or panel deletion."])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "call-index.json", call_index)
    save(output / "cut-index.json", cut_index)
    records = [dict(id=f"flux.single-stack.{role}.{stream}", kind="descriptive-observed-native-trace", model=config["expected_model_identity"],
        source_cuts={"changed": source["source_cuts"][role], "sham": source["source_cuts"]["sham"]}, stream=stream,
        clock=dict(logical_step=1, first_single=0, last_single=layers - 1), native_manifest_sha256=expected_manifest,
        curve=[dict(boundary=row["boundary"], after_block=row["after_block"], **row["responses"][role][stream]) for row in curves],
        equation_forecast=False, causal_component_attribution=False, independent_replication=False)
        for role in ("HH", "HL") for stream in ("image", "text")]
    save(output / "trace-records.json", dict(schema="mdb-observed-computation-traces-v1", records=records, audit_sha256=sha(output / "audit.json")))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only audit request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged audit source differs")
    start, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "source.json", "summary.json", "files.json", "accounts.json", "historical-replay.json", "native-contract.json", "selected-backend.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for name in ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json", "binding.json"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    original = read(root / "plan.json")
    save(output / "external-evidence.json", dict(host="beast", roots={"native": str(root), "reader": original["input_root"]},
        manifests={"native": {name: row["sha256"] for name, row in inventory["files"].items()},
                   "reader": {name: row["sha256"] for name, row in original["input_files"].items()}},
        scope="Complete raw LocalStore states remain near data; retrieve individual checksummed members. Reader alias denotes selected original recovered files, not a new reader forecast."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-single-stack-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=result["job"], output_root=str(output), seconds=time.monotonic() - start, model_calls=0, reference_forward_calls=0,
        descriptive_trace_records=4, equation_forecast_records=0, mechanics_status="verified-saved-array-custody", terminal_status="unassessed"))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        output_root=str(output), files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        launcher = runpy.run_path(str(HERE / "stage_saved_flux_audit.py"))
        result = (launcher["submit"](args.run, worker=Path(__file__)) if args.submit else
                  launcher["stage"](args.run, args.study, worker=Path(__file__), helpers=HELPERS,
                    experiment="mdb-single-stack-trace-audit", operations=63,
                    scope="Full63typed-cut single-stack localization, independent full-field response curves and signed increments; zero model/reference recurrence or forecasts."))
        print(json.dumps(result))
