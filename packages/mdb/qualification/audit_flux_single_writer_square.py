"""Saved-only audit of observed parallel writers and eight sealed local returns.

Local equations use observed native H/M and original native projection. The
following single-block futures are observations, never numerical forecasts.
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
WORKER_SHA256 = "69c99f215cf47c63da7e5f4873a94303b7406c1953064f0c1d35ce101db09d10"
ROLES = ("sham", "HH", "HL")
CORNERS = ("00", "01", "10", "11")
COUNTS = dict(native_blocks=19, native_attention=19, linear=46, layernorm=19, rmsnorm=38, swiglu=19,
              source_entries=3, target_entries=8, suffix_entries=8, prediction_to_out=8)
PAIRS = {"H_at_sham_M": ("10", "00"), "H_at_own_M": ("11", "01"),
         "M_at_sham_H": ("01", "00"), "M_at_own_H": ("11", "10"), "joint": ("11", "00")}
HELPERS = ("audit_flux_single_stack_trace.py", "audit_flux_image_tail_microscope.py", "flux_image_tail_microscope.py",
           "retrieve_reader_evidence.py", "flux_family_custody.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def bits(value):
    import numpy as np
    array = np.asarray(value)
    require(array.dtype == np.float32 and np.isfinite(array).all(), "BF16 widening/finite contract differs")
    words = np.ascontiguousarray(array).view(np.uint32)
    require(not np.any(words & 65535), "operand is not exact BF16 widening")
    return (words >> 16).astype("<u2")


def operand(root, record):
    import numpy as np
    path = root / record["path"]
    require(not Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts
            and path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
            and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "operand archive identity differs")
    value = np.load(path, allow_pickle=False)
    require(record["encoding"] == "bf16-little-endian-uint16-npy-v1" and value.dtype == np.dtype("<u2")
            and record["descriptor"]["dtype"] == "torch.bfloat16"
            and list(value.shape) == record["descriptor"]["shape"]
            and hashlib.sha256(value.tobytes(order="C")).hexdigest() == record["payload_sha256"] == record["descriptor"]["sha256"], "operand raw bytes/shape/dtype differs")
    return (value.astype(np.uint32) << 16).view(np.float32)


def compare(candidate, reference):
    import numpy as np
    a, r = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    require(a.shape == r.shape and a.size and np.isfinite(a).all() and np.isfinite(r).all(), "field geometry/nonfinite")
    d = a - r
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    denominator = rms(r)
    return dict(orientation="candidate-minus-reference", rms=rms(d), reference_rms=denominator, candidate_rms=rms(a),
                relative_to_reference_rms=None if denominator == 0 else rms(d) / denominator,
                signed_mean=float(d.mean()), maxabs=float(abs(d).max()), changed_values=int(np.count_nonzero(d)), sample_count=int(d.size),
                bitwise_bf16_exact=bool(np.array_equal(bits(np.asarray(candidate)), bits(np.asarray(reference)))))


def contrast(values, name):
    import numpy as np
    if name == "interaction":
        return np.asarray(values["11"], np.float64) - values["10"] - values["01"] + values["00"]
    high, low = PAIRS[name]
    return np.asarray(values[high], np.float64) - values[low]


def vector_stats(value):
    import numpy as np
    x = np.asarray(value, np.float64)
    return dict(rms=float(np.sqrt(np.mean(x * x))), signed_mean=float(x.mean()), maxabs=float(abs(x).max()), squared_l2=float(np.vdot(x, x)))


def novelty(consumed, own, sham, source_concats, own_H, sham_H, own_M, sham_M):
    import numpy as np
    changed = lambda a, b: int(np.count_nonzero(bits(a) != bits(b)))
    known = [role for role, value in source_concats.items() if changed(consumed, value) == 0]
    return dict(equal_recipient_own=changed(consumed, own) == 0, equal_sham=changed(consumed, sham) == 0,
        changed_vs_recipient_own=changed(consumed, own), changed_vs_sham=changed(consumed, sham),
        H_own_vs_sham_changed_values=changed(own_H, sham_H), M_own_vs_sham_changed_values=changed(own_M, sham_M),
        matches_known_source_concat_roles=known, numerically_new_concat=not known,
        units="Changed BF16 values, including signed-zero bit differences; not a bit population count.", quality_gate=False)


def tensor_square(actual, predicted):
    import numpy as np
    total = contrast(actual, "joint")
    denominator = vector_stats(total)["rms"]
    rows = {}
    for name in (*PAIRS, "interaction"):
        a, p = contrast(actual, name), contrast(predicted, name)
        an, pn = np.linalg.norm(a), np.linalg.norm(p)
        rows[name] = dict(native=vector_stats(a), predicted=vector_stats(p), prediction_minus_native=vector_stats(p - a),
            native_relative_to_joint_rms=None if denominator == 0 else vector_stats(a)["rms"] / denominator,
            error_relative_to_joint_rms=None if denominator == 0 else vector_stats(p - a)["rms"] / denominator,
            cosine=None if an == 0 or pn == 0 else float(np.vdot(a, p) / (an * pn)))
    terms = {name: contrast(actual, name) for name in ("H_at_sham_M", "M_at_sham_H", "interaction")}
    names = tuple(terms)
    cross = {a + "__" + b: float(2 * np.vdot(terms[a], terms[b])) for a, b in
             ((names[0], names[1]), (names[0], names[2]), (names[1], names[2]))}
    energies = {name: float(np.vdot(value, value)) for name, value in terms.items()}
    closure = total - terms[names[0]] - terms[names[1]] - terms[names[2]]
    return dict(native_joint_rms=denominator, denominator="Native11-minus00 at fixed recipient R/gate; 00 is not historical sham.", contrasts=rows,
        energy=dict(total_squared_l2=float(np.vdot(total, total)), terms=energies, signed_cross_terms=cross,
                    sum_terms_and_cross=sum(energies.values()) + sum(cross.values()), closure=vector_stats(closure)),
        scope="Local shared affine projection plus BF16 multiply/add; local interaction includes precision effects, not new nonlinear H–M coupling.")


def panel_accounts(fields, baseline, entries):
    panels, entry, growth, scalar = {}, {}, {}, {}
    for role in ("HH", "HL"):
        entry[role] = {stream: compare(entries[role][stream], entries["sham"][stream]) for stream in ("image", "text")}
    for stage in ("local", "final"):
        scalar[stage] = {}
        for corner in CORNERS:
            label = stage + "." + corner
            panels[label] = {role: {stream: compare(fields[stage][role + "." + corner][stream], baseline[stage][stream])
                                   for stream in ("image", "text")} for role in ("HH", "HL")}
            growth[label] = {}
            for stream in ("image", "text"):
                high = panels[label]["HH"][stream]["rms"] - entry["HH"][stream]["rms"]
                low = panels[label]["HL"][stream]["rms"] - entry["HL"][stream]["rms"]
                growth[label][stream] = dict(HH_growth=high, HL_growth=low, excess_growth=high - low)
        for role in ("HH", "HL"):
            scalar[stage][role] = {}
            for stream in ("image", "text"):
                scalar[stage][role][stream] = {}
                for metric in ("rms", "relative_to_reference_rms", "signed_mean", "maxabs"):
                    values = {corner: panels[stage + "." + corner][role][stream][metric] for corner in CORNERS}
                    scalar[stage][role][stream][metric] = (None if any(x is None for x in values.values()) else
                        {name: float(contrast(values, name)) for name in (*PAIRS, "interaction")})
    restoration = {stage: {stream: growth[stage + ".00"][stream]["excess_growth"] - growth[stage + ".11"][stream]["excess_growth"]
                           for stream in ("image", "text")} for stage in ("local", "final")}
    return dict(entry=entry, panels=panels, growth=growth, scalar_factorials=scalar, signed_joint_restoration=restoration,
        hypothesis=dict(contrast="HH-minus-HL excess absolute response growth: 00-minus11", expected_sign="negative", predeclared_streams=["image"],
            observations={stage: {stream: dict(signed_effect=value, predeclared=stream == "image",
                supports_predeclared_direction=(value < 0) if stream == "image" else None,
                status=("supports-direction" if value < 0 else "opposite-direction" if value > 0 else "zero") if stream == "image"
                    else ("exploratory-negative" if value < 0 else "exploratory-positive" if value > 0 else "zero")) for stream, value in values.items()}
                for stage, values in restoration.items()}, quality_gate=False),
        scope="Local and final response metrics use actual historical sham fields at their own clocks. Growth subtracts entry response RMS; scalar factorials are not intrinsic component gains.")


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np
    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    root, output = Path(root), Path(output)
    t = runpy.run_path(str(HERE / "audit_flux_single_stack_trace.py"))
    m = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))
    inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(inventory["files"]) | {"files.json"}, "raw inventory coverage differs")
    config, source, summary = [read(root / name) for name in ("plan.json", "source.json", "summary.json")]
    require(all(row["request_sha256"] == expected_request for row in (config, source, summary, inventory))
            and len({source["job_id"], summary["job_id"], inventory["job_id"]}) == 1, "native request/job differs")
    require(source["worker_sha256"] == WORKER_SHA256 and source["helper_source_sha256"] == config["helper_source_sha256"], "worker/helper source differs")
    require(source["source_provenance"] == config["source_provenance"] and source["input_files"] == config["input_files"], "trace custody differs")
    require(summary["counts"] == COUNTS and summary["complete_typed_cuts"] == 19 and summary["source_typed_cuts"] == 3
            and summary["intervention_typed_cuts"] == 16, "native census/retained coverage differs")
    require(all(summary[key] == 0 for key in ("act_receipts", "session_continue_receipts", "vae_calls", "encodes", "noise_draws"))
            and summary["suffix_steps_per_case"] == 1, "declared forbidden work differs")
    selected, suffix, layers = (config[key] for key in ("selected_block", "suffix_block", "block_count"))
    require(suffix == selected + 1 and layers == suffix + 1, "native horizon differs")
    ids = [role + "." + corner for role in ("HH", "HL") for corner in CORNERS]
    native_contract = read(root / "native-contract.json")
    require(native_contract["count_contract"]["counts"] == COUNTS and native_contract["module_config"] == config["module_config"]
            and native_contract["module_source_sha256"] == config["native_source_sha256"], "native contract differs")
    source_root = Path(config["input_root"])
    for name, row in config["input_files"].items():
        path = source_root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.resolve().is_relative_to(source_root.resolve())
                and not path.is_symlink() and sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "source inventory changed")
    for key, name in (("binding", "binding.json"), ("parameter_manifest", "parameter-artifacts.json"),
                      ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
        require(sha(root / name) == config[key + "_sha256"], "copied source metadata differs")
    transaction = root / "transaction"
    report = summary["transaction"]
    verified = verify_transaction(transaction, expected_seal_sha256=summary["seal"]["seal_sha256"], expected_event_head=report["event_head"], max_bytes=2 << 30)
    require(verified["phase"] == report["phase"] == "closed" and report["failure"] is None and not report["violations"]
            and report["frontiers"] == ["observed_sources"] and report["forecasts"] == ids and report["observations"] == ["local_targets", "single_suffix"], "closed transaction program differs")
    seal = read(transaction / "seal.json")
    require(seal["pre_release_work"]["executions"] == 0, "sealed target exposure differs")
    declarations = seal["declarations"]
    require(declarations["model_identity"] == config["expected_model_identity"] and declarations["source_identity"] == config["source_binding_sha256"]
            and declarations["input_identity"] == m["identity"](source["source_cuts"]), "sealed source/model identity differs")
    require(plain(load_forecast_tree(transaction, declarations["actual_weight_manifest"]))
            == dict(binding_sha256=config["binding_sha256"], parameter_manifest_sha256=config["parameter_manifest_sha256"]), "sealed parameter binding differs")
    frontier = load_forecast_tree(transaction, read(transaction / "frontiers/observed_sources.json")["value"])
    predictions = {key: load_forecast_tree(transaction, read(transaction / "forecasts" / (key + ".json"))["predictions"]) for key in ids}
    observed = {name: load_forecast_tree(transaction, read(transaction / "observations" / (name + ".json"))["value"]) for name in report["observations"]}
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    expected = [("source", role, selected) for role in ROLES] + [("seal", None, None)] + [("target", key, selected) for key in ids] + [("suffix", key, suffix) for key in ids]
    require(len(timeline) == len(expected), "native event count differs")
    running = dict.fromkeys(COUNTS, 0)
    for event, (stage, case, index) in zip(timeline, expected):
        if stage == "seal":
            running["linear"] += 8
            running["prediction_to_out"] += 8
            require(event["kind"] == "globally_sealed" and event["seal"] == summary["seal"] and event["target_entries_at_seal"] == 0, "global seal chronology differs")
            require(event["counts"] == running, "forecast primitive census differs")
        else:
            running["native_blocks"] += 1
            running[stage + "_entries"] += 1
            require(event["kind"] == "native_entry" and (event["stage"], event["case"], event["block_index"]) == (stage, case, index)
                    and event["counts"] == running, "native entry/order census differs")
            for key, count in (("native_attention", 1), ("linear", 2), ("layernorm", 1), ("rmsnorm", 2), ("swiglu", 1)):
                running[key] += count
    require(running == COUNTS, "final primitive ledger differs")

    def load_record(base, store, row, member_inventory):
        t["cut_members"](base, store, row["cut"], member_inventory)
        cut = LocalStore(base / store).load(row["cut"])
        require(cut.manifest() == row["manifest"] and cut.parent == row["parent"] and dict(cut.execution_point or {}) == row["point"]
                and cut.model_identity == config["expected_model_identity"], "cut metadata/point/model differs")
        for stream in ("image", "text"):
            value = cut.payload[stream]
            require(tensor_descriptor(value)["dtype"] == "torch.bfloat16" and value.ndim == 3 and value.shape[0] == 1
                    and value.shape[-1] == config["module_config"]["dim"], "cut native field geometry/dtype differs")
            require(row["fields"][stream]["descriptor"] == m["historical_descriptor"](value)
                    and row["fields"][stream]["native_descriptor"] == dict(tensor_descriptor(value), stride=list(value.stride())), "cut field descriptor/layout differs")
        return cut

    def fields(cut):
        payload = cut.payload
        return {stream: payload[stream].float().numpy() for stream in ("image", "text")}

    parents, historical, source_calls, source_controls, source_cut_index = {}, {}, read(root / "source-calls.json"), {}, {}
    for role in ROLES:
        spec = config["inputs"][role]
        metadata = {key: read(source_root / spec[key]) for key in ("entry_metadata", "exit_metadata", "final_metadata")}
        require(all(spec[key] in config["input_files"] for key in metadata), "trace metadata not inventoried")
        historical[role] = {key: load_record(source_root, spec["store"], row, config["input_files"]) for key, row in metadata.items()}
        parents[role] = historical[role]["entry_metadata"]
        require(parents[role].fingerprint == spec["cut"] == source["source_cuts"][role], "trace source-cut differs")
        p = parents[role].payload
        require((p["step"], p["phase"], p["cursor"]) == (1, "single", selected), "source cursor differs")
        src = source_calls[role]
        require(src["input_cut"] == parents[role].fingerprint and src["block_index"] == selected and src["hook_return"] == "None", "source call differs")
        concat = operand(root, src["observed_concat"])
        require(np.array_equal(bits(concat), bits(np.concatenate([frontier[role]["H"], frontier[role]["M"]], axis=-1))), "sealed H/M differ from observed source concat")
        width = config["module_config"]["dim"]
        residual = np.concatenate([p["text"].float().numpy(), p["image"].float().numpy()], axis=1)
        mod = p["context"]["single_mod"].float().numpy()
        gate = mod.reshape(mod.shape[0], -1, 3 * width)[..., 2 * width:]
        require(np.array_equal(bits(frontier[role]["R"]), bits(residual)) and np.array_equal(bits(frontier[role]["gate"]), bits(gate)), "sealed R/gate differs from actual input")
        original = load_record(root, "typed-store", src["output"], inventory["files"])
        source_cut_index[role] = src["output"]
        require(original.parent == parents[role].fingerprint and (original.payload["step"], original.payload["phase"], original.payload["cursor"])
                == (1, "single", selected + 1), "source output lineage/cursor differs")
        require({key: describe(value) for key, value in original.payload.items() if key not in ("image", "text", "phase", "cursor")}
                == {key: describe(value) for key, value in p.items() if key not in ("image", "text", "phase", "cursor")}, "source carried state differs")
        require(all(src["outputs"][stream]["descriptor"] == src["output"]["fields"][stream]["descriptor"]
                    and src["outputs"][stream]["native_descriptor"] == src["output"]["fields"][stream]["native_descriptor"] for stream in ("image", "text")), "source output observation join differs")
        source_controls[role] = {stream: compare(fields(original)[stream], fields(historical[role]["exit_metadata"])[stream]) for stream in ("image", "text")}
    entries = {role: fields(cut) for role, cut in parents.items()}
    baselines = {stage: fields(historical["sham"][key]) for stage, key in (("local", "exit_metadata"), ("final", "final_metadata"))}
    values = {"local": {}, "final": {}}
    local_records, final_records, local_errors, controls, input_novelty = {}, {}, {}, {}, {}
    for key in ids:
        role, corner = key.split(".")
        case = dict(id=key, recipient=role, H=role if corner[0] == "1" else "sham", M=role if corner[1] == "1" else "sham", newly_mixed=corner[0] != corner[1])
        local_row, final_row = read(root / "cases" / (key + ".local.json")), read(root / "cases" / (key + ".final.json"))
        require(local_row["case"] == final_row["case"] == plain(predictions[key]["case"]) == case, "case declaration differs")
        require(predictions[key]["hypothesis"] == native_contract["count_contract"]["hypothesis"], "sealed hypothesis differs")
        local = load_record(root, "typed-store", local_row["local"], inventory["files"])
        final = load_record(root, "typed-store", final_row["final"], inventory["files"])
        for cut, parent, cursor, phase in ((local, parents[role], selected + 1, "single"), (final, local, layers, "readout")):
            p, incoming = cut.payload, parent.payload
            require(cut.parent == parent.fingerprint and (p["step"], p["phase"], p["cursor"]) == (1, phase, cursor)
                    and cut.execution["component_trace"]["id"] == "mdb-flux-single-writer-square-v1"
                    and cut.execution["component_trace"]["case"] == key, "intervention/continuation lineage differs")
            require(all(p[stream].shape == incoming[stream].shape for stream in ("image", "text")), "continuation stream geometry differs")
            require({name: describe(value) for name, value in p.items() if name not in ("image", "text", "phase", "cursor")}
                    == {name: describe(value) for name, value in incoming.items() if name not in ("image", "text", "phase", "cursor")}, "carried source state changed")
        receiving, consumed = [operand(root, local_row["operands"][name]) for name in ("receiving", "consumed")]
        own = np.concatenate([frontier[role]["H"], frontier[role]["M"]], axis=-1)
        chosen = np.concatenate([frontier[case["H"]]["H"], frontier[case["M"]]["M"]], axis=-1)
        require(np.array_equal(bits(receiving), bits(own)) and np.array_equal(bits(consumed), bits(chosen)), "receiving/consumed writer operands differ")
        input_novelty[key] = dict(structurally_mixed=case["newly_mixed"], **novelty(consumed, own,
            np.concatenate([frontier["sham"]["H"], frontier["sham"]["M"]], axis=-1),
            {source_role: np.concatenate([frontier[source_role]["H"], frontier[source_role]["M"]], axis=-1) for source_role in ROLES},
            frontier[role]["H"], frontier["sham"]["H"], frontier[role]["M"], frontier["sham"]["M"]))
        require(local_row["operands"]["receiving"]["descriptor"]["stride"] == source_calls[role]["observed_concat"]["descriptor"]["stride"]
                and local_row["operands"]["consumed"]["descriptor"] == local_row["operands"]["expected_descriptor"]
                and local_row["operands"]["exact_expected"] is True, "consumer operand layout differs")
        contiguous, step = [], 1
        for size in reversed(consumed.shape):
            contiguous.insert(0, step)
            step *= size
        require(local_row["operands"]["consumed"]["descriptor"]["stride"] == contiguous, "consumed cat/clone layout is not native contiguous")
        values["local"][key], values["final"][key] = fields(local), fields(final)
        local_errors[key] = {stream: compare(np.asarray(predictions[key][stream]), values["local"][key][stream]) for stream in ("image", "text")}
        require(plain(observed["local_targets"]["cases"][ids.index(key)]) == dict(id=key, prediction_exact=local_row["prediction_exact"], cut=local.fingerprint)
                and observed["single_suffix"]["cuts"][key] == final.fingerprint, "closed observation/sidecar differs")
        local_records[key], final_records[key] = local_row, final_row
        if corner == "11":
            controls[key] = {stage: {stream: compare(values[stage][key][stream], fields(historical[role][ref])[stream]) for stream in ("image", "text")}
                            for stage, ref in (("local", "exit_metadata"), ("final", "final_metadata"))}
    accounts = panel_accounts(values, baselines, entries)
    squares = {role: {stream: tensor_square({corner: values["local"][role + "." + corner][stream] for corner in CORNERS},
        {corner: np.asarray(predictions[role + "." + corner][stream]) for corner in CORNERS}) for stream in ("image", "text")} for role in ("HH", "HL")}
    result = dict(schema="mdb-single-writer-square-audit-v1", job=summary["job_id"], native_request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, auditor_sha256=sha(__file__), worker_sha256=source["worker_sha256"],
        counts=COUNTS, source_provenance=config["source_provenance"], source_cuts=source["source_cuts"], native_contract=native_contract,
        verification=verified, local_prediction_errors=local_errors, source_controls=source_controls, own11_trace_controls=controls,
        input_novelty=input_novelty,
        accounts=accounts, local_tensor_squares=squares, producer_accounts=read(root / "accounts.json"),
        component_scope="Observed native H/M, shared original projection and BF16 write; not independent Flash/GEMM or upstream formation prediction.",
        final_scope="One unchanged native single19 from each actual local cut; no final numerical forecast or equation record.",
        new_model_calls=0, new_reference_forward_calls=0, mechanics_status="authenticated-full-array-audit", trend_status="observed", terminal_status="unassessed")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "cut-index.json", dict(source=source_cut_index, local={key: row["local"] for key, row in local_records.items()}, final={key: row["final"] for key, row in final_records.items()}))
    save(output / "call-index.json", dict(source=source_calls, local=local_records, final=final_records, timeline=timeline))
    catalog, queries = EquationCatalog(output / "catalog/equations"), []
    def record(role, occurrence, predicted, actual, kind):
        context = dict(model_identity=config["expected_model_identity"], recipient_cut=parents[role].fingerprint, occurrence=occurrence, kind=kind,
            clock=dict(step=1, phase="single", index=selected), horizon=1, native_manifest_sha256=expected_manifest,
            seal_sha256=summary["seal"]["seal_sha256"], numerical=native_contract,
            binding_sha256=config["binding_sha256"], source_cut_set=source["source_cuts"])
        item = catalog.record(law_id="law.flux.single.observed-parallel-writers.v1", route_id="route.flux.single.H-M.to_out.gate.residual.v1",
            alias="cmp.flux.single-writer-square." + occurrence + "." + kind,
            binding=dict(model_identity=config["expected_model_identity"], input_identity=parents[role].fingerprint, source_identity=expected_request,
                         frontier=context["clock"], execution=native_contract, horizon=1), report=compare_equations(predicted, actual, max_elements=25_000_000),
            conditions=[dict(field=name, op="eq", value=value) for name, value in context.items()], tested_context=context,
            readers=["native-single-shared-writer"], evidence=dict(audit_sha256=sha(output / "audit.json"), conditional_native_primitives=True,
                independent_kernel_arithmetic=False, final_numerical_forecast=False, independent_replication=False, quality_gate=False))
        wrong = {**context, "clock": {**context["clock"], "index": suffix}}
        exact, changed = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to=wrong))
        require(exact == 1 and changed == 0, "catalog applicability differs")
        queries.append(dict(fingerprint=item["fingerprint"], occurrence=occurrence, kind=kind, exact=exact, wrong_clock=changed))
    for key in ids:
        record(key.split(".")[0], key, {stream: predictions[key][stream] for stream in ("image", "text")}, values["local"][key], "local-endpoint")
    for role in ("HH", "HL"):
        for name in (*PAIRS, "interaction"):
            predicted = {stream: contrast({corner: predictions[role + "." + corner][stream] for corner in CORNERS}, name) for stream in ("image", "text")}
            actual = {stream: contrast({corner: values["local"][role + "." + corner][stream] for corner in CORNERS}, name) for stream in ("image", "text")}
            record(role, role + "." + name, predicted, actual, "local-tensor-contrast")
    save(output / "catalog/provenance.json", dict(records=len(queries), queries=queries, native_manifest_sha256=expected_manifest,
        independent_tasks=1, dependent_local_comparisons=20, family_induction=False, final_forecast_records=0))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU audit request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged source differs")
    start, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "source.json", "summary.json", "files.json", "accounts.json", "source-calls.json", "native-contract.json", "selected-backend.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for name in ("binding.json", "parameter-artifacts.json", "retrieve_weights.py", "source-binding.json"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(root / "transaction/seal.json", output / "source-transaction-seal.json")
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    original = read(root / "plan.json")
    save(output / "external-evidence.json", dict(host="beast", roots={"native": str(root), "reader": original["input_root"]},
        manifests={"native": {name: row["sha256"] for name, row in inventory["files"].items()},
                   "reader": {name: row["sha256"] for name, row in original["input_files"].items()}},
        scope="Raw sealed arrays, actual operands and full typed cuts remain near data; retrieve individual members by checksum."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-single-writer-square-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=result["job"], records=20, output_root=str(output), seconds=time.monotonic() - start,
        model_calls=0, reference_forward_calls=0, mechanics_status="verified-saved-array-custody", terminal_status="unassessed"))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        launch = runpy.run_path(str(HERE / "stage_saved_flux_audit.py"))
        result = (launch["submit"](args.run, worker=Path(__file__)) if args.submit else launch["stage"](args.run, args.study,
            worker=Path(__file__), helpers=HELPERS, experiment="mdb-single-writer-square-audit", operations=20,
            scope="Eight sealed local writer endpoints and12local tensor contrasts, actual operand/fullcut joins, native final observations; zero model/reference recurrence."))
        print(json.dumps(result))
