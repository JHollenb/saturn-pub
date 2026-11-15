"""Full saved-state audit of four sealed functional FLUX denoiser passes.

No model is constructed or called and no recurrence is recomputed. The source
graph, guards and primitive descriptors remain distinct from independent-kernel
arithmetic. Next-project calls are observed consumers, not forecasts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "7fdacdec13ada95309869cebad9fd90364141039042953bf49f4c4838fada1af"
CASES = ("sham", "HH", "HL", "new-3over8")
ROLES = CASES[:3]
WORKING = ("image", "text", "context", "noise_prediction")
CARRIED = ("conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")
HELPERS = ("audit_flux_single_stack_trace.py", "audit_flux_image_tail_microscope.py",
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


def identity(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def schedule(joints, singles):
    before = [dict(step=1, phase="project", cursor=0)]
    before += [dict(step=1, phase="joint", cursor=i) for i in range(joints)]
    before += [dict(step=1, phase="single", cursor=i) for i in range(singles)]
    before += [dict(step=1, phase="readout", cursor=singles), dict(step=1, phase="scheduler", cursor=singles)]
    return before, [*before[1:], dict(step=2, phase="project", cursor=0)]


def point(clock):
    phase, cursor = clock["phase"], clock["cursor"]
    return dict(family="flux2-klein", logical_step=clock["step"], phase=phase, operator=phase,
        index=cursor if phase in ("joint", "single") else None, edge="before", local_clock=cursor,
        next_operation={"project": "native_input_projection", "readout": "norm_and_projection", "scheduler": "flow_scheduler_commit"}.get(phase, f"{phase}_block.{cursor}"))


def expected_counts(joints, singles):
    one = dict(transitions=joints + singles + 3, linear=9 + 12 * joints + 2 * singles,
        layernorm=4 * joints + singles + 1, rmsnorm=4 * joints + 2 * singles, swiglu=2 * joints + singles,
        sdpa=joints + singles, rotary=2 * (joints + singles), rope=2, timestep=1)
    forecast = {**{key: 4 * value for key, value in one.items()}, "full_blocks": 0, "scheduler_calls": 0, "module_entries": 0}
    native = {**{key: 4 * value for key, value in one.items()}, "full_blocks": 4 * (joints + singles), "scheduler_calls": 4,
              "module_entries": 4 * (24 + 29 * joints + 9 * singles)}
    consumers = dict(transitions=8, linear=56, layernorm=0, rmsnorm=0, swiglu=0, sdpa=0, rotary=0, rope=16,
                     timestep=8, full_blocks=0, scheduler_calls=0, module_entries=152)
    return dict(forecast=forecast, native=native, consumers=consumers)


def numpy(value):
    return value.detach().cpu().numpy() if str(value.dtype) != "torch.bfloat16" else value.detach().float().cpu().numpy()


def metric(predicted, native):
    import numpy as np
    p0, n0 = np.asarray(predicted), np.asarray(native)
    p, n = np.asarray(p0, np.float64), np.asarray(n0, np.float64)
    require(p.shape == n.shape and p.size and np.isfinite(p).all() and np.isfinite(n).all(), "field geometry/nonfinite")
    d = p - n
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    scale, pn, nn = rms(n), np.linalg.norm(p.ravel()), np.linalg.norm(n.ravel())
    return dict(orientation="predicted-minus-native", rms=rms(d), native_rms=scale, predicted_rms=rms(p),
        relative_rms=None if scale == 0 else rms(d) / scale, signed_mean=float(d.mean()), maxabs=float(abs(d).max()),
        changed_values=int(np.count_nonzero(d)), sample_count=int(d.size), cosine=None if pn == 0 or nn == 0 else float(np.vdot(p, n) / (pn * nn)),
        exact=bool(np.array_equal(p, n)), bitwise_exact=p0.dtype == n0.dtype and p0.shape == n0.shape and p0.tobytes(order="C") == n0.tobytes(order="C"),
        abs_error_quantiles={str(q): float(np.quantile(abs(d), q)) for q in (.5, .95, .99, 1.)})


def tensor_fields(payload):
    result = {key: payload[key] for key in ("image", "text", "latent", "noise_prediction") if payload[key] is not None}
    if payload["context"] is not None:
        result.update({"context." + key: payload["context"][key] for key in ("temb", "img_mod", "txt_mod", "single_mod")})
        result.update({"context.rope." + str(i): payload["context"]["rope"][i] for i in (0, 1)})
    return result


def fields(payload):
    return {key: numpy(value) for key, value in tensor_fields(payload).items()}


def compare_states(predicted, native):
    a, b = tensor_fields(predicted), tensor_fields(native)
    require(set(a) == set(b), "working-state liveness differs")
    rows = {}
    for key in a:
        row = metric(numpy(a[key]), numpy(b[key]))
        row.update(predicted_native_dtype=str(a[key].dtype), observed_native_dtype=str(b[key].dtype),
                   dtype_exact=a[key].dtype == b[key].dtype, layout_exact=a[key].stride() == b[key].stride())
        row["bitwise_exact"] = row["bitwise_exact"] and row["dtype_exact"]
        rows[key] = row
    return rows


def rne_bf16(value):
    import numpy as np
    x = np.ascontiguousarray(value, dtype=np.float32)
    require(np.isfinite(x).all(), "input derivation nonfinite")
    words = x.view(np.uint32)
    return (((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype(np.uint32) << 16).view(np.float32)


def restore_parameters(manifest, root, expected, *, expected_module_sha256):
    """Stream original checkpoint and exact archive framing, no export/decoding."""
    import mdb.parameter_artifacts as api
    require(sha(api.__file__) == expected_module_sha256, "installed parameter restoration module differs")
    manifest = api.validate_manifest(manifest)
    require({row["source"]["tensor"]: row["source"]["payload_sha256"] for row in manifest["artifacts"]} == expected,
            "original parameter identity set differs")
    rows, total = {}, 0
    for entry, stream, size in api._source_entries(manifest, root):
        raw, archive = hashlib.sha256(), hashlib.sha256(base64.b64decode(entry["npy_prefix_b64"], validate=True))
        left = size
        while left:
            chunk = stream.read(min(left, 8 << 20))
            require(bool(chunk), "parameter payload truncated")
            raw.update(chunk)
            archive.update(chunk)
            left -= len(chunk)
        require(raw.hexdigest() == entry["source"]["payload_sha256"] and archive.hexdigest() == entry["sha256"], "exact parameter restoration differs")
        rows[entry["source"]["tensor"]] = dict(payload_sha256=raw.hexdigest(), archive_sha256=archive.hexdigest(), payload_bytes=size)
        total += size
    return dict(schema="mdb-streamed-parameter-restoration-audit-v1", tensors=len(rows), payload_bytes=total, rows=rows,
        module_sha256=expected_module_sha256, checkpoint_files={row["source"]["file"]: row["source"]["file_sha256"] for row in manifest["artifacts"]},
        scope="Full source-file SHA once/file, safetensors header/offset checks, every payload and original NPY archive SHA; no parameter export or neural call.")


def verify_external(root, records, prefixes):
    actual = {path.relative_to(root).as_posix() for prefix in prefixes for path in (root / prefix).rglob("*") if path.is_file()}
    require(set(records) == actual, "sealed external inventory scope differs")
    for name, row in records.items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve()) and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "sealed external evidence differs")


def witness_comparison(predicted, native, width):
    import numpy as np
    p, n = {row["id"]: row for row in predicted}, {row["id"]: row for row in native}
    require(len(p) == len(predicted) and len(n) == len(native), "duplicate primitive occurrence ID")
    for records in (p, n):
        for row in records.values():
            for value in row["fields"].values():
                size = int(np.prod(value["descriptor"]["shape"], dtype=np.int64))
                expected = np.unique(np.linspace(0, size - 1, min(width, size)).astype(np.int64)).tolist()
                require(value["indices"] == expected and len(value["values"]) == len(expected) and value["full_values_retained"] is False
                        and np.isfinite(np.asarray(value["values"], np.float64)).all(), "fixed primitive witness geometry differs")
    rows = {}
    for key, row in p.items():
        other = n.get(key)
        available = other is not None and other["kind"] == row["kind"] and set(other["fields"]) == set(row["fields"])
        fields_report = {}
        if available:
            for name, value in row["fields"].items():
                actual = other["fields"][name]
                pd, nd = value["descriptor"], actual["descriptor"]
                fields_report[name] = dict(full_descriptor_exact={k: v for k, v in pd.items() if k != "stride"} == {k: v for k, v in nd.items() if k != "stride"},
                    layout_exact=pd["stride"] == nd["stride"], witness=metric(np.asarray(value["values"], np.float32), np.asarray(actual["values"], np.float32)),
                    scope="Full source-recorded descriptor commitment plus fixed sampled values; intermediate arrays not retained for independent full norm recomputation.")
        rows[key] = dict(available=available, fields=fields_report,
            full_descriptor_exact=available and all(v["full_descriptor_exact"] for v in fields_report.values()),
            witness_bitwise_exact=available and all(v["witness"]["bitwise_exact"] for v in fields_report.values()),
            layout_exact=available and all(v["layout_exact"] for v in fields_report.values()))
    return dict(ports=rows, unmatched_native_ports=sorted(set(n) - set(p)))


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np
    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    root, output = Path(root), Path(output)
    inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(inventory["files"]) | {"files.json"}, "raw manifest coverage differs")
    config, source, summary, contract = [read(root / name) for name in ("plan.json", "source.json", "summary.json", "native-contract.json")]
    require(all(row["request_sha256"] == expected_request for row in (config, source, summary, inventory))
            and len({source["job_id"], summary["job_id"], inventory["job_id"]}) == 1, "raw job/request differs")
    require(source["worker_sha256"] == WORKER_SHA256 and source["helper_source_sha256"] == config["helper_source_sha256"], "executed worker/helper source differs")
    require(source["source_provenance"] == config.get("source_provenance"), "source provenance differs")
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script"):
        require(sha(root / Path(config[key]).name) == config[key + "_sha256"], "copied source metadata differs")
    model_config = read(root / Path(config["transformer_config"]).name)
    joints, singles = model_config["num_layers"], model_config["num_single_layers"]
    before, after = schedule(joints, singles)
    length = len(before)
    expected = expected_counts(joints, singles)
    require(summary["counts"] == expected and summary["cases"] == list(CASES), "exact native/functional/module census differs")
    require(summary["totals"] == dict(model_loads=1, historical_cut_reads=3, encodes=0, noise_draws=0, vae_calls=0, guard_refusals=0), "load/guard/forbidden work differs")
    require(summary["complete_forecast_cuts"] == summary["complete_native_cuts"] == 4 * length
            and summary["complete_consumer_cuts"] == 8 and summary["act_receipts"] == summary["session_continue_receipts"] == 0, "full state coverage differs")
    require(contract["source_model_identity"] == config["expected_source_model_identity"] and contract["verification_adapter_identity"]
            and contract["actual_attention_backend"] == "native" and contract["forecast_helper_sha256"] == config["helper_source_sha256"]["flux_functional_denoiser.py"], "numeric/source adapter contract differs")
    member_helper = runpy.run_path(str(HERE / "audit_flux_single_stack_trace.py"))["cut_members"]

    def load_cut(base, store, identifier, records):
        member_helper(base, store, identifier, records)
        cut = LocalStore(base / store).load(identifier)
        require(cut.model_identity == config["expected_source_model_identity"], "source identity replaced by verification adapter identity")
        return cut

    def source_files(base, sources, records):
        members = set()
        for spec in sources.values():
            members.update(member_helper(base, spec["store"], spec["cut"], records))
        require(members == set(records), "input closure contains undeclared/intermediate/future evidence")
        for name, row in records.items():
            path = base / name
            require(not path.is_symlink() and path.resolve().is_relative_to(base.resolve()) and sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "original source bytes differ")

    source_files(Path(config["input_root"]), config["inputs"], config["input_files"])
    tx0, tx = root / "input-transaction", root / "forecast-transaction"
    input_verify = verify_transaction(tx0, expected_seal_sha256=summary["input_seal"]["seal_sha256"], max_bytes=64 << 20)
    forecast_verify = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"], expected_event_head=summary["transaction"]["event_head"], max_bytes=256 << 20)
    require(input_verify["phase"] == forecast_verify["phase"] == summary["transaction"]["phase"] == "closed"
            and summary["transaction"]["failure"] is None and not summary["transaction"]["violations"], "transactions not closed cleanly")
    p0 = load_forecast_tree(tx0, read(tx0 / "forecasts/inputs.json")["predictions"])
    predicted = load_forecast_tree(tx, read(tx / "forecasts/cohort.json")["predictions"])
    require(plain(p0["cases"]) == source["input_records"] and p0["target_entries"] == 0
            and plain(predicted["input_seal"]) == summary["input_seal"] == source["input_seal"]
            and source["forecast_seal"] == summary["forecast_seal"] and predicted["native_full_blocks_at_seal"] == predicted["native_scheduler_calls_at_seal"] == 0,
            "globally sealed case/input bindings differ")
    verify_external(root, p0["external_files"], ("input-store",))
    verify_external(root, predicted["external_files"], ("prediction-store", "forecast-ports", "forecast-rows"))
    require(set(predicted["cases"]) == set(CASES) and plain(predicted["forecast_counts"]) == expected["forecast"], "forecast cohort/census differs")
    origins = {case: load_cut(root, "input-store", p0["cases"][case]["cut"], inventory["files"]) for case in CASES}
    for case, cut in origins.items():
        p = cut.payload
        require((p["step"], p["phase"], p["cursor"]) == (1, "project", 0) and all(p[key] is None for key in WORKING)
                and dict(cut.execution_point) == point(before[0]), "complete original project input closure differs")
        require(plain(p0["cases"][case]["payload"]) == plain(describe(p)) and p0["cases"][case]["point"] == point(before[0]), "sealed original payload differs")
        if case in ROLES:
            original = load_cut(Path(config["input_root"]), config["inputs"][case]["store"], config["inputs"][case]["cut"], config["input_files"])
            require(original.fingerprint == cut.fingerprint, "copied original input identity differs")
        if case != "sham":
            require(all(describe(p[key]) == describe(origins["sham"].payload[key]) for key in p if key != "latent"), "nonlatent input nuisance differs")
    z, hh, new = [numpy(origins[case].payload["latent"]) for case in ("sham", "HH", CASES[-1])]
    delta = np.subtract(hh, z, dtype=np.float32)
    proposed = np.add(z, np.multiply(delta, np.float32(3 / 8), dtype=np.float32), dtype=np.float32)
    require(new.tobytes(order="C") == rne_bf16(proposed).tobytes(order="C") and origins[CASES[-1]].parent == origins["sham"].fingerprint
            and list(origins[CASES[-1]].execution["input_derivation"]["coefficient"]) == [3, 8], "fixed BF16 3/8 input recipe differs")
    require(origins[CASES[-1]].payload["latent"].stride() == origins["sham"].payload["latent"].stride(), "derived native layout differs")
    effective, direction = new.astype(np.float64) - z, hh.astype(np.float64) - z
    direction_norm = float(np.linalg.norm(direction.ravel()))
    novelty = dict(coefficient=[3, 8], bitwise_matches_existing={role: new.tobytes(order="C") == numpy(origins[role].payload["latent"]).tobytes(order="C") for role in ROLES},
        changed_vs_sham_values=int(np.count_nonzero(new.view(np.uint32) != z.view(np.uint32))), geometry=metric(effective, (3 / 8) * direction),
        effective_norm_ratio=None if direction_norm == 0 else float(np.linalg.norm(effective.ravel()) / direction_norm),
        effective_projection_dose=None if direction_norm == 0 else float(np.vdot(effective, direction) / direction_norm ** 2),
        original_geometry=read(root / "input-geometry.json"), quality_gate=False)
    novelty["numerically_novel_input"] = not any(novelty["bitwise_matches_existing"].values())
    parameter_manifest = read(root / Path(config["parameter_manifest"]).name)
    restoration = restore_parameters(parameter_manifest, config["parameter_root"], source["original_parameter_sha256"], expected_module_sha256=config["retrieval_script_sha256"])
    require(restoration["tensors"] == contract["original_parameter_count"] and restoration["payload_bytes"] == contract["original_parameter_bytes"], "restored parameter coverage differs")
    if config.get("profile") != "tiny-cpu":
        require(restoration["tensors"] == 169 and (joints, singles) == (5, 20), "production geometry differs")

    def cut_row(store, row, parent, clock, lane, case):
        cut = load_cut(root, store, row["cut"], inventory["files"])
        payload = cut.payload
        require(cut.parent == row["parent"] == parent and dict(cut.execution_point) == row["point"] == point(clock)
                and cut.boundary == row["boundary"] and {k: payload[k] for k in ("step", "phase", "cursor")} == clock, "full state lineage/clock differs")
        component = cut.execution["component_pass"]
        require(component["id"] == "mdb-flux-functional-pass-v1" and component["lane"] == lane and component["case"] == case
                and component["source_cut"] == origins[case].fingerprint and component["verification_adapter_identity"] == contract["verification_adapter_identity"], "component/source adapter binding differs")
        require(row["fields"] == {k: describe(payload[k]) for k in ("image", "text", "latent", "context", "noise_prediction")}, "boundary field sidecar differs")
        return cut

    rows = {lane: {case: read(root / (lane + "-rows") / (case + ".json")) for case in CASES} for lane in ("forecast", "native")}
    require(plain(predicted["cases"]) == rows["forecast"], "sealed forecast rows changed")
    states_index, comparisons, first, finite = {}, {}, {}, []
    for case in CASES:
        comparisons[case], states_index[case], first[case] = [], {}, None
        for lane in ("forecast", "native"):
            require(len(rows[lane][case]) == length, "complete pass coverage differs")
            previous = origins[case]
            for number, row in enumerate(rows[lane][case]):
                require(row["number"] == number and row["before"] == before[number], "boundary sequence differs")
                store = "prediction-store" if lane == "forecast" else "native-store"
                cut = cut_row(store, row["state"], previous.fingerprint, after[number], lane, case)
                p, old = cut.payload, previous.payload
                require(all(describe(p[k]) == describe(origins[case].payload[k]) for k in CARRIED), "own carried closure refreshed")
                phase = before[number]["phase"]
                if phase != "scheduler":
                    require(describe(p["latent"]) == describe(old["latent"]), "latent changed before commit")
                if phase in ("joint", "single", "readout"):
                    require(describe(p["context"]) == describe(old["context"]), "own project context refreshed")
                if phase == "readout":
                    require(all(describe(p[k]) == describe(old[k]) for k in ("image", "text")), "readout changed stream state")
                require((p["noise_prediction"] is not None) == (after[number]["phase"] == "scheduler"), "noise field lifetime differs")
                require(all((p[k] is None) == (after[number]["phase"] == "project") for k in ("image", "text", "context")), "working-state reset differs")
                previous = cut
            states_index[case][lane] = [row["state"] for row in rows[lane][case]]
        for number in range(length):
            f = LocalStore(root / "prediction-store").load(rows["forecast"][case][number]["state"]["cut"])
            n = LocalStore(root / "native-store").load(rows["native"][case][number]["state"]["cut"])
            full = compare_states(f.payload, n.payload)
            fp, np_ = [read(root / rows[lane][case][number]["ports"]) for lane in ("forecast", "native")]
            require(fp["before"] == np_["before"] == before[number], "primitive occurrence clock differs")
            witnesses = witness_comparison(fp["ports"], np_["ports"], config.get("witness_values", 16))
            semantics = {key: ("committed-latent" if before[number]["phase"] == "scheduler" else "carried-origin-latent") if key == "latent"
                else "observed-readout-noise" if key == "noise_prediction" else "own-project-context-carried" if key.startswith("context.")
                else "carried-pre-readout-stream" if before[number]["phase"] == "readout" else "updated-stream" for key in full}
            row = dict(number=number, before=before[number], fields=full, field_semantics=semantics, primitive_witnesses=witnesses)
            comparisons[case].append(row)
            mismatches = [key for key, value in full.items() if not value["bitwise_exact"]]
            ports = [key for key, value in witnesses["ports"].items() if not value["full_descriptor_exact"] or not value["witness_bitwise_exact"]]
            if first[case] is None and (mismatches or ports or witnesses["unmatched_native_ports"]):
                first[case] = dict(number=number, before=before[number], fields=mismatches, primitive_ports=ports, unmatched_native_ports=witnesses["unmatched_native_ports"])
    for number in range(length):
        sham = {lane: fields(LocalStore(root / store).load(rows[lane]["sham"][number]["state"]["cut"]).payload)
                for lane, store in (("forecast", "prediction-store"), ("native", "native-store"))}
        panel = {}
        for case in CASES[1:]:
            value = {lane: fields(LocalStore(root / store).load(rows[lane][case][number]["state"]["cut"]).payload)
                     for lane, store in (("forecast", "prediction-store"), ("native", "native-store"))}
            panel[case] = {}
            for key in ("image", "text", "latent", "noise_prediction"):
                if key in value["native"]:
                    response = {lane: value[lane][key].astype(np.float64) - sham[lane][key] for lane in ("forecast", "native")}
                    panel[case][key] = {**metric(response["forecast"], response["native"]),
                        "native_signed_mean": float(response["native"].mean()), "predicted_signed_mean": float(response["forecast"].mean()),
                        "field_semantics": "committed-latent" if key == "latent" and before[number]["phase"] == "scheduler" else "carried-origin-latent" if key == "latent" else "native-working-field",
                        "scope": "Finite case-minus-sham response at this clock; no intrinsic gain or independent semantic replication."}
        finite.append(dict(number=number, before=before[number], responses=panel))
    observation = {name: load_forecast_tree(tx, read(tx / "observations" / (name + ".json"))["value"])
                   for name in ("native_targets", "observed_next_projects")}
    require(dict(observation["native_targets"]["final_cuts"]) == {case: rows["native"][case][-1]["state"]["cut"] for case in CASES}, "native transaction output join differs")
    source_files(Path(config["historical_root"]), config["historical"], config["historical_files"])
    historical = {}
    for case in ROLES:
        spec = config["historical"][case]
        old = load_cut(Path(config["historical_root"]), spec["store"], spec["cut"], config["historical_files"])
        own = LocalStore(root / "native-store").load(rows["native"][case][-1]["state"]["cut"])
        require((old.payload["step"], old.payload["phase"], old.payload["cursor"]) == (2, "project", 0), "historical comparison target differs")
        historical[case] = dict(source_cut=old.fingerprint, native_cut=own.fingerprint, latent=metric(numpy(own.payload["latent"]), numpy(old.payload["latent"])))
    consumers, consumer_index = {}, {}
    sidecars = read(root / "comparisons.json")
    require(plain(observation["observed_next_projects"]["consumers"]) == sidecars["consumers"]
            and plain(observation["observed_next_projects"]["historical"]) == sidecars["historical"], "observed consumer sidecar differs from closed transaction")
    for case in CASES:
        records = sidecars["consumers"][case]
        consumed, port_joins = {}, {}
        for origin, lane, store in (("predicted", "forecast", "prediction-store"), ("native_reference", "native", "native-store")):
            record = records[origin]
            parent = LocalStore(root / store).load(rows[lane][case][-1]["state"]["cut"])
            require(record["input_cut"] == parent.fingerprint and record["consumed_state"] == describe(parent.payload) and record["predicted"] is False,
                    "consumer silently refreshed/borrowed committed input")
            cut = cut_row("consumer-store", record["output"], parent.fingerprint, dict(step=2, phase="joint", cursor=0), "observed-consumer." + origin, case)
            require(all(describe(cut.payload[k]) == describe(parent.payload[k]) for k in (*CARRIED, "latent")), "next project altered committed input/carried state")
            raw_ports = read(root / "consumer-ports" / (case + "." + origin + ".json"))
            ports = {row["id"]: row for row in raw_ports}
            require(len(ports) == len(raw_ports), "duplicate consumer primitive occurrence")
            joins = {}
            def join(name, field, value):
                witness = ports[name]["fields"][field]
                desc = dict(tensor_descriptor(value), stride=list(value.stride()))
                require(witness["descriptor"] == desc, "actual consumer operand/output descriptor differs: " + name + "." + field)
                expected_values = numpy(value).astype(np.float32).reshape(-1)[witness["indices"]]
                require(expected_values.tobytes() == np.asarray(witness["values"], np.float32).tobytes(), "actual consumer coordinate witness differs")
                joins[name + "." + field] = dict(full_descriptor_exact=True, witness_bitwise_exact=True, descriptor=desc)
            parent_state, output_state = parent.payload, cut.payload
            join("x_embedder.0", "input.0", parent_state["latent"])
            join("context_embedder.0", "input.0", parent_state["conditioning"])
            join("pos_embed.0", "input.0", parent_state["img_ids"][0])
            join("pos_embed.1", "input.0", parent_state["txt_ids"][0])
            join("x_embedder.0", "output", output_state["image"])
            join("context_embedder.0", "output", output_state["text"])
            join("time_guidance_embed.timestep_embedder.linear_2.0", "output", output_state["context"]["temb"])
            for module, key in (("double_stream_modulation_img", "img_mod"), ("double_stream_modulation_txt", "txt_mod"), ("single_stream_modulation", "single_mod")):
                join(module + ".linear.0", "output", output_state["context"][key])
            text_rows = output_state["text"].shape[1]
            for index in (0, 1):
                rope = output_state["context"]["rope"][index]
                join("pos_embed.0", "output." + str(index), rope[text_rows:])
                join("pos_embed.1", "output." + str(index), rope[:text_rows])
            scheduled = np.asarray([float(parent_state["timesteps"][parent_state["step"]])], np.float32)
            timestep = rne_bf16(np.multiply(rne_bf16(np.divide(rne_bf16(scheduled), np.float32(1000), dtype=np.float32)), np.float32(1000), dtype=np.float32))
            time_port = ports["time_guidance_embed.time_proj.0"]["fields"]["input.0"]
            expected_sha = hashlib.sha256((timestep.view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest()
            require(time_port["descriptor"]["sha256"] == expected_sha and time_port["descriptor"]["shape"] == [1]
                    and time_port["descriptor"]["dtype"] == "torch.bfloat16"
                    and np.asarray(time_port["values"], np.float32).tobytes() == timestep.tobytes(), "native consumer timestep does not match own committed clock/schedule")
            joins["time_guidance_embed.time_proj.0.input.0"] = dict(full_descriptor_exact=True, expected_bf16_sha256=expected_sha, value=timestep.tolist())
            port_joins[origin] = joins
            consumed[origin] = cut.payload
        consumers[case] = dict(comparison=compare_states(consumed["predicted"], consumed["native_reference"]), predicted=False,
            actual_primitive_joins=port_joins,
            distinct_parent_cut_identities=records["predicted"]["input_cut"] != records["native_reference"]["input_cut"],
            scope="Two actual native next-project calls consuming their distinct committed-cut lineages; input values may be numerically equal. This output was not forecast.")
        consumer_index[case] = records
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    input_at = [i for i, row in enumerate(timeline) if row["kind"] == "inputs_globally_sealed"]
    cohort_at = [i for i, row in enumerate(timeline) if row["kind"] == "cohort_globally_sealed"]
    require(len(input_at) == len(cohort_at) == 1 and input_at[0] < cohort_at[0], "global seal chronology differs")
    prior, later = timeline[:cohort_at[0]], timeline[cohort_at[0] + 1:]
    require([row["kind"] for row in timeline[:input_at[0]]] == ["checkpoint_validation_started", "original_transformer_hydrated"]
            and timeline[0]["recipes"] == restoration["tensors"] and timeline[1]["parameters"] == restoration["tensors"]
            and timeline[1]["model_loads"] == 1, "initial hydration/model census differs")
    require(all(row["kind"] in ("checkpoint_validation_started", "original_transformer_hydrated", "inputs_globally_sealed", "boundary_completed") for row in prior), "native/future evidence before forecasts sealed")
    require([(row["case"], row["number"]) for row in prior if row["kind"] == "boundary_completed"] == [(case, i) for case in CASES for i in range(length)]
            and all(row["lane"] == "forecast" for row in prior if row["kind"] == "boundary_completed"), "all four functional passes did not precede target release")
    require([(row["case"], row["number"]) for row in later if row["kind"] == "boundary_completed"] == [(case, i) for case in CASES for i in range(length)]
            and all(row["lane"] == "native" for row in later if row["kind"] == "boundary_completed"), "native pass order differs")
    for row in timeline:
        if row["kind"] == "boundary_completed":
            lane, case, number = row["lane"], row["case"], row["number"]
            require(row["before"] == before[number] and row["cut"] == rows[lane][case][number]["state"]["cut"], "event/full boundary join differs")
    last_native = max(i for i, row in enumerate(timeline) if row["kind"] == "boundary_completed" and row["lane"] == "native")
    reads = [(i, row) for i, row in enumerate(timeline) if row["kind"] == "historical_cut_read"]
    require([row["case"] for _, row in reads] == list(ROLES) and all(i > last_native and row["native_transitions_completed"] == 4 * length for i, row in reads), "historical targets read before all forecasts/native passes")
    entries = [row for row in later if row["kind"] == "native_entry"]
    native_entries = [(row["case"], row["phase"], row["index"], row["module"]) for row in entries if row["lane"] == "native"]
    expected_entries = [(case, phase, index, module) for case in CASES for phase, index, module in
                        [("project", 0, "time_guidance_embed"), *[("joint", i, f"transformer_blocks.{i}") for i in range(joints)],
                         *[("single", i, f"single_transformer_blocks.{i}") for i in range(singles)]]]
    require(native_entries == expected_entries and [row["case"] for row in later if row["kind"] == "native_scheduler_entry"] == list(CASES), "native target/commit guards or block order differ")
    require([(row["case"], row["module"]) for row in entries if row["lane"] == "consumers"]
            == [(case + "." + origin, "time_guidance_embed") for case in CASES for origin in ("predicted", "native_reference")], "eight native consumer entries differ")
    result = dict(schema="mdb-functional-pass-audit-v1", job=summary["job_id"], native_request_sha256=expected_request, native_manifest_sha256=expected_manifest,
        worker_sha256=source["worker_sha256"], auditor_sha256=sha(__file__), source_provenance=source["source_provenance"], source_model_identity=config["expected_source_model_identity"],
        verification_adapter_identity=contract["verification_adapter_identity"], native_contract=contract, counts=expected, totals=summary["totals"],
        input_verification=input_verify, forecast_verification=forecast_verify, parameter_restoration=restoration, input_novelty=novelty,
        boundaries=comparisons, first_covered_bitwise_mismatch=first, signed_finite_responses=finite, historical=historical, consumers=consumers,
        scope="Four own-state functional graphs sharing original Torch primitives, all forecasts sealed before native targets; no independent kernel, new prompt or semantic claim.",
        guard_scope="Source-bound guards instrument every registered original module.forward and native scheduler entry; no OS sandbox or arbitrary-code noninterference claim.",
        new_model_calls=0, new_reference_forward_calls=0, new_recurrence_transitions=0,
        mechanics_status="authenticated-full-state-saved-evidence-audit", trend_status="observed", terminal_status="unassessed")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "state-index.json", states_index)
    save(output / "consumer-index.json", consumer_index)
    catalog, queries = EquationCatalog(output / "catalog/equations"), []
    final_arrays = {lane: {case: numpy(LocalStore(root / store).load(rows[lane][case][-1]["state"]["cut"]).payload["latent"]) for case in CASES}
                    for lane, store in (("forecast", "prediction-store"), ("native", "native-store"))}
    catalog_rows = [("endpoint." + case, origins[case].fingerprint, {"committed_latent": final_arrays["forecast"][case]},
                     {"committed_latent": final_arrays["native"][case]}) for case in CASES]
    catalog_rows += [("finite-response." + case, {"case": origins[case].fingerprint, "sham": origins["sham"].fingerprint},
                     {"finite_response": final_arrays["forecast"][case].astype(np.float64) - final_arrays["forecast"]["sham"]},
                     {"finite_response": final_arrays["native"][case].astype(np.float64) - final_arrays["native"]["sham"]}) for case in CASES[1:]]
    for occurrence, input_identity, predicted_fields, native_fields in catalog_rows:
        context = dict(source_model_identity=config["expected_source_model_identity"], verification_adapter_identity=contract["verification_adapter_identity"],
            input_cut=input_identity, occurrence=occurrence, clock=before[0], horizon=length,
            forecast_seal_sha256=summary["forecast_seal"]["seal_sha256"], parameter_manifest_sha256=config["parameter_manifest_sha256"], numerical=contract)
        item = catalog.record(law_id="law.flux.functional-complete-pass.v1", route_id="route.flux.project.joint.single.readout.flow-commit.v1", alias="cmp.flux.functional-pass." + occurrence,
            binding=dict(model_identity=config["expected_source_model_identity"], input_identity=input_identity, source_identity=expected_request,
                frontier=before[0], execution=contract, horizon=length), report=compare_equations(predicted_fields, native_fields),
            conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()], tested_context=context,
            readers=["next-native-project"], evidence=dict(audit_sha256=sha(output / "audit.json"), independent_orchestration=True,
                independent_kernel_arithmetic=False, downstream_project_predicted=False, dependent_case_count=4, quality_gate=False))
        exact, wrong = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to={**context, "clock": {**before[0], "step": 2}}))
        require(exact == 1 and wrong == 0, "exact catalog applicability differs")
        queries.append(dict(occurrence=occurrence, fingerprint=item["fingerprint"], exact=exact, wrong_clock=wrong))
    save(output / "catalog/provenance.json", dict(records=7, queries=queries, downstream_project_forecast_records=0,
        dependent_cases=4, family_induction=False, scope="Four complete-pass committed-latent equations; full signed finite responses are retained in audit, not independent replications."))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged source changed")
    root, start = Path(config["native_root"]), time.monotonic()
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    original = read(root / "plan.json")
    for name in ("source.json", "summary.json", "plan.json", "files.json", "comparisons.json", "response-accounts.json", "input-geometry.json", "native-contract.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script"):
        shutil.copyfile(root / Path(original[key]).name, output / Path(original[key]).name)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots={"native": str(root)}, manifests={"native": {name: row["sha256"] for name, row in inventory["files"].items()}},
        scope="Complete predicted/native states, sealed arrays and primitive witnesses retained near data; retrieve individually by checksum."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-functional-pass-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], source_job=result["job"],
        records=7, output_root=str(output), seconds=time.monotonic() - start, model_calls=0, reference_forward_calls=0, mechanics_status="verified-saved-evidence", terminal_status="unassessed"))
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
        result = launch["submit"](args.run, worker=Path(__file__)) if args.submit else launch["stage"](args.run, args.study, worker=Path(__file__), helpers=HELPERS,
            experiment="mdb-functional-pass-audit", operations=112,
            scope="Four sealed complete functional passes, full state/primitive witnesses, exact parameter restoration and eight observed next-project consumers; zero neural/reference recurrence.")
        print(json.dumps(result))
