"""Authenticate and audit saved Qwen cache-restoration futures without a model.

Only NumPy and framework-free public MDB readers are used. Numerical agreement,
historical restoration, and factorial effects remain measurements; the hard
checks protect the identity, declared intervention, and recurrence mechanics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import numpy as np

CORNERS = {"QQ": {"K": "Q", "V": "Q"}, "EQ": {"K": "E", "V": "Q"},
           "QE": {"K": "Q", "V": "E"}, "EE": {"K": "E", "V": "E"}}
LAYERS, LENGTH, STEPS, PENDING, WIDTH = 24, 13, 5, 425, 896
CACHE_FIELDS = tuple(f"{kind}{layer:02d}" for kind in "kv" for layer in range(LAYERS))
PORT_FIELDS = tuple(f"{kind}{layer:02d}" for kind in "ar" for layer in range(LAYERS))
STEP_FIELDS = {"logits", "input_ids", "token", *CACHE_FIELDS, *PORT_FIELDS}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    require(value is None or type(value) in (str, int, float, bool), "non-JSON sealed metadata")
    return value


def safe_path(root, name):
    require(isinstance(name, str) and name, "empty artifact path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name
            and all(part not in (".", "..") for part in relative.parts)
            and "\\" not in name, "unsafe artifact path: " + name)
    current = Path(root)
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), "symbolic artifact path: " + name)
    require(current.resolve().is_relative_to(Path(root).resolve()), "artifact escapes root: " + name)
    require(current.is_file(), "missing artifact: " + name)
    return current


def authenticate(root, name, record):
    path = safe_path(root, name)
    require(isinstance(record, dict) and type(record.get("bytes")) is int
            and record["bytes"] >= 0 and isinstance(record.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", record["sha256"]), "invalid artifact reference: " + name)
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
            "immutable member differs: " + name)
    return path


def inventory(root, manifest_sha, *, complete):
    root = Path(root)
    require(not root.is_symlink(), "artifact root is a symbolic link")
    path = safe_path(root, "files.json")
    require(sha(path) == manifest_sha, "published inventory differs")
    files = read(path)["files"]
    require(isinstance(files, dict) and "files.json" not in files, "invalid inventory census")
    if complete:
        actual = set()
        for member in root.rglob("*"):
            require(not member.is_symlink(), "unlisted symbolic artifact: " + str(member))
            if member.is_file():
                actual.add(member.relative_to(root).as_posix())
            else:
                require(member.is_dir(), "nonregular artifact: " + str(member))
        require(actual == set(files) | {"files.json"}, "all-file inventory census differs")
        for name, record in files.items():
            authenticate(root, name, record)
    return files


def exact(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def compare(reference, observed):
    require(reference.shape == observed.shape and reference.dtype == observed.dtype,
            "numeric geometry differs")
    require(reference.dtype.kind in "fiu" and np.isfinite(reference).all()
            and np.isfinite(observed).all(), "nonfinite or nonnumeric saved evidence")
    delta = observed.astype(np.float64) - reference.astype(np.float64)
    changed = delta != 0
    count = int(np.count_nonzero(changed))
    return dict(shape=list(reference.shape), dtype=str(reference.dtype), elements=int(reference.size),
                bit_equal=exact(reference, observed), changed_elements=count,
                max_abs=float(np.abs(delta).max(initial=0.)),
                rms=float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.,
                signed_mean=float(delta.mean()) if delta.size else 0.,
                first_numeric_difference=[int(i) for i in np.unravel_index(int(changed.argmax()), delta.shape)] if count else None,
                residual_direction="observed-minus-reference")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate NPZ fields")
        result = {name: archive[name].copy() for name in archive.files}
    for name, value in result.items():
        require(value.dtype.kind in "fiu" and np.isfinite(value).all(),
                "invalid saved array: " + name)
    return result


def validate_design(design):
    require(design.get("schema") == "mdb-qwen-cache-restoration-design-v1"
            and design.get("corners") == CORNERS, "factorial design differs")
    require(design.get("layers") == LAYERS and design.get("cache_length") == LENGTH
            and design.get("changed_layer_start") == 13
            and design.get("new_choices_per_corner") == STEPS
            and design.get("pending_token") == PENDING and design.get("physical_batch") == 1,
            "declared bounded geometry differs")
    require(design.get("new_forward_census") == dict(functional=20, native=20, model_loads=1, prefills=0)
            and design.get("scientific_gates") == [], "discovery execution scope differs")


def source_support(q, e):
    """Measure every source difference; refuse any undeclared cache footprint."""
    rows = []
    require(set(q) == set(e) and set(CACHE_FIELDS) <= set(q), "parent array census differs")
    require(exact(q["input_ids"], e["input_ids"]), "source parents do not share prompt tokens")
    for parent in (q, e):
        require(parent["input_ids"].shape == (1, LENGTH) and parent["input_ids"].dtype == np.int64
                and parent["token"].shape == (1,) and parent["token"].dtype == np.int64
                and int(parent["token"][0]) == PENDING, "parent pending token or input geometry differs")
        require(parent["logits"].dtype == np.float32 and parent["logits"].ndim == 3
                and parent["logits"].shape[:2] == (1, 1)
                and parent["logits"].shape[2] > PENDING
                and int(parent["logits"].argmax(axis=-1)[0, 0]) == PENDING,
                "parent full-logit decision differs")
    require(q["logits"].shape == e["logits"].shape, "parent vocabulary geometry differs")
    for field in CACHE_FIELDS:
        left, right = q[field], e[field]
        require(left.shape == right.shape == (1, 2, LENGTH, 64)
                and left.dtype == right.dtype == np.float32, "parent cache geometry differs: " + field)
        layer = int(field[1:])
        if layer < 13:
            require(exact(left, right), "source difference outside declared layers: " + field)
        else:
            require(exact(left[..., :12, :], right[..., :12, :]),
                    "source difference outside declared row12: " + field)
        report = compare(left, right)
        different_bits = left.view(np.uint32) != right.view(np.uint32)
        changed_rows = np.flatnonzero(np.any(different_bits, axis=(0, 1, 3))).tolist()
        report["numeric_changed_elements"] = report["changed_elements"]
        report["changed_elements"] = int(np.count_nonzero(different_bits))
        # The worker records support with the original FP32 arithmetic.
        report["max_abs"] = float(np.abs(left - right).max(initial=0.))
        rows.append(dict(field=field, layer=layer, changed_rows=changed_rows, **report))
    return rows


def corner_cache(q, e, corner):
    return {field: (e if CORNERS[corner][field[0].upper()] == "E" else q)[field].copy()
            for field in CACHE_FIELDS}


def validate_step(row, *, cache_length, vocab):
    require(set(row) == STEP_FIELDS, "retained numeric field census differs")
    require(row["logits"].shape == (1, 1, vocab) and row["logits"].dtype == np.float32,
            "full logits geometry differs")
    require(row["input_ids"].shape == (1, 1) and row["input_ids"].dtype == np.int64
            and row["token"].shape == (1,) and row["token"].dtype == np.int64,
            "token geometry differs")
    for field in CACHE_FIELDS:
        require(row[field].shape == (1, 2, cache_length, 64) and row[field].dtype == np.float32,
                "complete FP32 cache geometry differs: " + field)
    for field in PORT_FIELDS:
        require(row[field].shape == (1, 1, WIDTH) and row[field].dtype == np.float32,
                "all-layer consumed port geometry differs: " + field)


def validate_lane(rows, references, initial, vocab):
    require(len(rows) == len(references) == STEPS, "token horizon differs")
    previous, pending = initial, PENDING
    for step, (row, reference) in enumerate(zip(rows, references, strict=True)):
        validate_step(row, cache_length=LENGTH + step + 1, vocab=vocab)
        require(int(row["input_ids"][0, 0]) == pending, "own-token recurrence broken")
        token = int(row["token"][0])
        require(token == int(row["logits"][0, 0].argmax())
                and type(reference.get("token_id")) is int and reference["token_id"] == token,
                "saved greedy token or sealed token metadata differs")
        for field in CACHE_FIELDS:
            require(exact(row[field][..., :-1, :], previous[field]),
                    f"cache prefix mutated or corner landing differs: step{step}/{field}")
        previous, pending = row, token
    return dict(tokens=[int(row["token"][0]) for row in rows],
                consumed_tokens=[int(row["input_ids"][0, 0]) for row in rows],
                cache_lengths=[int(row["k00"].shape[-2]) for row in rows],
                final_pending_token=pending, final_pending_token_consumed=False,
                parent_prefix_bitwise_immutable=True, every_append_one_row=True)


def tensor_fingerprint(value):
    """Public TITAN content fingerprint convention, without importing Torch."""
    dtype = {np.dtype("float32"): "torch.float32", np.dtype("int64"): "torch.int64"}[value.dtype]
    return digest(dict(dtype=dtype, shape=list(value.shape),
                       bytes=hashlib.sha256(value.tobytes(order="C")).hexdigest()))


def verify_frame(frame, values):
    require(frame.get("schema") == "saturn-frame-v1" and set(frame["slots"]) == set(values),
            "typed Frame slot census differs")
    identity = {key: value for key, value in frame.items() if key != "frame_id"}
    require(set(identity) == {"schema", "parent_frame_id", "act", "implementation",
                             "program_fingerprint", "slots", "metadata"}
            and digest(identity) == frame["frame_id"], "typed Frame identity differs")
    for name, value in values.items():
        slot = frame["slots"][name]
        require(slot.get("name") == name and slot.get("fingerprint") == tensor_fingerprint(value)
                and slot.get("kind") and slot.get("handle"), "typed Frame content differs: " + name)
        abi = slot.get("abi", {})
        require(abi.get("shape") == list(value.shape)
                and abi.get("dtype") in (str(value.dtype), "torch." + str(value.dtype)),
                "typed Frame ABI geometry differs: " + name)
        fingerprint_abi = abi.get("saturn:fingerprint", {})
        require(not fingerprint_abi or (fingerprint_abi.get("durable") is True
                and fingerprint_abi.get("policy") == "content-sha256-v1"),
                "typed Frame does not declare durable content identity: " + name)
        expected_kind = ("pending-token" if name == "pending" else "cache-position" if name == "position"
                         else "post-rope-key" if name.removeprefix("e.").startswith("k") else "value-cache")
        require(slot["kind"] == expected_kind, "typed Frame state kind differs: " + name)


def verify_frames(frames, q, e):
    parent = frames["parent"]
    values = {field: q[field] for field in CACHE_FIELDS}
    values.update({"e." + field: e[field] for field in CACHE_FIELDS})
    for name, value in (("pending", PENDING), ("position", LENGTH)):
        shape = parent["slots"][name]["abi"]["shape"]
        require(isinstance(shape, list) and all(type(dim) is int and dim >= 1 for dim in shape)
                and np.prod(shape, dtype=np.int64) == 1, "typed scalar state geometry differs")
        values[name] = np.full(shape, value, dtype=np.int64)
    verify_frame(parent, values)
    require(set(frames["corners"]) == set(CORNERS), "typed corner census differs")
    reports = {}
    for corner, metadata in frames["corners"].items():
        before, after, act, receipt = (metadata[name] for name in ("before", "after", "act", "receipt"))
        require(before == parent, "typed Act does not start from common parent")
        changed = {field for field in CACHE_FIELDS if int(field[1:]) >= 13
                   and CORNERS[corner][field[0].upper()] == "E"}
        required_reads = set(CACHE_FIELDS) | {"e." + field for field in changed}
        require(act.get("schema") == "saturn-act-spec-v1"
                and act.get("identity") == act.get("name", "") + ":" + act.get("abi_version", "")
                and set(act["writes"]) == changed and len(act["writes"]) == len(changed)
                and not act["invalidates"] and required_reads <= set(act["reads"])
                and set(act["reads"]) <= required_reads | {"pending", "position"},
                "typed Act read/write footprint differs: " + corner)
        after_values = {**values, **corner_cache(q, e, corner)}
        verify_frame(after, after_values)
        require(after["parent_frame_id"] == before["frame_id"] and after["act"] == act["identity"]
                and receipt["input_frame_id"] == before["frame_id"]
                and receipt["output_frame_id"] == after["frame_id"]
                and receipt["act"] == act["identity"]
                and receipt["implementation"] == after["implementation"]
                and receipt["numerical_contract"] == act["numerical_contract"],
                "typed Act evidence linkage differs: " + corner)
        require(act["numerical_contract"] == "exact-source-row-replacement"
                and set(act["write_kinds"]) == set(act["write_abis"]) == changed
                and all(act["write_kinds"][field] == after["slots"][field]["kind"]
                        and act["write_abis"][field] == after["slots"][field]["abi"] for field in changed),
                "typed Act output contract differs: " + corner)
        for field in set(values) - changed:
            require(before["slots"][field] == after["slots"][field],
                    "typed Act changed undeclared slot: " + field)
        reports[corner] = dict(input_frame_id=before["frame_id"], output_frame_id=after["frame_id"],
                               writes=sorted(changed), exact_cache_landing=True)
    return reports


def verify_timeline(root, counts, verification):
    expected = dict(functional_calls=20, native_calls=20, denied_entries=0)
    require(all(counts.get(key) == value for key, value in expected.items()),
            "physical forward census or denied-entry census differs")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seals = [index for index, row in enumerate(timeline) if row["kind"] == "global_forecast_sealed"]
    require(len(seals) == 1, "global seal timeline census differs")
    seal_index = seals[0]
    require(timeline[seal_index]["counts"]["functional_calls"] == 20
            and timeline[seal_index].get("seal_sha256") == verification["seal_sha256"],
            "forecast seal does not bind complete functional futures")
    previous = dict(functional_calls=0, native_calls=0, denied_entries=0)
    for index, row in enumerate(timeline):
        current = row["counts"]
        require(all(type(current.get(key)) is int and previous[key] <= current[key] <= expected[key]
                    for key in expected), "timeline counters differ")
        if index <= seal_index:
            require(current["native_calls"] == current["denied_entries"] == 0
                    and not row["kind"].startswith("native_"), "native entry before global seal")
        else:
            require(current["functional_calls"] == 20, "functional prediction continued after seal")
        previous = current
    completions = [row for row in timeline[seal_index + 1:] if row["kind"] == "native_future_complete"]
    require(len(completions) == 4 and [row["counts"]["native_calls"] for row in completions] == [5, 10, 15, 20]
            and all(previous[key] == value for key, value in expected.items()),
            "native future completion census differs")
    seal = read(root / "transaction/seal.json")
    require(seal["pre_release_work"]["native_calls"] == 0
            and seal["pre_release_work"]["denied_attempts"] == 0
            and seal["pre_release_work"]["captures"] == 0 and not seal["violations"],
            "transaction had native work or a violation before seal")
    events = [read(path) for path in sorted((root / "transaction/events").glob("*.json"))]
    global_events = [i for i, row in enumerate(events) if row["kind"] == "globally_sealed"]
    require(len(global_events) == 1 and not any(row["kind"] in ("failed", "violation") for row in events)
            and all(row["work"]["native_calls"] == row["work"]["denied_attempts"] == 0
                    and row["work"]["executions"] == row["work"]["captures"] == 0
                    for row in events[:global_events[0] + 1]), "transaction native-exclusion events differ")
    return dict(seal_row=seal_index, timeline_rows=len(timeline), native_completions=4,
                native_calls_before_seal=0, denied_entries=0,
                guard_scope="trusted worker bound-model module hooks; not an OS sandbox")


def pair_trajectory(reference, observed):
    result, first = [], {name: None for name in ("logits", "attention", "residual", "key_cache",
                                                 "value_cache", "key_new_row", "value_new_row")}
    for step, (left, right) in enumerate(zip(reference, observed, strict=True)):
        logits = compare(left["logits"], right["logits"])
        if not logits["bit_equal"] and first["logits"] is None:
            first["logits"] = dict(step=step, **logits)
        layers = []
        for layer in range(LAYERS):
            fields = {}
            for group, kind, new_row in (("attention", "a", False), ("residual", "r", False),
                                         ("key_cache", "k", False), ("value_cache", "v", False),
                                         ("key_new_row", "k", True), ("value_new_row", "v", True)):
                a, b = left[f"{kind}{layer:02d}"], right[f"{kind}{layer:02d}"]
                report = compare(a[..., -1:, :] if new_row else a, b[..., -1:, :] if new_row else b)
                fields[group] = report
                if not report["bit_equal"] and first[group] is None:
                    first[group] = dict(step=step, layer=layer, **report)
            layers.append(dict(layer=layer, **fields))
        result.append(dict(step=step, logits=logits, layers=layers))
    return dict(steps=result, first_bitwise_divergence=first,
                first_token_divergence=next((step for step, (a, b) in enumerate(zip(reference, observed, strict=True))
                    if int(a["token"][0]) != int(b["token"][0])), None))


def audit(source, manifest_sha256, output):
    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(source), Path(output)
    require(not output.resolve().is_relative_to(root.resolve()), "audit output must be outside source inventory")
    files = inventory(root, manifest_sha256, complete=True)
    design, request, runtime = (read(root / name) for name in ("design.json", "request.json", "runtime.json"))
    validate_design(design)
    verification = verify_transaction(root / "transaction",
        expected_seal_sha256=files["transaction/seal.json"]["sha256"])
    declarations = read(root / "transaction/declarations.json")
    require(declarations["model_identity"] == design["weights_sha256"]
            and declarations["source_identity"] == request["source_hashes"]
            and declarations["input_identity"] == digest(design["source_files"]),
            "transaction model/input/source binding differs")
    predicted, native = (read(root / name) for name in ("forecast-index.json", "native-index.json"))
    sealed_record = read(root / "transaction/forecasts/all_futures.json")
    sealed = plain(load_forecast_tree(root / "transaction", sealed_record["predictions"]))
    require(sealed == predicted and set(predicted) == set(native) == set(CORNERS),
            "sealed prediction or corner census differs")
    observation_record = read(root / "transaction/observations/restored_native_futures.json")
    require(plain(load_forecast_tree(root / "transaction", observation_record["value"])) == native,
            "transaction native observation index differs")
    timeline = verify_timeline(root, read(root / "counts.json"), verification)

    old_root = Path(design["parent_native_root"])
    old_files = inventory(old_root, design["parent_manifest_sha256"], complete=False)
    old_index = read(authenticate(old_root, "native-index.json", old_files["native-index.json"]))
    old_runtime = read(authenticate(old_root, "runtime.json", old_files["runtime.json"]))
    require(sha(root / "parent-files.json") == design["parent_manifest_sha256"],
            "copied source manifest differs")
    require(sha(root / "parent-parameter-artifacts.json") == old_files["parameter-artifacts.json"]["sha256"],
            "source parameter provenance differs")
    for key in ("native_source_sha256", "math_source_sha256", "torch", "transformers", "dtype", "device", "attention"):
        require(runtime.get(key) == old_runtime.get(key) and runtime.get(key) is not None,
                "qualified runtime profile differs: " + key)
    require(runtime["dtype"] in ("torch.float32", "float32") and runtime["device"] == "cpu"
            and runtime["attention"] == "eager", "CPU FP32 eager profile differs")
    for key, design_key in (("checkpoint_sha256", "checkpoint_file_sha256"),
                            ("weights_identity_sha256", "weights_sha256")):
        require(runtime.get(key) == old_runtime.get(key) == design[design_key],
                "qualified checkpoint identity differs: " + key)

    parents, historical = {}, {}
    for side, action in (("q", "recipient"), ("e", "value")):
        path = f"native/{design['context']}/{action}/00.npz"
        record = old_index[f"{design['context']}/{action}"]["steps"][0]
        require(record["path"] == path and all(record[key] == old_files[path][key] for key in ("sha256", "bytes")),
                "source parent index binding differs")
        authenticate(old_root, path, old_files[path])
        copied = root / f"parents/{side}.npz"
        require(sha(copied) == old_files[path]["sha256"] and copied.stat().st_size == old_files[path]["bytes"],
                "source parent snapshot differs")
        parents[side] = load_arrays(copied)
        rows = []
        for step, reference in enumerate(old_index[f"{design['context']}/{action}"]["steps"][1:STEPS + 1], 1):
            expected = f"native/{design['context']}/{action}/{step:02d}.npz"
            require(reference["path"] == expected
                    and all(reference[key] == old_files[expected][key] for key in ("sha256", "bytes")),
                    "historical continuation index differs")
            rows.append(load_arrays(authenticate(old_root, expected, old_files[expected])))
        require(len(rows) == STEPS, "historical continuation horizon differs")
        historical["QQ" if side == "q" else "EE"] = rows
    frontier_record = read(root / "transaction/frontiers/observed_cache.json")
    frontier = load_forecast_tree(root / "transaction", frontier_record["value"])
    require(set(frontier) == {"Q", "E", "pending", "cache_position", "parameter_manifest_sha256"}
            and frontier["pending"] == PENDING and frontier["cache_position"] == LENGTH
            and frontier["parameter_manifest_sha256"] == sha(root / "parameter-artifacts.json")
            and sha(root / "parameter-artifacts.json") == old_files["parameter-artifacts.json"]["sha256"],
            "sealed observed frontier identity differs")
    for side, parent_side in (("Q", "q"), ("E", "e")):
        require(set(frontier[side]) == set(CACHE_FIELDS)
                and all(exact(frontier[side][field], parents[parent_side][field]) for field in CACHE_FIELDS),
                "sealed observed frontier cache differs: " + side)
    support = source_support(parents["q"], parents["e"])
    declared_support = read(root / "support.json")
    require(declared_support.get("cache_length") == LENGTH
            and declared_support.get("first_layer") == 13
            and declared_support.get("declared_support_verified") is True,
            "declared support metadata differs")
    declared_rows = {row["field"]: row for row in declared_support["rows"]}
    require(len(declared_support["rows"]) == len(declared_rows) == len(CACHE_FIELDS)
            and set(declared_rows) == set(CACHE_FIELDS), "declared support field census differs")
    for row in support:
        require(all(declared_rows[row["field"]].get(key) == row[key]
                    for key in ("layer", "changed_elements", "max_abs", "changed_rows")),
                "declared source support measurement differs: " + row["field"])
    frames = verify_frames(read(root / "frames.json"), parents["q"], parents["e"])

    lanes, cases, comparisons, equation_reports, equation_records = {}, {}, [], {}, {}
    catalog = EquationCatalog(output / "equations")
    vocab = parents["q"]["logits"].shape[-1]
    retained_paths = set()
    for corner in CORNERS:
        corner_lanes, case = {}, {}
        for lane, index in (("forecast", predicted), ("native", native)):
            require(isinstance(index[corner].get("text"), str), "missing retained continuation text")
            references = index[corner]["steps"]
            require(len(references) == STEPS, "new continuation horizon differs")
            rows = []
            for step, reference in enumerate(references):
                expected = f"{lane}/{corner}/{step:02d}.npz"
                require(reference["path"] == expected and expected not in retained_paths
                        and all(reference[key] == files[expected][key] for key in ("sha256", "bytes")),
                        "step index/inventory binding differs")
                retained_paths.add(expected)
                rows.append(load_arrays(authenticate(root, expected, reference)))
                if "fields" in reference:
                    require(reference["fields"] == sorted(rows[-1]), "sealed field census differs")
            case[lane] = {**validate_lane(rows, references, corner_cache(parents["q"], parents["e"], corner), vocab),
                          "text": index[corner]["text"]}
            corner_lanes[lane] = rows
        lanes[corner], cases[corner] = corner_lanes, case
        for step, (forecast, actual) in enumerate(zip(corner_lanes["forecast"], corner_lanes["native"], strict=True)):
            comparisons.extend(dict(corner=corner, step=step, field=field, **compare(forecast[field], actual[field]))
                               for field in sorted(STEP_FIELDS))
        logits = {lane: np.concatenate([row["logits"] for row in rows], axis=1)
                  for lane, rows in corner_lanes.items()}
        equation_reports[corner] = compare_equations({"logits": logits["forecast"]}, {"logits": logits["native"]},
            decision_rules={"logits": "argmax"}, max_elements=int(logits["forecast"].size))
        record = catalog.record(law_id="qwen.cached-greedy-functional-recurrence.v1",
            route_id="cache-restoration/" + corner,
            binding=dict(model_identity=design["weights_sha256"], execution=runtime,
                input_identity={"parent_manifest_sha256": design["parent_manifest_sha256"], "corner": corner, "pending_token": PENDING},
                frontier={"cache_length": LENGTH, "K": CORNERS[corner]["K"], "V": CORNERS[corner]["V"]},
                horizon={"choices": STEPS, "final_cache_length": 18, "last_choice_pending": True},
                source_identity={"manifest_sha256": manifest_sha256, "forecast_seal_sha256": verification["seal_sha256"]}),
            report=equation_reports[corner], tested_context={"context": design["context"], "corner": corner},
            readers=["full-logit-argmax", "z271-z198"],
            evidence={"source_root": str(root), "forecast_steps": predicted[corner]["steps"], "native_steps": native[corner]["steps"]})
        equation_records[corner] = record["fingerprint"]

    require(retained_paths == {name for name in files if name.startswith(("forecast/", "native/"))},
            "unindexed retained continuation files")
    historical_comparisons = {}
    for corner, old_rows in historical.items():
        reports = []
        for step, (old, new) in enumerate(zip(old_rows, lanes[corner]["native"], strict=True)):
            common = {"logits", "input_ids", "token", *CACHE_FIELDS}
            require(set(old) == common, "historical comparison field census differs")
            reports.append(dict(new_step=step, old_step=step + 1,
                fields={name: compare(old[name], new[name]) for name in sorted(common)}))
        historical_comparisons[corner] = dict(steps=reports,
            all_common_arrays_bit_equal=all(value["bit_equal"] for row in reports for value in row["fields"].values()),
            scope="post-execution saved-only comparison; old traces do not contain all-layer a/r ports")

    contrasts, interactions, relative = {}, {}, {}
    for corner in CORNERS:
        contrasts[corner] = {lane: [float(row["logits"][0, 0, 271]) - float(row["logits"][0, 0, 198])
                                  for row in lanes[corner][lane]] for lane in ("forecast", "native")}
        relative[corner] = {reference: pair_trajectory(lanes[reference]["native"], lanes[corner]["native"])
                            for reference in ("QQ", "EE")}
    for lane in ("forecast", "native"):
        rows = []
        for step in range(STEPS):
            values = {corner: lanes[corner][lane][step]["logits"].astype(np.float64) for corner in CORNERS}
            mixed = values["EE"] - values["EQ"] - values["QE"] + values["QQ"]
            rows.append(dict(step=step, c=contrasts["EE"][lane][step] - contrasts["EQ"][lane][step]
                             - contrasts["QE"][lane][step] + contrasts["QQ"][lane][step],
                             logits=compare(np.zeros_like(mixed), mixed)))
        interactions[lane] = rows
    report = dict(schema="mdb-qwen-cache-restoration-saved-audit-v1", mechanics_status="verified",
        instrument_status="executed-on-all-declared-fields", trend_status="measurements-retained",
        terminal_status="unassessed", model_forwards=0, model_loads=0,
        source_root=str(root), source_manifest_sha256=manifest_sha256, request_file_sha256=sha(root / "request.json"),
        request_sha256=request.get("request_sha256"), design_sha256=sha(root / "design.json"),
        parent_native_root=str(old_root), parent_manifest_sha256=design["parent_manifest_sha256"],
        authenticated_files=len(files), authenticated_bytes=sum(row["bytes"] for row in files.values()),
        transaction_verification=verification, timeline_verification=timeline, parent_cache_support=support,
        typed_frame_verification=frames, cases=cases, comparisons=comparisons,
        compared_array_pairs=len(comparisons), compared_elements=sum(row["elements"] for row in comparisons),
        array_bit_misses=sum(not row["bit_equal"] for row in comparisons),
        max_logit_error=max(row["max_abs"] for row in comparisons if row["field"] == "logits"),
        max_cache_error=max(row["max_abs"] for row in comparisons if row["field"] in CACHE_FIELDS),
        max_port_error=max(row["max_abs"] for row in comparisons if row["field"] in PORT_FIELDS),
        token_matches=sum(cases[corner]["forecast"]["tokens"][step] == cases[corner]["native"]["tokens"][step]
                          for corner in CORNERS for step in range(STEPS)), token_predictions=20,
        equation_reports=equation_reports, equation_catalog_records=equation_records,
        historical_restoration=historical_comparisons, contrast_definition="c = z271 - z198",
        contrasts=contrasts, first_c={corner: {lane: values[0] for lane, values in rows.items()}
                                     for corner, rows in contrasts.items()},
        factorial_definition="EE - EQ - QE + QQ; first factor K source, second factor V source",
        factorial_interactions=interactions, native_differences_from_QQ_and_EE=relative,
        limitations=["One observed development cache pair; no fresh prompt or unseen-state generalization claim",
            "Saved-only integrity binds retained bytes and declared guards; no OS sandbox attestation",
            "Historical QQ/EE full-logit and cache equality are observations, with no scientific pass/fail threshold",
            "All four branches use private own-token recurrence; later differences can include token feedback",
            "Five cached choices consume positions13..17; cache length18; fifth choice remains pending",
            "Natural-Q QQ restorer is distinct from the older one-query QQ control",
            "Numerical recurrence retains original parameters and kernels; no compression or faster-inference claim"])
    save(output / "audit.json", report)
    print("MDB_QWEN_CACHE_AUDIT_READY " + json.dumps(dict(output_root=str(output),
        audit_sha256=sha(output / "audit.json")), sort_keys=True), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    audit(args.source, args.manifest_sha256, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
