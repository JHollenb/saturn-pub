"""Saved-array accounts for the bounded block0 matched-operand microscope.

No model, tokenizer, attention kernel or learned recurrence is run here. Shared
primitive implementations and passive profiler events do not establish an
independent kernel, factorial effect, semantic reading or future-image result.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

WORKER_SHA256 = "70e52a6624947029b906e0a02f2f36d1a165b321a40c9e42f665224a3710361b"
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
HELPER_PINS = {
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
}
CASES = ("short", "long")
ARMS = ("A.explicit-repeated", "B.causal-repeated", "C.causal-grouped")
PORTS = ("H0", "normalized", "q_raw", "k_raw", "v_raw", "q_norm", "k_norm", "q", "k", "v", "head",
         "projection", "after_attention", "mlp_input", "mlp", "H1")
TAIL = ("head", "projection", "after_attention", "mlp_input", "mlp", "H1")
HEAD_PORTS = {"q_norm", "k_norm", "q", "k", "v", "head"}
ALL_PORTS = set(PORTS) | {"cos", "sin", "gate", "up", "activation"}
TAIL_PORTS = set(TAIL) | {"gate", "up", "activation"}
PARAMETERS = {"model.embed_tokens.weight"} | {"model.layers.0." + name for name in (
    "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def member(root, name):
    path, root = Path(name), Path(root)
    require(not path.is_absolute() and path.parts and ".." not in path.parts, "unsafe evidence member")
    current = root
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), "evidence symlink forbidden")
    require(current.resolve().is_relative_to(root.resolve()) and current.is_file(), "missing evidence member")
    return current


def array(root, record):
    """Validate stored bits, public descriptor, and separately retained layout."""
    import numpy as np

    path = member(root, record["path"])
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "array file checksum differs")
    bits = np.load(path, allow_pickle=False)
    require(list(bits.shape) == record["shape"] and str(bits.dtype) == record.get("stored_dtype", record.get("dtype")),
        "array stored shape/dtype differs")
    raw = hashlib.sha256(bits.tobytes(order="C")).hexdigest()
    if record.get("encoding") == "bf16-bits-v1":
        require(bits.dtype == np.dtype("uint16"), "BF16 uint16 archive required")
        value = (bits.astype(np.uint32) << 16).view(np.float32)
        logical_dtype = "torch.bfloat16"
    else:
        value = bits
        logical_dtype = "torch." + str(bits.dtype)
    require(np.isfinite(value).all(), "nonfinite saved field")
    if "raw_sha256" in record:
        require(raw == record["raw_sha256"] and logical_dtype == record["dtype"], "array raw identity differs")
        descriptor = record["descriptor"]
        require(descriptor["shape"] == record["shape"] and descriptor["dtype"] == logical_dtype
            and descriptor["sha256"] == raw and descriptor["preview"] == value.reshape(-1)[:8].tolist(),
            "public full descriptor differs")
        require(len(record["stride"]) == value.ndim and all(type(item) is int and item >= 0 for item in record["stride"]),
            "capture stride annotation differs")
    elif "content_sha256" in record:
        require(raw == record["content_sha256"], "original archived payload differs")
    return value


def metric(left, right):
    import numpy as np

    require(left.shape == right.shape and left.size > 0, "comparison geometry differs")
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite comparison")
    delta = a - b
    l2, r2, d2 = (float(np.sum(value * value, dtype=np.float64)) for value in (a, b, delta))
    changed = np.flatnonzero(delta.reshape(-1))
    return dict(orientation="left-minus-right", shape=list(a.shape), elements=int(a.size), rms=(d2 / a.size) ** .5,
        reference_rms=(r2 / a.size) ** .5, relative_rms=None if r2 == 0 else (d2 / r2) ** .5,
        signed_mean=float(delta.mean()), maxabs=float(np.max(np.abs(delta))), changed_values=int(changed.size),
        bitwise_exact=left.dtype == right.dtype and left.tobytes() == right.tobytes(),
        cosine=None if l2 * r2 == 0 else float(np.sum(a * b, dtype=np.float64)) / (l2 * r2) ** .5,
        first_nonzero_flat=None if not changed.size else int(changed[0]), worst_flat=int(np.argmax(np.abs(delta))))


def prefix(value, name, rows):
    slices = [slice(None)] * value.ndim
    axis = 2 if name in HEAD_PORTS else 1
    require(0 < rows <= value.shape[axis], "shared-prefix extent differs")
    slices[axis] = slice(0, rows)
    return value[tuple(slices)]


def vector_account(a, b, observed):
    import numpy as np

    x, y, z = (np.asarray(value, np.float64).reshape(-1) for value in (a, b, observed))
    energy = lambda value: float(np.sum(value * value, dtype=np.float64))
    dot = float(np.sum(x * y, dtype=np.float64))
    first, second, historic, total = (energy(value) for value in (x, y, z, x + y))
    return dict(first_energy=first, second_energy=second, sum_energy=total, cross_dot=dot, signed_cross_term=2 * dot,
        energy_identity_residual=total - first - second - 2 * dot,
        first_vs_historical_cosine=None if first * historic == 0 else float(np.sum(x * z)) / (first * historic) ** .5,
        second_vs_historical_cosine=None if second * historic == 0 else float(np.sum(y * z)) / (second * historic) ** .5,
        historical_energy=historic, coordinate_sum_elements=int(x.size),
        scope="Order-conditioned vector telescope and alignment, not factorial interaction, gain or causal shares.")


def repeat(value, heads):
    import numpy as np

    require(heads % value.shape[1] == 0, "KV group geometry differs")
    return np.repeat(value, heads // value.shape[1], axis=1)


def same(left, right, message):
    require(left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(), message)


def authenticate(root, expected_manifest, expected_request):
    root = Path(root)
    require(sha(root / "files.json") == expected_manifest, "wrapper manifest checksum differs")
    inventory = read(root / "files.json")
    require(inventory["schema"] == "mdb-image-tail-files-v1" and inventory["request_sha256"] == expected_request,
        "wrapper schema/request differs")
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != root / "files.json"}
    require(files == set(inventory["files"]) and not any(path.is_symlink() for path in root.rglob("*")),
        "complete wrapper member inventory differs")
    for name, row in inventory["files"].items():
        path = member(root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "retained member differs: " + name)
    require(sum(row["bytes"] for row in inventory["files"].values()) == inventory["expanded_bytes"], "expanded byte count differs")
    plan, request, source, summary = (read(root / name) for name in ("plan.json", "request.json", "source.json", "summary.json"))
    require(identity(request) == request["request_sha256"] == expected_request, "standard MDB request identity differs")
    require(all(value["request_sha256"] == expected_request and value["job_id"] == inventory["job_id"] for value in (source, summary)),
        "actual job/request joins differ")
    require(plan["request_sha256"] == expected_request and all(plan[key] == value for key, value in request["config"].items()),
        "effective plan differs from executed request")
    require("job_id" not in plan or plan["job_id"] == inventory["job_id"], "optional plan job identity differs")
    require(plan["schema"] == "mdb-qwen-block0-attention-plan-v1" and set(plan["cases"]) == set(CASES)
        and plan["fixed_bank_case"] == "long", "bounded microscope specimen differs")
    require(request["source_hashes"]["qwen_block0_attention_microscope.py"] == source["worker_sha256"] == WORKER_SHA256,
        "executed microscope worker differs")
    require(source["helper_sources"] == HELPER_PINS and all(request["source_hashes"][key] == value for key, value in HELPER_PINS.items()),
        "frozen arithmetic helper source differs")
    require(source["input_files"] == plan["input_files"] and source["source_provenance"] == plan.get("source_provenance", {}),
        "original selected source closure differs")
    if plan.get("profile") != "tiny-cpu":
        require(source["source_provenance"] == dict(job_id="job-0b6c4340d5e6",
            request_sha256="783e8dc0c0a731c458eed739c5d8b68c932bf622ef5f426d7f484596a8c011dd",
            manifest_sha256="89dafe939e0143596edf88b42e073f97cc5c945c893b5b8cf9897eff809c7154"), "accepted original paired source differs")
    for name, row in source["native_sources"].items():
        require(row["sha256"] == plan["native_sources"][name], "actual installed native source differs")
    metadata = {}
    for key in ("qwen_config", "qwen_manifest", "retrieval_script"):
        path = member(root, Path(plan[key]).name)
        require(sha(path) == plan[key + "_sha256"], "copied standalone source metadata differs")
        metadata[key] = path
    require(sha(metadata["retrieval_script"]) == RETRIEVER_SHA256, "standalone parameter retriever differs")
    actual = summary["actual"]
    require(actual["native_block_calls"] == actual["native_rotary_calls"] == 2 and actual["sdpa_calls"] == 7
        and actual["source_parameter_reads"] == 12, "observed finite census differs")
    require(source["recipe_reads"] == source["recipe_payload_reads"] == actual["source_parameter_reads"]
        and source["selected_payload_bytes"] == actual["selected_payload_bytes"], "actual recipe/byte census joins differ")
    require(summary["contract"]["total_sdpa_calls"] == 7 and summary["contract"]["full_encoder_calls"] == 0
        and summary["contract"]["tokenizer_calls"] == 0 and summary["contract"]["profiler_extra_forwards"] == 0,
        "declared finite component scope differs")
    return plan, source, summary, inventory, metadata


def parameters(metadata, source, root):
    """Stream selected original archive bytes without materializing a weight bank."""
    import mdb.parameter_artifacts as api

    require(sha(api.__file__) == RETRIEVER_SHA256, "installed parameter restoration source differs")
    manifest = api.validate_manifest(read(metadata["qwen_manifest"]))
    wanted = source["selected_recipe_names"]
    before = source["parameter_identity_before"]
    require(before == source["parameter_identity_after"] and set(before) == set(wanted) == PARAMETERS and len(wanted) == 12,
        "observed final resident/selected parameter identity differs")
    rows = [row for row in manifest["artifacts"] if row["source"]["tensor"] in wanted]
    require(len(rows) == len(wanted), "selected original recipe set differs")
    selected = api.validate_manifest({**manifest, "artifacts": rows})
    result, total = {}, 0
    for row, stream, size in api._source_entries(selected, root):
        raw, archive = hashlib.sha256(), hashlib.sha256(base64.b64decode(row["npy_prefix_b64"], validate=True))
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, 8 << 20))
            require(bool(chunk), "selected original parameter truncated")
            raw.update(chunk)
            archive.update(chunk)
            remaining -= len(chunk)
        name = row["source"]["tensor"]
        require(raw.hexdigest() == row["source"]["payload_sha256"] == before[name]["sha256"]
            and archive.hexdigest() == row["sha256"] and before[name]["shape"] == row["source"]["shape"]
            and before[name]["dtype"] == "torch.bfloat16", "selected original parameter restoration differs")
        result[name] = dict(payload_sha256=raw.hexdigest(), archive_sha256=archive.hexdigest(), payload_bytes=size)
        total += size
    require(total == source["selected_payload_bytes"], "actual selected parameter byte ledger differs")
    return dict(tensors=len(result), payload_bytes=total, rows=result, full_manifest_tensors=len(manifest["artifacts"]),
        source_file_sha256={row["source"]["file"]: row["source"]["file_sha256"] for row in rows},
        scope="Original file SHA/header/offsets and selected12 payload+NPY hashes; no persistent weights or neural calls.")


def original_sources(root, plan, cases, input_root):
    import numpy as np

    reports, operands = {}, {}
    provenance = plan["source_provenance"]
    require(sha(Path(input_root) / "files.json") == provenance["manifest_sha256"], "original source wrapper checksum differs")
    original_inventory = read(Path(input_root) / "files.json")
    require(original_inventory["schema"] == "mdb-image-tail-files-v1" and original_inventory["job_id"] == provenance["job_id"]
        and original_inventory["request_sha256"] == provenance["request_sha256"], "original source job/request differs")
    copied = {path.relative_to(input_root).as_posix() for path in Path(input_root).rglob("*") if path.is_file()}
    require(copied == set(plan["input_files"]) | {"files.json"}, "selected original input closure member set differs")
    for name, row in plan["input_files"].items():
        path = member(input_root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "selected original source file differs")
        require(original_inventory["files"][name] == row, "selected original input is outside original wrapper")
    for case in CASES:
        reports[case] = {}
        require(cases[case]["source_sidecars"] == plan["cases"][case], "actual original sidecar declaration differs")
        for lane in ("forecast", "native"):
            name = plan["cases"][case][lane + "_qwen"]
            require(name in plan["input_files"], "undeclared original sidecar")
            sidecar = read(member(input_root, name))
            base = Path(input_root) / "cases" / case
            for record in [sidecar["input_ids"], sidecar["attention_mask"], *sidecar["hidden_states"][:2]]:
                require("cases/" + case + "/" + record["path"] in plan["input_files"], "undeclared original source operand")
            ids, mask = (array(base, sidecar[key]) for key in ("input_ids", "attention_mask"))
            require(ids.shape == mask.shape and ids.shape[0] == 1, "original ID/mask shape differs")
            if lane == "native":
                operands[case] = dict(ids=ids, mask=mask)
            else:
                # Both original execution lanes remain separately observed controls.
                reports[case]["original_forecast_id_mask"] = dict(ids_shape=list(ids.shape), attended=int(mask.sum()))
            reports[case][lane] = {
                "H0": metric(array(root, cases[case]["functional"]["H0"]), array(base, sidecar["hidden_states"][0])),
                "H1": metric(array(root, cases[case]["native"]["H1"]), array(base, sidecar["hidden_states"][1]))}
    rows = plan["shared_prefix_rows"]
    same(operands["short"]["ids"][:, :rows], operands["long"]["ids"][:, :rows], "original common token prefix differs")
    for case in CASES:
        mask = operands[case]["mask"]
        require(np.all((mask == 0) | (mask == 1)) and np.all(mask[:, :rows] == 1), "original shared-prefix binary attention mask differs")
        require(cases[case]["attended_rows"] == int(mask.sum()) and cases[case]["retained_rows"] == int(mask.size), "actual retained/attended mask census differs")
        captured = cases[case]["original_mask"]
        if np.all(mask):
            require(captured is None, "unpadded original mask dispatch differs")
        else:
            require(captured is not None, "padded original mask missing")
            allowed = np.tril(np.ones((1, 1, mask.shape[1], mask.shape[1]), dtype=bool)) & mask[:, None, None, :].astype(bool)
            same(array(root, captured), allowed, "actual causal-padding mask differs from original saved operand")
    cross_case = {}
    for lane in ("forecast", "native"):
        cross_case[lane] = {}
        original = {case: read(member(input_root, plan["cases"][case][lane + "_qwen"])) for case in CASES}
        for index, name in ((0, "H0"), (1, "H1")):
            short, long = (array(Path(input_root) / "cases" / case, original[case]["hidden_states"][index]) for case in CASES)
            delta = long.astype(np.float64) - short.astype(np.float64)
            response_errors = {}
            for current_lane in ("functional", "native"):
                current = array(root, cases["long"][current_lane][name]).astype(np.float64) - array(root, cases["short"][current_lane][name]).astype(np.float64)
                response_errors[current_lane] = dict(full=metric(current, delta), shared_prefix=metric(prefix(current, name, rows), prefix(delta, name, rows)))
            cross_case[lane][name] = dict(orientation="long-minus-short", arithmetic="float64(long)-float64(short)",
                full=metric(long, short), shared_prefix=metric(prefix(long, name, rows), prefix(short, name, rows)),
                current_response_vs_original=response_errors,
                scope="Original historical source values; full fields include differing padding/content, with causal bridge restricted to shared prefix.")
    return reports, operands, cross_case


def receipts(root, cases, triangle, rows, model_config):
    import numpy as np

    labels = ["functional.short", "functional.long", *ARMS, "native.short", "native.long"]
    require(len(rows) == 7 and [row["ordinal"] for row in rows] == list(range(7))
        and [row["label"] for row in rows] == labels, "all seven actual SDPA receipt ordinals differ")
    reports = []
    for index, row in enumerate(rows):
        values = {name: array(root, row[name]) for name in ("query", "key", "value", "output")}
        require(all(row[name]["dtype"] == "torch.bfloat16" for name in values), "actual SDPA logical dtype differs")
        require(row["dropout_p"] == 0 and row["scale"] in (None, model_config["head_dim"] ** -.5), "SDPA scale/dropout differs")
        if index in (0, 1, 5, 6):
            case = CASES[index] if index < 2 else CASES[index - 5]
            lane = "functional" if index < 2 else "native"
            bank = cases[case][lane]
            same(values["query"], array(root, bank["q"]), "source actual consumed Q differs")
            for field, port in (("key", "k"), ("value", "v")):
                expected = array(root, bank[port])
                if values[field].shape[1] != expected.shape[1]:
                    expected = repeat(expected, values[field].shape[1])
                same(values[field], expected, "source actual consumed KV differs")
            same(values["output"], array(root, bank["head"]), "source actual attention return differs")
            original_mask = cases[case]["original_mask"]
            require((row["mask"] is None) == (original_mask is None), "source actual mask presence differs")
            if row["mask"] is not None:
                same(array(root, row["mask"]), array(root, original_mask), "source actual mask bytes differ")
            require(row["is_causal"] == cases[case]["dispatch"]["is_causal"], "source actual causal flag differs")
            require(row["enable_gqa"] == (row["mask"] is None), "source actual GQA dispatch differs")
        else:
            arm = ARMS[index - 2]
            bank = cases["long"]["functional"]
            same(values["query"], array(root, bank["q"]), "triangle changed fixed Q bank")
            for field, port in (("key", "k"), ("value", "v")):
                expected = array(root, bank[port])
                if index != 4:
                    expected = repeat(expected, model_config["num_attention_heads"])
                same(values[field], expected, "triangle changed fixed KV bank")
            same(values["output"], array(root, triangle["arms"][arm]["head"]), "triangle actual return differs")
            require(row["is_causal"] is (index != 2) and row["enable_gqa"] is (index == 4), "triangle causal/GQA flags differ")
            require((row["mask"] is not None) is (index == 2), "triangle mask presence differs")
            if index == 2:
                mask = array(root, row["mask"])
                same(mask, array(root, triangle["inputs"]["explicit_causal_mask"]), "actual explicit mask differs")
                same(mask, np.tril(np.ones((1, 1, values["query"].shape[2], values["query"].shape[2]), dtype=bool)),
                    "explicit full causal mask contents differ")
        profiler = row["profiler"]
        require(profiler["status"] in ("not-requested", "unavailable", "instrument-broken", "observed"), "unknown passive profiling status")
        if profiler["status"] == "observed":
            require(isinstance(profiler["events"], list), "missing observed passive profiler events")
        reports.append(dict(label=row["label"], ordinal=index, consumed_operand_bits_verified=True,
            operands={key: {name: row[key][name] for name in ("shape", "stride", "dtype", "device_at_capture", "raw_sha256")}
                      for key in ("query", "key", "value", "output")},
            mask=row["mask"], is_causal=row["is_causal"], enable_gqa=row["enable_gqa"], scale=row["scale"],
            dropout_p=row["dropout_p"], profiler=profiler,
            selected_kernel_claim="Passive event names only; no backend forcing, extra calls or independent kernel identification."))
    for name, port in (("repeated_k", "k"), ("repeated_v", "v")):
        same(array(root, triangle["inputs"][name]), repeat(array(root, cases["long"]["functional"][port]), model_config["num_attention_heads"]),
            "declared repeated-bank bytes differ")
    return reports


def frontier(root, inventory, seal, source, sdpa, plan):
    """Reconstruct the exact writer frontier, including no future receipt files."""
    initial_metadata = {"plan.json", "request.json", "runtime-contract.json"} | {Path(plan[key]).name for key in ("qwen_config", "qwen_manifest", "retrieval_script")}
    expected = {name for name in inventory["files"] if name in initial_metadata
        or name.startswith(("original-inputs/", "rotary-source/", "functional/", "triangle/", "triangle-inputs/"))
        or any(name.startswith("sdpa/" + str(index) + "/") for index in range(5))}
    require(set(seal["external_files"]) == expected, "complete pre-native file frontier differs")
    require(seal["sdpa_receipts"] == sdpa[:5] and seal["parameter_identity"] == source["parameter_identity_before"],
        "sealed first-five actual receipts/parameters differ")
    for name, record in seal["external_files"].items():
        require(record == inventory["files"][name], "sealed external member differs")
    records = [record for case in CASES for record in seal["cases"][case]["functional"].values()]
    records += [record for case in CASES if (record := seal["cases"][case]["original_mask"]) is not None]
    records += [record for arm in ARMS for record in seal["triangle"][arm].values()] + list(seal["grid_inputs"].values())
    records += [record for row in seal["sdpa_receipts"] for name in ("query", "key", "value", "output", "mask") if (record := row[name]) is not None]
    require(all(record["path"] in expected for record in records), "sealed field points outside pre-native inventory")
    digest = sha(Path(root) / "functional-seal.json")
    require(source["functional_seal_sha256"] == digest, "source seal checksum differs")
    events = read(Path(root) / "timeline.json")
    journal = [json.loads(line) for line in (Path(root) / "timeline.jsonl").read_text().splitlines()]
    require(events == journal and all(events[index]["seconds"] <= events[index + 1]["seconds"] for index in range(len(events) - 1)),
        "durable timeline/order differs")
    kinds = [row["kind"] for row in events]
    require(kinds == ["functional_sources_and_triangle_complete", "functional_fields_sealed", "native_block_entry", "native_block_completed",
                     "native_block_entry", "native_block_completed"], "actual functional/native entry chronology differs")
    require(events[0]["sdpa_calls"] == events[1]["sdpa_calls"] == 5 and events[0]["native_block_calls"] == events[1]["native_block_calls"] == 0
        and events[1]["seal_sha256"] == digest, "pre-native census/seal witness differs")
    for offset, case in ((2, "short"), (4, "long")):
        require(events[offset]["case"] == events[offset + 1]["case"] == case and events[offset]["functional_seal_sha256"] == digest
            and events[offset + 1]["sdpa_calls"] == offset // 2 + 5, "actual native entry/completion receipt differs")
    runtime = read(Path(root) / "runtime-contract.json")
    flags = runtime["flags"]
    require(runtime["device"] == plan["device"] and flags["deterministic_algorithms"] is True
        and flags["float32_matmul_precision"] == "highest" and flags["cuda_tf32"] is flags["cudnn_tf32"] is flags["cudnn_benchmark"] is False
        and flags["cudnn_deterministic"] is True and flags["cuda_bf16_reduced_precision_reduction"] == plan["source_reduction_policy"],
        "sealed actual numeric policy differs")
    require(all(runtime["versions"][name] == version for name, version in plan.get("expected_versions", {}).items()), "actual version policy differs")
    return dict(seal_sha256=digest, members=len(expected), bytes=sum(row["bytes"] for row in seal["external_files"].values()),
        first_five_sdpa_receipts_exact=True, native_blocks_at_seal=0, actual_native_entries=2, timeline=events, runtime=runtime)


def per_token(left, right, head_axis=False):
    if head_axis:
        return [[metric(left[:, head, token], right[:, head, token]) for token in range(left.shape[2])] for head in range(left.shape[1])]
    return [metric(left[:, token], right[:, token]) for token in range(left.shape[1])]


def reported_reductions(actual, recorded, path=""):
    """Reduction agreement is reported without making a numerical quality gate."""
    rows = []
    if isinstance(actual, dict) and isinstance(recorded, dict):
        for key in sorted(actual.keys() & recorded.keys()):
            rows += reported_reductions(actual[key], recorded[key], path + "/" + key)
    elif isinstance(actual, list) and isinstance(recorded, list):
        for index, (left, right) in enumerate(zip(actual, recorded, strict=False)):
            rows += reported_reductions(left, right, path + "/" + str(index))
    elif isinstance(actual, (float, int, bool)) and isinstance(recorded, (float, int, bool)) and actual != recorded:
        rows.append(dict(path=path, saved=recorded, recomputed=actual, signed_difference=float(actual) - float(recorded)))
    elif (actual is None) != (recorded is None):
        rows.append(dict(path=path, saved=recorded, recomputed=actual))
    return rows


def audit(root, output, *, expected_manifest, expected_request, input_root=None, parameter_root=None):
    import numpy as np

    root, output = Path(root), Path(output)
    plan, source, summary, inventory, metadata = authenticate(root, expected_manifest, expected_request)
    cases, triangle, sdpa = (read(root / name) for name in ("cases.json", "triangle.json", "sdpa.json"))
    require(set(cases) == set(CASES) and set(triangle["arms"]) == set(ARMS) and triangle["fixed_bank_case"] == "long", "case/triangle schema differs")
    seal = read(root / "functional-seal.json")
    require(seal["native_block_calls"] == 0 and seal["triangle"] == triangle["arms"] and seal["grid_inputs"] == triangle["inputs"],
        "sealed triangle/frontier differs")
    for case in CASES:
        require(set(seal["cases"][case]) == {"functional", "dispatch", "source_sidecars", "original_mask", "attended_rows", "retained_rows"}
            and set(cases[case]["functional"]) == set(cases[case]["native"]) == ALL_PORTS,
            "complete functional/native retained port schema differs")
        require(seal["cases"][case] == {key: value for key, value in cases[case].items() if key in seal["cases"][case]},
            "sealed source-generated fields differ")
        require(cases[case]["module_entries"].count("") == 1, "actual native block-entry census differs")
    frontier_report = frontier(root, inventory, seal, source, sdpa, plan)
    for name, record in source["rotary_buffers_before"].items():
        value = array(root, record)
        after = source["rotary_buffers_after"][name]
        require(record["dtype"] == "torch.float32" and value.dtype == np.dtype("float32") and after["dtype"] == record["dtype"]
            and after["shape"] == record["shape"] and after["raw_sha256"] == record["raw_sha256"], "actual FP32 rotary buffer identity differs")
    # Validate every retained field, including fields outside the comparison port subset.
    records = [record for case in CASES for lane in ("functional", "native") for record in cases[case][lane].values()]
    records += [record for arm in ARMS for record in triangle["arms"][arm].values()] + list(triangle["inputs"].values())
    for record in records:
        array(root, record)
    require(all(record["dtype"] == "torch.bfloat16" for case in CASES for lane in ("functional", "native") for record in cases[case][lane].values())
        and all(record["dtype"] == "torch.bfloat16" for arm in ARMS for record in triangle["arms"][arm].values()), "retained port logical dtype differs")
    require(all(set(triangle["arms"][arm]) == TAIL_PORTS for arm in ARMS), "complete triangle retained port schema differs")
    parameter_report = parameters(metadata, source, parameter_root or plan["parameter_root"])
    historical, operands, original_cross_case = original_sources(root, plan, cases, input_root or root / "original-inputs")
    sdpa_report = receipts(root, cases, triangle, sdpa, read(metadata["qwen_config"]))
    rows = plan["shared_prefix_rows"]
    require(summary["shared_prefix_rows"] == rows == triangle["historical_bridge_valid_prefix_rows"], "shared-prefix metadata differs")
    matched, pair, layouts = {}, {}, {}
    for case in CASES:
        matched[case] = {}
        layouts[case] = {}
        for name in set(cases[case]["functional"]) & set(cases[case]["native"]):
            p, n = (cases[case][lane][name] for lane in ("functional", "native"))
            matched[case][name] = metric(array(root, p), array(root, n))
            layouts[case][name] = dict(capture_stride_exact=p["stride"] == n["stride"], functional_stride=p["stride"], native_stride=n["stride"],
                logical_dtype_exact=p["dtype"] == n["dtype"], device_exact=p["device_at_capture"] == n["device_at_capture"],
                scope="Captured layout annotations; NPY payloads serialize contiguous logical values.")
    for lane in ("functional", "native"):
        pair[lane] = {name: metric(prefix(array(root, cases["long"][lane][name]), name, rows),
            prefix(array(root, cases["short"][lane][name]), name, rows)) for name in PORTS}
    accounts = {}
    for name in TAIL:
        values = {arm: array(root, triangle["arms"][arm][name]) for arm in ARMS}
        a, b, c = (values[arm] for arm in ARMS)
        historic = array(root, cases["long"]["native"][name]).astype(np.float64) - array(root, cases["short"]["native"][name]).astype(np.float64)
        first, second = b.astype(np.float64) - a.astype(np.float64), c.astype(np.float64) - b.astype(np.float64)
        accounts[name] = {label: dict(full=metric(left, right), shared_prefix=metric(prefix(left, name, rows), prefix(right, name, rows)))
            for label, left, right in (("B-minus-A", b, a), ("C-minus-B", c, b), ("C-minus-A", c, a))}
        accounts[name].update(prefix_telescope=vector_account(prefix(first, name, rows), prefix(second, name, rows), prefix(historic, name, rows)),
            coordinate_identity=np.array_equal(prefix(c.astype(np.float64) - a.astype(np.float64), name, rows), prefix(first + second, name, rows)))
    detailed = dict(historical_head=per_token(array(root, cases["long"]["native"]["head"]), array(root, cases["short"]["native"]["head"]), True),
        historical_H1=per_token(array(root, cases["long"]["native"]["H1"]), array(root, cases["short"]["native"]["H1"])),
        triangle_head={left + "-minus-" + right: per_token(array(root, triangle["arms"][left]["head"]), array(root, triangle["arms"][right]["head"]), True)
            for left, right in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]))})
    timeline = read(root / "timeline.json")
    report_differences = {"pair": reported_reductions(pair, read(root / "pair-comparisons.json")),
        "triangle": reported_reductions(accounts, triangle["accounts"]),
        "per_token_head": reported_reductions(detailed, read(root / "per-token-head.json")),
        "matched": {case: reported_reductions(matched[case], cases[case]["matched_functional_vs_native"]) for case in CASES},
        "historical": {case: reported_reductions(historical[case], cases[case]["historical_replay"]) for case in CASES}}
    head_account = accounts["head"]["prefix_telescope"]
    hypotheses = dict(positioned_prefix_QKV_exact={name: pair["native"][name]["bitwise_exact"] for name in ("q", "k", "v")},
        first_retained_prefix_difference_in_declared_port_order=next((name for name in PORTS if not pair["native"][name]["bitwise_exact"]), None),
        nonzero_historical_head=head_account["historical_energy"] > 0,
        equivalent_contrast_nonzero_positive_alignment={label: head_account[energy] > 0 and head_account[cosine] is not None and head_account[cosine] > 0
            for label, energy, cosine in (("B-minus-A", "first_energy", "first_vs_historical_cosine"), ("C-minus-B", "second_energy", "second_vs_historical_cosine"))},
        authority="Diagnostic observations; no hypothesis, equality or alignment outcome blocks evidence collection.")
    result = dict(schema="mdb-qwen-block0-attention-saved-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request,
        raw_manifest_sha256=expected_manifest, source=source, raw_members=len(inventory["files"]), raw_expanded_bytes=inventory["expanded_bytes"],
        parameters=parameter_report, actual=summary["actual"], timeline=timeline, frontier=frontier_report,
        historical_replay=historical, original_cross_case=original_cross_case,
        matched_functional_vs_native=matched, capture_layout=layouts, pair_comparisons=pair,
        triangle_accounts=accounts, per_token_head=detailed, sdpa=sdpa_report,
        hypotheses_observations=hypotheses, saved_vs_recomputed_reduction_differences=report_differences,
        port_provenance=dict(native=dict(H0="Supplied matched functional embedding operand, not a fresh native embedding call.",
            after_attention="Reconstructed H0 + captured o_proj output in BF16; not an independently hooked residual-input port.",
            mlp_input="Actual post-attention RMSNorm output captured by native module hook.",
            activation="Actual native down_proj input captured by prehook.", cos_sin="Returned by two parameter-free native rotary calls.",
            q_k_v_head="Actual SDPA consumed operands and return; repeated native KV are validated/canonicalized.",
            H1="Actual native block return.", other_module_ports="Actual observation-only native module output hooks; q_norm/k_norm transposed to head layout."),
            functional="Explicit original primitive graph fields, source formation observed; no independent kernel implementation.",
            triangle="Only SDPA head is the actual primitive intervention return; later ports are functional same-weight tails, not native branch calls."),
        source_masks={case: dict(attended=int(operands[case]["mask"].sum()), retained=int(operands[case]["mask"].size)) for case in CASES},
        sealed_functional_fields=dict(sha256=sha(root / "functional-seal.json"), native_block_calls_at_record=seal["native_block_calls"],
            scope="Source-recorded pre-native field commitment; original operands and all saved receipt payloads independently authenticated."),
        numerical_status="observed-full-saved-array-comparisons", mechanics_status="authenticated-saved-matched-operands", terminal_status="unassessed",
        new_model_calls=0, new_tokenizer_calls=0, new_cuda_calls=0, new_attention_calls=0,
        limitations=["All source endpoints were previously observed; triangle is an order-conditioned primitive representation contrast.",
            "Shared prefix alone supplies the historical causal-equivalence bridge; complete fixed-bank triangle covers all retained rows.",
            "Full fields and signed effects are observations. No automatic family induction, independent kernel, semantic reading, intrinsic gain, causal percentage or factorial attribution.",
            "Captured strides are source annotations joined across saved records; contiguous NPY serialization does not independently measure a historical live tensor stride."])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "per-token-head.json", detailed)
    catalog(root, output, result, cases, triangle)
    return result, inventory


def catalog(root, output, result, cases, triangle):
    from mdb.equation_comparison import EquationCatalog, compare_equations

    store = EquationCatalog(output / "catalog/equations")
    comparisons = [(case, cases[case]["functional"]["H1"], cases[case]["native"]["H1"], "matched-functional-native") for case in CASES]
    comparisons += [(left + "-minus-" + right, triangle["arms"][right]["H1"], triangle["arms"][left]["H1"], "fixed-bank-triangle")
        for left, right in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[2], ARMS[0]))]
    contexts = []
    for name, reference, observed, kind in comparisons:
        left, right = array(root, reference), array(root, observed)
        context = dict(occurrence=name, kind=kind, clock=dict(component="qwen", block=0, port="H1"),
            request_sha256=result["request_sha256"], source_job=result["job_id"],
            raw_manifest_sha256=result["raw_manifest_sha256"], functional_seal=result["sealed_functional_fields"]["sha256"],
            parameter_identity=result["parameters"]["rows"], fixed_triangle_bank="long")
        row = store.record(law_id="law.qwen.block0.matched-operand.observed.v1", route_id="route.qwen.block0.original-shared-primitives.v1",
            alias="cmp.qwen.block0." + name, binding=dict(model_identity="source-bound-original12-recipes", execution=dict(id="mdb-qwen-block0-attention-v1"),
                input_identity=context["functional_seal"], source_identity=result["request_sha256"], frontier="matched block0 operands", horizon=dict(block=0, port="H1")),
            report=compare_equations({"H1": left}, {"H1": right}, max_elements=int(left.size)),
            conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
            evidence=dict(audit_sha256=sha(output / "audit.json"), observations=True, independent_kernel=False, automatic_family_induction=False))
        exact = len(store.find(applicable_to=context))
        wrong = len(store.find(applicable_to={**context, "clock": dict(component="qwen", block=1, port="H1")}))
        require(exact == 1 and wrong == 0, "saved catalog wrong-clock abstention differs")
        contexts.append(dict(fingerprint=row["fingerprint"], context=context, exact=exact, wrong_clock=wrong))
    save(output / "catalog/provenance.json", dict(records=len(contexts), queries=contexts, automatic_family_induction=False,
        scope="Dependent saved H1 comparisons. Triangle catalog residuals use observed-minus-reference; audit vector accounts explicitly use B-A/C-B/C-A.",
        port_provenance=result["port_provenance"], residual_input_limit="Native after_attention is reconstructed H0+captured projection, not independently observed residual input."))


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(identity(request) == request["request_sha256"] and request["config"] == config
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged saved-reader source differs")
    started = time.monotonic()
    root = Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"],
        input_root=config.get("input_root"), parameter_root=config.get("parameter_root"))
    for name in ("plan.json", "request.json", "source.json", "summary.json", "cases.json", "triangle.json", "sdpa.json",
                 "functional-seal.json", "pair-comparisons.json", "timeline.json", "timeline.jsonl", "runtime-contract.json", "files.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key in ("qwen_config", "qwen_manifest", "retrieval_script"):
        shutil.copyfile(member(root, Path(read(root / "plan.json")[key]).name), output / Path(read(root / "plan.json")[key]).name)
    save(output / "external-evidence.json", dict(host="beast", roots=dict(native=str(root)),
        manifests=dict(native={name: row["sha256"] for name, row in inventory["files"].items()}),
        scope="Full raw arrays remain near data; compact reductions, original source metadata and standalone parameter recipes/retriever only."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-qwen-block0-attention-compact-v1", job_id=os.environ["MRUN_JOB_ID"],
        request_sha256=request["request_sha256"], source_job=result["job_id"], model_calls=0, records=2, triangle_arms=3,
        seconds=time.monotonic() - started, mechanics_status="verified-saved-block0-panel"))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        output_root=str(output), files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for key in ("root", "output", "expected-manifest", "expected-request"):
            parser.add_argument("--" + key, required=True)
        parser.add_argument("--input-root")
        parser.add_argument("--parameter-root")
        args = parser.parse_args()
        audit(args.root, args.output, expected_manifest=args.expected_manifest, expected_request=args.expected_request,
            input_root=args.input_root, parameter_root=args.parameter_root)
