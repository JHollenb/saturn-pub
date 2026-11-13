"""Bounded saved-vector bridge for the observed block0 attention triangle.

This supplements the complete frozen audit with selected full-vector accounts.
No model, attention, tokenizer, recipe hydration or learned recurrence is run.
Numerical disagreement is an observation, never an evidence-collection gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

HELPER = "audit_qwen_block0_attention.py"
HELPER_SHA256 = "5ab0ffc52537384341158f41edaa44d546232b7d4030d1c35a8949db81d11f8b"
PORTS = ("head", "projection", "after_attention", "mlp_input", "mlp", "H1")
ARMS = ("A.explicit-repeated", "B.causal-repeated", "C.causal-grouped")
TARGET = dict(job_id="job-0a3c41fce1d0", request_sha256="5579dc04e6fbcb20d675386833b4e3cc545f0f071383c2443cb72b108c260c36",
    manifest_sha256="e593670a65263168b08fb70fb684dca676ceb078fc78234dd11fb8e3cb3876a8")
COORDINATE_CAP = 128


def helpers():
    path = Path(__file__).with_name(HELPER)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError("frozen saved-array helper differs")
    return runpy.run_path(str(path))


def authenticate(root, expected_manifest, expected_request, h):
    root = Path(root)
    require, sha, read = h["require"], h["sha"], h["read"]
    require(sha(root / "files.json") == expected_manifest, "original wrapper checksum differs")
    inventory = read(root / "files.json")
    require(inventory["schema"] == "mdb-image-tail-files-v1" and inventory["request_sha256"] == expected_request,
        "original wrapper schema/request differs")
    selected = {}

    def record_file(name):
        path = h["member"](root, name)
        row = inventory["files"][name]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "selected original member differs: " + name)
        selected[name] = row
        return path

    names = ("plan.json", "request.json", "source.json", "summary.json", "cases.json", "triangle.json", "functional-seal.json", "timeline.json", "timeline.jsonl")
    metadata = {name: read(record_file(name)) for name in names if name != "timeline.jsonl"}
    journal = [json.loads(line) for line in record_file("timeline.jsonl").read_text().splitlines()]
    plan, request, source, summary, seal = (metadata[name] for name in ("plan.json", "request.json", "source.json", "summary.json", "functional-seal.json"))
    require(h["identity"](request) == request["request_sha256"] == expected_request and plan["request_sha256"] == expected_request,
        "standard executed request identity differs")
    require(all(plan[key] == value for key, value in request["config"].items()), "effective source plan differs")
    require(all(row["job_id"] == inventory["job_id"] and row["request_sha256"] == expected_request for row in (source, summary)), "actual job/request joins differ")
    require(request["source_hashes"]["qwen_block0_attention_microscope.py"] == source["worker_sha256"] == h["WORKER_SHA256"], "frozen producer source differs")
    require(source["helper_sources"] == h["HELPER_PINS"] and all(request["source_hashes"][key] == value for key, value in h["HELPER_PINS"].items()),
        "original arithmetic source pins differ")
    require(plan["schema"] == "mdb-qwen-block0-attention-plan-v1" and plan["fixed_bank_case"] == "long", "original diagnostic plan differs")
    if plan.get("profile") != "tiny-cpu":
        require(dict(job_id=inventory["job_id"], request_sha256=expected_request, manifest_sha256=expected_manifest) == TARGET,
            "accepted actual microscope execution differs")
        require(plan["shared_prefix_rows"] == 30, "actual bridge prefix differs")
    require(source["functional_seal_sha256"] == sha(root / "functional-seal.json") and seal["native_block_calls"] == 0,
        "original pre-native seal differs")
    require(journal == metadata["timeline.json"] and [row["kind"] for row in journal] == ["functional_sources_and_triangle_complete", "functional_fields_sealed",
        "native_block_entry", "native_block_completed", "native_block_entry", "native_block_completed"], "source durable entry chronology differs")
    require(journal[1]["seal_sha256"] == source["functional_seal_sha256"] and journal[1]["sdpa_calls"] == 5
        and journal[1]["native_block_calls"] == 0 and all(journal[index]["seconds"] <= journal[index + 1]["seconds"] for index in range(5)),
        "source seal/entry timeline differs")
    for offset, case in ((2, "short"), (4, "long")):
        require(journal[offset]["case"] == journal[offset + 1]["case"] == case
            and journal[offset]["functional_seal_sha256"] == source["functional_seal_sha256"], "actual source native entry differs")
    require(summary["actual"]["sdpa_calls"] == 7 and summary["actual"]["native_block_calls"] == 2, "source finite receipt census differs")
    cases, triangle = metadata["cases.json"], metadata["triangle.json"]
    require(set(cases) == {"short", "long"} and set(triangle["arms"]) == set(ARMS)
        and triangle["fixed_bank_case"] == "long" and triangle["historical_bridge_valid_prefix_rows"] == plan["shared_prefix_rows"], "selected case/triangle schema differs")
    for arm in ARMS:
        require(triangle["arms"][arm] == seal["triangle"][arm], "selected triangle differs from pre-native seal")
    return metadata, inventory, selected, record_file


def comparison(left, right, h):
    import numpy as np

    report = h["metric"](left, right)
    changed = np.flatnonzero((left.astype(np.float64) - right.astype(np.float64)).reshape(-1))
    report["exact_value_equality"] = report["changed_values"] == 0
    report["first_difference"] = None if not changed.size else dict(flat_index=int(changed[0]), coordinate=[int(x) for x in np.unravel_index(changed[0], left.shape)],
        left=float(left.reshape(-1)[changed[0]]), right=float(right.reshape(-1)[changed[0]]),
        signed_difference=float(left.reshape(-1)[changed[0]]) - float(right.reshape(-1)[changed[0]]))
    different_bits = np.flatnonzero(np.frombuffer(left.tobytes(), np.uint8) != np.frombuffer(right.tobytes(), np.uint8))
    first_bit = None if not different_bits.size else int(different_bits[0] // left.dtype.itemsize)
    report["first_bit_difference"] = None if first_bit is None else dict(flat_index=first_bit,
        coordinate=[int(x) for x in np.unravel_index(first_bit, left.shape)], left=float(left.reshape(-1)[first_bit]), right=float(right.reshape(-1)[first_bit]),
        left_hex=left.reshape(-1)[first_bit:first_bit + 1].tobytes().hex(), right_hex=right.reshape(-1)[first_bit:first_bit + 1].tobytes().hex())
    return report


def token_changes(left, right, name):
    import numpy as np

    delta = left.astype(np.float64) - right.astype(np.float64)
    axis = 2 if name == "head" else 1
    changed = np.moveaxis(delta != 0, axis, 0).reshape(delta.shape[axis], -1)
    counts = np.count_nonzero(changed, axis=1)
    return dict(total_changed_values=int(counts.sum()), changed_tokens=int(np.count_nonzero(counts)),
        tokens=[dict(token=index, changed_values=int(value)) for index, value in enumerate(counts) if value],
        scope="Same retained token coordinates; counts across ports are dependent observations, not new physical transitions or a gain estimate.")


def head_coordinates(a, b, c, short, long):
    import numpy as np

    first = b.astype(np.float64) - a.astype(np.float64)
    historical = long.astype(np.float64) - short.astype(np.float64)
    second = c.astype(np.float64) - b.astype(np.float64)
    changed = np.flatnonzero(((first != 0) | (historical != 0) | (second != 0)).reshape(-1))
    records = []
    for index in changed[:COORDINATE_CAP]:
        coordinate = tuple(int(x) for x in np.unravel_index(index, a.shape))
        records.append(dict(batch=coordinate[0], head=coordinate[1], token=coordinate[2], channel=coordinate[3],
            A=float(a[coordinate]), B=float(b[coordinate]), C=float(c[coordinate]), native_short=float(short[coordinate]), native_long=float(long[coordinate]),
            B_minus_A=float(first[coordinate]), native_long_minus_short=float(historical[coordinate]), C_minus_B=float(second[coordinate]),
            bridge_residual=float(first[coordinate] - historical[coordinate])))
    return dict(scope="Shared-prefix head-coordinate union of nonzero B-A, native.long-native.short and C-B; no causal attribution inferred.",
        total_coordinates=int(changed.size), descriptive_cap=COORDINATE_CAP, complete=int(changed.size) <= COORDINATE_CAP, coordinates=records)


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np

    h = helpers()
    root, output = Path(root), Path(output)
    metadata, inventory, selected, record_file = authenticate(root, expected_manifest, expected_request, h)
    cases, triangle, plan = (metadata[name] for name in ("cases.json", "triangle.json", "plan.json"))
    rows = plan["shared_prefix_rows"]
    reports, token_reports, coordinates = {}, {}, None
    for name in PORTS:
        records = {arm: triangle["arms"][arm][name] for arm in ARMS}
        records.update({case: cases[case]["native"][name] for case in ("short", "long")})
        values = {}
        for lane, record in records.items():
            record_file(record["path"])
            h["require"](inventory["files"][record["path"]] == {key: record[key] for key in ("bytes", "sha256")}
                and record["dtype"] == "torch.bfloat16", "selected port inventory/dtype differs")
            if lane in ARMS:
                h["require"](metadata["functional-seal.json"]["external_files"][record["path"]] == inventory["files"][record["path"]],
                    "selected triangle array differs from pre-native commitment")
            values[lane] = h["array"](root, record)
        full_a, full_b, full_c = (values[arm] for arm in ARMS)
        h["require"](full_a.shape == full_b.shape == full_c.shape == values["short"].shape == values["long"].shape, "selected bridge geometry differs")
        if plan.get("profile") != "tiny-cpu":
            h["require"](full_a.shape[2 if name == "head" else 1] == 512, "actual complete triangle extent differs")
        a, b, c, short, long = (h["prefix"](value, name, rows) for value in (full_a, full_b, full_c, values["short"], values["long"]))
        first = b.astype(np.float64) - a.astype(np.float64)
        historical = long.astype(np.float64) - short.astype(np.float64)
        reports[name] = dict(A_vs_native_short_shared_prefix=comparison(a, short, h), B_vs_native_long_shared_prefix=comparison(b, long, h),
            C_vs_native_long_shared_prefix=comparison(c, long, h), signed_vector_bridge_shared_prefix=comparison(first, historical, h),
            C_minus_B_complete=comparison(full_c, full_b, h), B_minus_A_shared_prefix=comparison(b, a, h),
            native_long_minus_short_shared_prefix=comparison(long, short, h),
            arithmetic="float64(B)-float64(A) versus float64(native.long)-float64(native.short); bridge residual is first minus second.",
            first_vector_sha256=hashlib.sha256(np.ascontiguousarray(first, dtype="<f8").tobytes()).hexdigest(),
            historical_vector_sha256=hashlib.sha256(np.ascontiguousarray(historical, dtype="<f8").tobytes()).hexdigest(),
            complete_C_minus_B_rows=int(full_c.shape[2 if name == "head" else 1]))
        token_reports[name] = {"B-minus-A": token_changes(b, a, name), "native.long-minus-native.short": token_changes(long, short, name),
            "bridge-residual": token_changes(first, historical, name), "C-minus-B-complete": token_changes(full_c, full_b, name)}
        if name == "head":
            coordinates = head_coordinates(a, b, c, short, long)
    result = dict(schema="mdb-qwen-attention-signed-vector-bridge-v1", source_job=inventory["job_id"], source_request_sha256=expected_request,
        source_manifest_sha256=expected_manifest, producer_sha256=metadata["source.json"]["worker_sha256"], frozen_saved_reader_sha256=HELPER_SHA256,
        shared_prefix_rows=rows, ports=reports, mechanics_status="authenticated-selected-saved-vectors", numerical_status="observed-full-vector-comparisons",
        terminal_status="unassessed", new_model_calls=0, new_attention_calls=0, new_tokenizer_calls=0, new_recipe_reads=0,
        scope="Supplement to completed original audit; selected arrays only. No recurrence, weights, automatic family induction, independent kernel or factorial attribution.",
        port_provenance_limit="Native after_attention is reconstructed BF16 H0+captured projection, not an independently hooked residual-input port. Native post-attention RMSNorm/MLP ports and H1 are captured.",
        response_limit="Shared prefix supplies the historical bridge; complete C-B uses the fixed long bank. Token counts across ports and stages are dependent and use different coordinates/units.")
    output.mkdir(parents=True, exist_ok=False)
    h["save"](output / "bridge.json", result)
    h["save"](output / "head-coordinates.json", coordinates)
    h["save"](output / "token-changes.json", token_reports)
    h["save"](output / "source-custody.json", dict(source_job=inventory["job_id"], source_request_sha256=expected_request,
        source_manifest_sha256=expected_manifest, functional_seal_sha256=metadata["source.json"]["functional_seal_sha256"],
        selected_members=selected, selected_members_count=len(selected), selected_bytes=sum(row["bytes"] for row in selected.values()),
        original_inventory_members=len(inventory["files"]), original_expanded_bytes=inventory["expanded_bytes"],
        scope="Exact original wrapper and selected metadata/full arrays checked; original full recipe/array audit not repeated."))
    return result


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    h = helpers()
    request, config = h["read"](os.environ["SATURN_DEBUG_REQUEST"]), h["read"](os.environ["SATURN_RUN_CONFIG"])
    h["require"](h["identity"](request) == request["request_sha256"] and request["config"] == config
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "bounded saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        h["require"](h["sha"](name) == digest, "staged bridge source differs")
    started = time.monotonic()
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result = audit(config["native_root"], output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    h["save"](output / "summary.json", dict(schema="mdb-qwen-attention-bridge-compact-v1", job_id=os.environ["MRUN_JOB_ID"],
        request_sha256=request["request_sha256"], source_job=result["source_job"], model_calls=0, ports=len(PORTS), seconds=time.monotonic() - started,
        numerical_status=result["numerical_status"], mechanics_status=result["mechanics_status"]))
    h["save"](output / "request.json", request)
    h["save"](output / "files.json", {path.relative_to(output).as_posix(): h["sha"](path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=h["sha"](output / "files.json"), summary_sha256=h["sha"](output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for key in ("root", "output", "expected-manifest", "expected-request"):
            parser.add_argument("--" + key, required=True)
        args = parser.parse_args()
        audit(args.root, args.output, expected_manifest=args.expected_manifest, expected_request=args.expected_request)
