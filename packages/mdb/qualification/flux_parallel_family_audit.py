"""Independent saved-only audit of the prospective parallel image family.

No parameters are hydrated and no model module may run. Original projection
products are supplied targets; NumPy independently reconstructs their declared
merge operands and finite BF16 gate/add endpoints. Numeric and rank misses are
observations. Only custody, typed closure and instrument census are hard checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import math
import os
import runpy
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np

SCHEMA = "mdb-flux-parallel-family-saved-audit-v1"
RESULT = "parallel-family-audit-result"
HELPERS = {"flux_binding_family_audit.py": "705fc21e5ab6d1fd012ecb0ed6157ef4195323643b7f66e1781bb67b1a05cadd"}
WRITER_SHA256 = "23c02673a04bfa8e0f28e6fd70a2f104395f3c22434acd0de1f550c8da0a372f"
MODEL_ID = "a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0"
ARMS = ("Q", "P", "R", "A", "RA")
SOURCE_PORTS = {"R", "A", "M", "gate", "projection_return", "out", "image_in", "text_in", "image_out", "text_out"}
CONSUMER_PORTS = {"image_in", "text_in", "A", "M", "image_out", "text_out", "projection_return", "gate"}
PORT_NAMES = {
    "image_in", "text_in", "R", "shift", "scale", "gate", "norm", "x", "fused_input", "fused_return",
    "q_raw", "k_raw", "v_raw", "q_norm", "k_norm", "q_consumed", "k_consumed", "v_consumed",
    "sdpa_return", "A", "mlp_input", "M", "merge_input", "projection_return", "write", "out", "image_out", "text_out",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def request_identity(request):
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


@cache
def utilities():
    name, expected = next(iter(HELPERS.items()))
    path = Path(__file__).with_name(name)
    require(sha(path) == expected, "saved-only utility helper differs")
    return runpy.run_path(str(path))


def metric(left, right):
    return utilities()["metric"](left, right)


def bf(value):
    return utilities()["bf"](value)


def bf_bits(value):
    return (np.asarray(value, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def content(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def pair_account(left, right):
    a, b = (np.asarray(value) for value in (left, right))
    result = metric(a, b)
    result.update(raw_bit_exact=a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C"),
        zero_sign_mismatches=int(np.count_nonzero((a == 0) & (b == 0) & (np.signbit(a) != np.signbit(b)))),
        candidate_negative_zeros=int(np.count_nonzero((a == 0) & np.signbit(a))),
        reference_negative_zeros=int(np.count_nonzero((b == 0) & np.signbit(b))))
    return result


class Publication:
    """Authenticate every member before any CAS or NumPy hydration."""
    def __init__(self, root, manifest_sha256, job_id, request_sha256):
        self.root = Path(root).resolve()
        require(sha(self.root / "files.json") == manifest_sha256, "publication inventory differs")
        manifest = read(self.root / "files.json")
        require(manifest["job_id"] == job_id and manifest["request_sha256"] == request_sha256, "publication owner differs")
        self.files, self.arrays, self.elements = manifest["files"], set(), 0
        actual = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
        require(actual == set(self.files) | {"files.json"}, "unpublished or absent file in result")
        for name in self.files:
            self.member(name)

    def member(self, name):
        require(isinstance(name, str) and name and not Path(name).is_absolute()
                and ".." not in Path(name).parts and "\\" not in name, "unsafe inventoried member")
        path = self.root / name
        require(path.resolve().is_relative_to(self.root) and not any(item.is_symlink() for item in (path, *path.parents))
                and path.is_file() and name in self.files, "nonregular or unlisted member")
        record = self.files[name]
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "published member changed: " + name)
        return path

    def array(self, record, base=None, *, raw=False):
        path = (self.root if base is None else Path(base)) / record["path"]
        name = path.relative_to(self.root).as_posix()
        path = self.member(name)
        require(sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "array/file binding differs")
        value = np.load(path, allow_pickle=False)
        stored_dtype = record.get("stored_dtype", record.get("dtype"))
        require(list(value.shape) == record["shape"] and str(value.dtype) == stored_dtype, "array interpretation differs")
        if "content_sha256" in record:
            require(content(value) == record["content_sha256"], "array content differs")
        encoding = record.get("encoding", "native-numpy-v1")
        require(encoding in ("bf16-bits-v1", "native-numpy-v1"), "unknown array encoding")
        if encoding == "bf16-bits-v1":
            require(value.dtype == np.uint16 and record["logical_dtype"] == "torch.bfloat16", "invalid BF16 storage")
        elif "encoding" not in record:
            require(set(record) == {"path", "sha256", "bytes", "shape", "dtype"} and value.dtype == np.uint8,
                    "unsupported byte-array record")
        self.arrays.add(name)
        self.elements += value.size
        result = value if raw or encoding != "bf16-bits-v1" else utilities()["decode_bits"](value)
        require(np.isfinite(result).all(), "nonfinite saved array")
        return result

    def compare(self, a, b, *, base_a=None, base_b=None):
        left, right = self.array(a, base_a, raw=True), self.array(b, base_b, raw=True)
        require(left.dtype == right.dtype and left.shape == right.shape
                and all(a.get(k) == b.get(k) for k in ("encoding", "logical_dtype", "stored_dtype", "dtype")), "full array ABI differs")
        result = pair_account(self.array(a, base_a), self.array(b, base_b))
        result["raw_bit_exact"] = left.tobytes(order="C") == right.tobytes(order="C")
        return result


def port_account(publication, record, expected_ports):
    require(set(record["ports"]) == expected_ports and set(record["descriptors"]) == PORT_NAMES
            and set(record["retained_descriptors"]) == expected_ports, "port/descriptor census differs")
    require(set(record["retention"]["full_arrays"]) == expected_ports
            and set(record["retention"]["hash_and_layout_only"]) == PORT_NAMES - expected_ports, "retention declaration differs")
    layout = {}
    for name, descriptor in record["descriptors"].items():
        require(set(descriptor) == {"shape", "stride", "dtype", "device", "content_sha256", "hash_order"}
                and len(descriptor["shape"]) == len(descriptor["stride"])
                and descriptor["hash_order"] == "contiguous logical element order, dtype bytes"
                and len(descriptor["content_sha256"]) == 64, "invalid live descriptor")
        if name not in expected_ports:
            layout[name] = dict(coverage="live logical-byte hash and layout only")
            continue
        array_record, kept = record["ports"][name], record["retained_descriptors"][name]
        raw = publication.array(array_record, raw=True)
        require(content(raw) == descriptor["content_sha256"] == kept["content_sha256"]
                and array_record["shape"] == descriptor["shape"] == kept["shape"]
                and array_record["logical_dtype"] == descriptor["dtype"] == kept["dtype"]
                and array_record["stride"] == kept["stride"]
                and array_record["device_at_capture"] == kept["device"] == descriptor["device"], "retained/live port join differs")
        layout[name] = dict(coverage="complete retained array plus live descriptor",
            live_stride=descriptor["stride"], retained_stride=kept["stride"],
            snapshot_layout_changed=descriptor["stride"] != kept["stride"])
    return layout


def descriptor_comparison(left, right):
    require(set(left) == set(right) == PORT_NAMES, "live descriptor pair census differs")
    return {name: dict(hash_equal=left[name]["content_sha256"] == right[name]["content_sha256"],
        live_layout_equal=all(left[name][key] == right[name][key] for key in ("shape", "stride", "dtype", "device")),
        forecast_layout={key: left[name][key] for key in ("shape", "stride", "dtype", "device")},
        native_layout={key: right[name][key] for key in ("shape", "stride", "dtype", "device")}) for name in PORT_NAMES}


def boundary_account(left, right, path="", result=None):
    result = dict(tensors=0, scalar_fields=0, hash_mismatches=[], layout_mismatches=[], value_mismatches=[], scalar_mismatches=[]) if result is None else result
    if isinstance(left, dict) and "tensor" in left:
        require(isinstance(right, dict) and "tensor" in right and set(left) == set(right), "boundary tensor ABI differs")
        a, b = left["tensor"], right["tensor"]
        result["tensors"] += 1
        if a["sha256"] != b["sha256"]:
            result["hash_mismatches"].append(path)
        if any(a[key] != b[key] for key in ("shape", "dtype", "stride")):
            result["layout_mismatches"].append(path)
        require(left["elements"] == right["elements"] and left["witness_indices"] == right["witness_indices"], "boundary witness addresses differ")
        values = pair_account(left["witness_values"], right["witness_values"])
        if values["changed"] or not values["raw_bit_exact"] or left["rms"] != right["rms"] or left["signed_mean"] != right["signed_mean"]:
            result["value_mismatches"].append(dict(path=path, sampled_values=values,
                rms_difference=left["rms"] - right["rms"], signed_mean_difference=left["signed_mean"] - right["signed_mean"]))
    elif isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), "boundary field census differs")
        for name in left:
            boundary_account(left[name], right[name], path + "/" + name, result)
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), "boundary sequence differs")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            boundary_account(a, b, path + "/" + str(index), result)
    else:
        result["scalar_fields"] += 1
        if type(left) is not type(right) or left != right:
            result["scalar_mismatches"].append(dict(path=path, forecast=left, native=right))
    return result


def endpoint_account(q, p, projection, arm, ntext):
    """NumPy BF16 multiply/add; original full projection is already paid."""
    require(arm in ARMS, "unknown endpoint arm")
    if arm == "P":
        return p["image_out"].copy(), None, None
    r, a = q["R"].copy(), q["A"].copy()
    if arm in ("R", "RA"):
        r[:, ntext:] = p["R"][:, ntext:]
    if arm in ("A", "RA"):
        a[:, ntext:] = p["A"][:, ntext:]
    merged = np.concatenate((a, q["M"]), axis=-1)
    write = bf(q["gate"] * projection)
    return bf(r + write)[:, ntext:], merged, (write, bf(r + write))


def typed_mapping(value):
    require(isinstance(value, dict) and set(value) == {"kind", "items"} and value["kind"] == "mapping", "typed mapping required")
    result = {}
    for item in value["items"]:
        key = item["key"]
        require(set(item) == {"key", "value"} and key == dict(kind="scalar", type="str", value=key.get("value"))
            and isinstance(key["value"], str) and key["value"] not in result, "invalid or duplicate typed mapping key")
        result[key["value"]] = item["value"]
    return result


def c_stride(shape):
    require(isinstance(shape, list) and shape and all(type(size) is int and size > 0 for size in shape), "positive array shape required")
    result, step = [], 1
    for size in reversed(shape):
        result.insert(0, step)
        step *= size
    return result


def patch_parameter_account(receipt, before, after, operand, loaded_stride):
    """Bind endpoint capture, NPY reload and recipient layout separately.

    The registered full-mask patch clones the receiver, then assigns supplied
    values. Its parameter tensor therefore need not share the receiver stride.
    This function consumes descriptors only; complete CAS/receipt lineage is
    checked by the caller and the actual endpoint file is independently hashed.
    """
    params = typed_mapping(receipt["act"]["parameters"])
    require(set(params) == {"address", "dose", "mask", "mode", "value"}, "Act parameter census differs")
    for name, value, scalar_type in (("address", "image", "str"), ("mode", "replace", "str"), ("dose", 1, "int")):
        require(params[name] == dict(kind="scalar", type=scalar_type, value=value), "Act parameter differs: " + name)
    require(before["kind"] == after["kind"] == "tensor" and before["layout"] == after["layout"] == "strided-dense"
        and all(before[name] == after[name] for name in ("shape", "stride", "dtype")), "Act destination layout differs from receiver")
    require(loaded_stride == c_stride(operand["shape"]), "endpoint NPY reload is not the declared C layout")
    expected_value = dict(kind="tensor", layout="strided-dense", sha256=operand["content_sha256"], tensor=operand["content_sha256"],
        shape=operand["shape"], dtype=operand["logical_dtype"], stride=loaded_stride)
    require(params["value"] == expected_value, "Act value differs from authenticated endpoint reload")
    require(all(after[key] == expected_value[key] for key in ("shape", "dtype", "sha256", "tensor")), "Act result content differs from endpoint")
    writes = typed_mapping(receipt["writes"])
    require(set(writes) == {"image"} and writes["image"] == after, "Act typed writes differ from after CAS")
    size = math.prod(before["shape"])
    mask_hash = hashlib.sha256()
    chunk = b"\x01" * min(size, 1 << 20)
    for offset in range(0, size, len(chunk)):
        mask_hash.update(chunk[:min(len(chunk), size-offset)])
    expected_mask = dict(kind="tensor", layout="strided-dense", sha256=mask_hash.hexdigest(), tensor=mask_hash.hexdigest(),
        shape=before["shape"], dtype="torch.bool", stride=before["stride"])
    require(params["mask"] == expected_mask, "Act mask differs from full receiver-shaped ones")
    return dict(endpoint_capture=dict(origin="producer endpoint before NPY serialization", shape=operand["shape"],
                    dtype=operand["logical_dtype"], stride=operand["stride"], content_sha256=operand["content_sha256"]),
        supplied_value=dict(origin="authenticated NPY C reload supplied to Act.patch", descriptor=params["value"]),
        recipient_before=dict(origin="pre-Act CAS receiving slot", descriptor=before),
        recipient_after=dict(origin="post-Act CAS and exact receipt.writes", descriptor=after),
        mask=dict(origin="full ones_like receiver, preserved receiver layout", descriptor=params["mask"]),
        supplied_and_receiver_layout_differ=params["value"]["stride"] != after["stride"], receiver_layout_preserved=True)


def axis_accounts(values):
    """The frozen signed coordinate, distances and orthogonal errors; no gate."""
    p, q = (np.asarray(values[name], dtype=np.float64) for name in ("P", "Q"))
    axis = q-p
    denominator = float(np.sum(axis*axis))
    rows = {}
    for arm, value in values.items():
        displacement = np.asarray(value, dtype=np.float64)-p
        coordinate = float(np.sum(displacement*axis)/denominator) if denominator else None
        orthogonal = displacement-coordinate*axis if coordinate is not None else None
        rows[arm] = dict(p=coordinate, source_l2=float(np.linalg.norm(displacement)), source_rms=metric(value, p)["rms"],
            recipient_l2=float(np.linalg.norm(np.asarray(value, dtype=np.float64)-q)), recipient_rms=metric(value, q)["rms"],
            orthogonal_l2=None if orthogonal is None else float(np.linalg.norm(orthogonal)),
            orthogonal_rms=None if orthogonal is None else float(np.sqrt(np.mean(orthogonal*orthogonal))))
    applicable = denominator != 0.
    margins = {name: None if not applicable else rows[name]["p"]-rows["RA"]["p"] for name in ("R", "A")}
    d_r, d_a = (np.asarray(values[name], dtype=np.float64)-q for name in ("R", "A"))
    factor_doses = {name: dict(l2=float(np.linalg.norm(np.asarray(values[name], dtype=np.float64)-q)),
                              rms=metric(values[name], q)["rms"]) for name in ("R", "A", "RA")}
    return dict(status="applicable" if applicable else "not-applicable-zero-native-axis", axis_l2=float(np.sqrt(denominator)),
        axis_rms=float(np.sqrt(np.mean(axis*axis))), rows=rows, signed_rank_margins=margins,
        frozen_predictions={"pRA_lt_pR": None if not applicable else margins["R"] > 0.,
                            "pRA_lt_pA": None if not applicable else margins["A"] > 0.},
        both_rank_predictions=None if not applicable else margins["R"] > 0. and margins["A"] > 0.,
        recipient_displacement_doses=factor_doses, r_a_displacement_pair=metric(d_r, d_a),
        projected_mixed_remainder=None if not applicable else rows["RA"]["p"]-rows["R"]["p"]-rows["A"]["p"]+1.)


def transaction_account(publication, summary):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root = publication.root
    transaction = verify_transaction(root / "forecast-transaction", expected_seal_sha256=summary["forecast_seal"],
        expected_event_head=summary["transaction"]["event_head"])
    forecast = load_forecast_tree(root / "forecast-transaction",
        read(root / "forecast-transaction/forecasts/all-numeric-futures.json")["predictions"])
    for name in ("native_targets_at_seal", "native_source_blocks_at_seal", "native_hybrid_projections_at_seal", "native_decodes_at_seal"):
        require(forecast[name] == 0, "native work before global seal: " + name)
    for name, record in forecast["external_files"].items():
        require(publication.files.get(name) == record, "sealed external file differs")
    timeline = [json.loads(line) for line in publication.member("timeline.jsonl").read_text().splitlines()]
    seals = [index for index, row in enumerate(timeline) if row["kind"] == "all_forecasts_sealed"]
    require(len(seals) == 1 and timeline[seals[0]]["seal"] == summary["forecast_seal"], "global timeline seal differs")
    require(all(index > seals[0] for index, row in enumerate(timeline) if row["kind"].startswith("native_")), "native chronology precedes seal")
    return forecast, transaction


@contextmanager
def model_free():
    class NoModelFramework(importlib.abc.MetaPathFinder):
        roots = {"diffusers", "transformers", "tensorflow", "jax", "saturn", "common", "common_harness"}

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.roots:
                raise ImportError("saved reader refuses model frameworks/private packages")

    require(not NoModelFramework.roots.intersection(sys.modules), "saved-only process already contains a model framework")
    guard = NoModelFramework()
    sys.meta_path.insert(0, guard)
    handle = None
    try:
        import torch

        require(not torch.cuda.is_initialized(), "saved reader initialized CUDA")
        torch.set_num_threads(2)

        def reject(_module, _args):
            raise RuntimeError("saved-only reader refuses every nn.Module forward")

        handle = torch.nn.modules.module.register_module_forward_pre_hook(reject)
        yield torch
        require(not torch.cuda.is_initialized(), "saved reader initialized CUDA")
    finally:
        if handle is not None:
            handle.remove()
        sys.meta_path.remove(guard)


def validate_config(config):
    for name in ("native_root", "native_manifest_sha256", "native_request_sha256", "native_job_id"):
        require(isinstance(config.get(name), str) and config[name], "missing audit config: " + name)
    require("output" in config or "output_scratch" in config, "missing audit output")
    return dict(config)


def run(config):
    config = validate_config(config)
    with model_free() as torch:
        return _run(config, torch)


def _run(config, torch):
    from PIL import Image
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe, digest

    from mdb.equation_comparison import compare_equations
    from mdb.forecast_transaction import verify_transaction

    pub = Publication(config["native_root"], config["native_manifest_sha256"], config["native_job_id"], config["native_request_sha256"])
    root = pub.root
    plan, request, summary = (read(pub.member(name)) for name in ("plan.json", "request.json", "summary.json"))
    require(request_identity(request) == request["request_sha256"] == config["native_request_sha256"], "executed request changed")
    require({k: v for k, v in plan.items() if k not in ("job_id", "request_sha256", "output")} == request["config"], "executed plan/config differs")
    require(summary["schema"] == "mdb-flux-parallel-relation-family-v1" and summary["job_id"] == config["native_job_id"]
            and summary["request_sha256"] == config["native_request_sha256"], "source summary differs")
    require(request["source_hashes"]["flux_parallel_family_writer.py"] == WRITER_SHA256, "unqualified writer source")
    rule = read(pub.member("family-rule.json"))
    require(rule == plan["family_rule"] and sha(root / "family-rule.json") == plan["family_rule_file_sha256"], "frozen family rule differs")
    require(rule["frontier"] == dict(step=0, phase="single", index=4, edge="before")
        and rule["selected_component"] == "RA" and rule["effect_hypotheses"] == ["p(RA)<p(R)", "p(RA)<p(A)"], "unsupported family rule")
    contexts = {row["id"]: row for row in plan["contexts"]}
    require(len(contexts) == 6 and plan["contexts"] == rule["fresh_contexts"] and plan["arms"] == rule["arms"] == list(ARMS), "finite panel differs")
    expected = dict(functional_source_blocks=12, native_source_blocks=12, functional_hybrid_projections=24, native_hybrid_projections=24,
        functional_transitions=3030, native_target_transitions=3030, native_blocks=2712,
        functional_vae=30, native_vae=30, acts=48, forecast_acts=24, native_acts=24, rewinds=6)
    contract = read(pub.member("native-contract.json"))
    require(summary["counts"] == contract["counts"] == expected and summary["observed"] == contract["observed"], "actual physical census differs")
    require(all(not values.get("module_entries", 0) for lane, values in contract["observed"].items() if lane.startswith("forecast")), "functional lane entered model modules")
    forecast, transaction = transaction_account(pub, summary)
    output = Path(config["output"])
    require(not output.resolve().is_relative_to(root) and not root.is_relative_to(output.resolve()), "source/output overlap")
    output.mkdir(parents=True, exist_ok=False)
    store = LocalStore(root / "state-store")

    def cut(record):
        value = store.load(record["cut"], device="cpu")
        require(value.fingerprint == record["cut"] and value.parent == record["parent"]
                and value.model_identity == record["model_identity"] == MODEL_ID
                and dict(value.execution_point) == record["execution_point"], "CAS/cut record binding differs")
        return value

    def join(value, record):
        actual = describe(value)
        require(actual["sha256"] == record["content_sha256"] and actual["shape"] == record["shape"]
                and actual["stride"] == record["stride"] and actual["dtype"] == record["logical_dtype"], "full array/CAS join differs")

    def numpy_tensor(value):
        return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()

    roots, provenance = (read(pub.member(name)) for name in ("source-roots.json", "source-provenance.json"))
    require(set(roots) == set(contexts), "source context census differs")
    source = provenance["source"]
    require(source == plan["fresh_source"] and provenance["family_rule_sha256"] == plan["family_rule_file_sha256"], "source publication binding differs")
    original = Publication(source["output_root"], source["files_manifest_sha256"], source["job_id"], source["request_sha256"])
    for name, record in provenance["consumed_members"].items():
        require(original.files.get(name) == record, "consumed prefix member lineage differs")
    prefix_request, prefix_summary = (read(original.member(name)) for name in ("request.json", "summary.json"))
    require(request_identity(prefix_request) == source["request_sha256"]
        and prefix_request["config"]["family_rule"] == rule
        and sha(original.member("family-rule.json")) == plan["family_rule_file_sha256"], "prefix rule/request differs")
    require(prefix_summary["schema"] == "mdb-parallel-relation-prefix-v1"
        and prefix_summary["counts"] == dict(contexts=6, prompt_encodes=12, noise_draws=3, native_transitions=120,
            native_blocks=108, selected_block_calls=0, decoder_calls=0, images=0, input_cuts=12, pipeline_loads=1,
            encoder_cuda_placements=1, transformer_cuda_placements=1), "prefix no-target census differs")
    prefix_tx = verify_transaction(original.root / "source-transaction", expected_seal_sha256=prefix_summary["source_policy_seal"])
    require(roots == read(original.member("source-roots.json")), "copied source roots differ")
    original_store = LocalStore(original.root / "state-store")
    writers = {lane: read(pub.member(lane + "-source-writers.json")) for lane in ("forecast", "native")}
    writer_checks, source_joins, endpoint_checks, construction_checks = {}, {}, {}, {}
    layout_checks, live_checks = {}, {}
    full_array_fields = 0

    for cid, source_root in roots.items():
        require(source_root["context"] == contexts[cid], "source prompt/seed differs")
        inputs = {role: cut(source_root["inputs"][role]) for role in ("P", "Q")}
        for role, source_cut in inputs.items():
            original_cut = original_store.load(source_cut.fingerprint, device="cpu")
            require(source_cut.manifest() == original_cut.manifest()
                and (source_cut.payload["step"], source_cut.payload["phase"], source_cut.payload["cursor"]) == (0, "single", 4), "source input frontier differs")
        for name in ("latent", "img_ids", "txt_ids", "timesteps", "sigmas"):
            require(describe(inputs["P"].payload[name]) == describe(inputs["Q"].payload[name]), "paired source noise/IDs/schedule differ")
        writer_checks[cid], source_joins[cid], layout_checks[cid], live_checks[cid] = {}, {}, {}, {}
        for role in ("P", "Q"):
            f, n = (writers[lane][cid]["sources"][role] for lane in ("forecast", "native"))
            live_checks[cid][role] = descriptor_comparison(f["descriptors"], n["descriptors"])
            layout_checks[cid][role] = {lane: port_account(pub, writers[lane][cid]["sources"][role], SOURCE_PORTS) for lane in ("forecast", "native")}
            writer_checks[cid][role] = dict(comparisons={name: pub.compare(f["ports"][name], n["ports"][name]) for name in SOURCE_PORTS}, finite_closure={})
            full_array_fields += len(SOURCE_PORTS)
            source_joins[cid][role] = {}
            for lane in ("forecast", "native"):
                saved = writers[lane][cid]["sources"][role]
                after = cut(saved["state"])
                profile = saved["profile"]
                require(profile["model_identity"] == MODEL_ID and profile["index"] == 4
                    and profile["point"] == dict(inputs[role].execution_point)
                    and profile["execution"] == dict(inputs[role].execution), "writer profile/source point differs")
                for name, value in dict(image=inputs[role].payload["image"], text=inputs[role].payload["text"],
                        single_mod=inputs[role].payload["context"]["single_mod"],
                        rope_cos=inputs[role].payload["context"]["rope"][0], rope_sin=inputs[role].payload["context"]["rope"][1]).items():
                    descriptor = describe(value)
                    require(all(descriptor[key] == profile["interfaces"][name][target] for key, target in
                        (("sha256", "content_sha256"), ("shape", "shape"), ("stride", "stride"), ("dtype", "dtype"))), "source profile operand differs")
                require((after.payload["step"], after.payload["phase"], after.payload["cursor"]) == (0, "single", 5), "writer exit clock differs")
                for stream in ("image", "text"):
                    join(inputs[role].payload[stream], saved["ports"][stream + "_in"])
                    join(after.payload[stream], saved["ports"][stream + "_out"])
                values = {name: pub.array(saved["ports"][name]) for name in SOURCE_PORTS}
                require(content(bf_bits(np.concatenate((values["text_in"], values["image_in"]), 1)))
                    == saved["ports"]["R"]["content_sha256"], "source residual/input concat differs")
                require(content(bf_bits(np.concatenate((values["text_out"], values["image_out"]), 1)))
                    == saved["ports"]["out"]["content_sha256"], "source full output/streams differ")
                width = values["R"].shape[-1]
                mod = inputs[role].payload["context"]["single_mod"].reshape(1, 1, 3 * width)
                require(describe(mod[..., 2 * width:])["sha256"] == saved["ports"]["gate"]["content_sha256"], "consumed source gate/modulation differs")
                writer_checks[cid][role]["finite_closure"][lane] = pair_account(
                    bf(values["R"] + bf(values["gate"] * values["projection_return"])), values["out"])
                require(describe({k: v for k, v in inputs[role].payload.items() if k not in ("image", "text", "cursor")})
                    == describe({k: v for k, v in after.payload.items() if k not in ("image", "text", "cursor")}), "source writer changed persistent state")
                source_joins[cid][role][lane] = dict(input_cut=inputs[role].fingerprint, output_cut=after.fingerprint)
        endpoint_checks[cid] = {arm: pub.compare(writers["forecast"][cid]["endpoints"][arm], writers["native"][cid]["endpoints"][arm]) for arm in ARMS}
        full_array_fields += len(ARMS)
        construction_checks[cid] = {}
        for lane in ("forecast", "native"):
            saved = writers[lane][cid]
            q, p = ({name: pub.array(value) for name, value in saved["sources"][role]["ports"].items()} for role in ("Q", "P"))
            require(np.array_equal(bf_bits(q["gate"]), bf_bits(p["gate"])), "P/Q gates differ")
            ntext = q["text_in"].shape[1]
            construction_checks[cid][lane] = {}
            for arm in ARMS:
                meta = saved["constructions"][arm]
                projection = None if arm == "P" else pub.array(meta["projection_return"])
                expected_image, merged, intermediates = endpoint_account(q, p, projection, arm, ntext)
                actual = pub.array(saved["endpoints"][arm])
                if arm != "P":
                    require(content(bf_bits(merged)) == meta["descriptors"]["merge_input"]["content_sha256"], "full mixed concat operand differs")
                    require(content(bf_bits(projection)) == meta["descriptors"]["projection_return"]["content_sha256"], "projection record binding differs")
                    write, out = intermediates
                    intermediate_accounts = {name: dict(hash_equal=content(bf_bits(value)) == meta["descriptors"][name]["content_sha256"])
                        for name, value in (("write", write), ("out", out))}
                else:
                    intermediate_accounts = {}
                construction_checks[cid][lane][arm] = dict(endpoint=pair_account(expected_image, actual), intermediates=intermediate_accounts)
            values = {arm: pub.array(saved["endpoints"][arm]).astype(np.float64) for arm in ("Q", "R", "A", "RA")}
            mixed = values["RA"] - values["R"] - values["A"] + values["Q"]
            construction_checks[cid][lane]["mixed_remainder"] = pair_account(mixed, pub.array(saved["constructions"]["finite_mixed_remainder"]))

    index = read(pub.member("state-index.json"))
    consumers = read(pub.member("consumer-ports.json"))
    planned = {cid + "/" + arm for cid in roots for arm in ARMS}
    require(set(index) == set(forecast["paths"]) == planned and set(consumers) == {"forecast", "native"}
        and all(set(values) == planned for values in consumers.values()), "case/consumer census differs")
    cases, images, act_checks, equation_reports, native_values, consumer_layouts, consumer_hashes = {}, {}, {}, {}, {}, {}, {}
    boundary_counts = dict(paired_rows=0, tensors=0, scalar_fields=0, hash_mismatches=0, layout_mismatches=0, value_mismatches=0, scalar_mismatches=0)

    def act(record, operand):
        receipt, after = record["receipt"], cut(record["after"])
        require(digest({k: v for k, v in receipt.items() if k != "fingerprint"}) == receipt["fingerprint"], "Act receipt changed")
        require(receipt["result"] == after.fingerprint and receipt["parent"] == after.parent
            and receipt["act"]["reads"] == receipt["act"]["writes"] == ["image"] and not receipt["act"]["invalidates"], "Act address/lineage differs")
        before = store.load(receipt["parent"], device="cpu")
        common = cut(record["common"])
        require(before.parent == common.fingerprint and describe(before.payload) == describe(common.payload), "fork/common parent closure differs")
        require(describe({k: v for k, v in before.payload.items() if k != "image"})
            == describe({k: v for k, v in after.payload.items() if k != "image"}), "image Act changed another slot")
        require(receipt["writes"] == describe({"image": after.payload["image"]})
            and receipt["act"]["preconditions"] == {"image": digest(describe(before.payload["image"]))}, "Act typed writes/preconditions differ")
        loaded = pub.array(operand, raw=True)
        require(loaded.flags.c_contiguous, "endpoint NPY reload layout differs")
        loaded_stride = [stride // loaded.itemsize for stride in loaded.strides]
        parameters = patch_parameter_account(receipt, describe(before.payload["image"]), describe(after.payload["image"]), operand, loaded_stride)
        join(after.payload["image"], operand)
        parameters["landed_and_producer_endpoint_layout_exact"] = True
        return after, dict(parent=before.fingerprint, common=common.fingerprint, after=after.fingerprint, slot="image", parameters=parameters)

    def row_join(payload, fields):
        if isinstance(payload, torch.Tensor):
            actual = describe(payload)
            require(all(actual[name] == fields["tensor"][name] for name in ("sha256", "shape", "dtype", "stride")), "boundary/final CAS tensor differs")
            values = numpy_tensor(payload).astype(np.float64)
            require(fields["elements"] == values.size and fields["witness_values"] == values.reshape(-1)[fields["witness_indices"]].tolist(), "boundary sampled values differ from CAS")
        elif isinstance(payload, dict):
            require(set(payload) == set(fields), "boundary/CAS slots differ")
            for name in payload:
                row_join(payload[name], fields[name])
        elif isinstance(payload, (tuple, list)):
            require(len(payload) == len(fields), "boundary/CAS sequence differs")
            for a, b in zip(payload, fields, strict=True):
                row_join(a, b)
        else:
            require(type(payload) is type(fields) and payload == fields, "boundary/CAS scalar differs")

    for key, case in index.items():
        cid, arm = case["context"], case["arm"]
        require(key == cid + "/" + arm and case["forecast_final"] == forecast["paths"][key]["forecast_final"], "sealed case identity differs")
        branch = root / case["root"]
        rows = {lane: read(pub.member(case["root"] + "/" + lane + "-rows.json")) for lane in ("forecast", "native")}
        require(all(len(values) == 102 and [row["number"] for row in values] == list(range(102)) for values in rows.values()), "full 102-boundary census differs")
        boundary_result = []
        for number, (f, n) in enumerate(zip(rows["forecast"], rows["native"], strict=True)):
            if number:
                require(f["event"] == n["event"] == {name: rows["native"][number-1]["fields"][name] for name in ("step", "phase", "cursor")}, "boundary event clock differs")
            account = boundary_account(f["fields"], n["fields"])
            boundary_counts["paired_rows"] += 1
            for name in ("tensors", "scalar_fields"):
                boundary_counts[name] += account[name]
            for name in ("hash_mismatches", "layout_mismatches", "value_mismatches", "scalar_mismatches"):
                boundary_counts[name] += len(account[name])
            if any(account[name] for name in ("hash_mismatches", "layout_mismatches", "value_mismatches", "scalar_mismatches")):
                boundary_result.append(dict(number=number, **account))
        consumer_layouts[key] = {lane: port_account(pub, consumers[lane][key], CONSUMER_PORTS) for lane in consumers}
        consumer_hashes[key] = descriptor_comparison(consumers["forecast"][key]["descriptors"], consumers["native"][key]["descriptors"])
        consumer_comparisons = {name: pub.compare(consumers["forecast"][key]["ports"][name], consumers["native"][key]["ports"][name]) for name in CONSUMER_PORTS}
        full_array_fields += len(CONSUMER_PORTS)
        decoders = {lane: read(pub.member(case["root"] + "/" + filename)) for lane, filename in
            (("forecast", "decoder-forecast.json"), ("native", "decoder-native-reference.json"))}
        decoder_comparisons = {name: pub.compare(decoders["forecast"][name], decoders["native"][name], base_a=branch, base_b=branch)
                               for name in ("decoded", "pixels", "decoder_input")}
        full_array_fields += 3
        lane_values, act_checks[key] = {}, {}
        for lane in ("forecast", "native"):
            entry = read(pub.member(case["root"] + "/" + lane + "-entry-act.json"))
            source_state = cut(writers[lane][cid]["sources"]["P" if arm == "P" else "Q"]["state"])
            if arm == "P":
                require(entry["reference"] == "natural-P" and entry["receipt"] is None
                    and entry["source"] == writers[lane][cid]["sources"]["P"]["state"], "P must be a natural full-state reference")
                start = cut(entry["after"])
                require(describe(start.payload) == describe(source_state.payload), "natural P full payload differs")
                act_checks[key][lane] = dict(reference="natural-P", after=start.fingerprint, source=source_state.fingerprint)
            else:
                start, act_checks[key][lane] = act(entry, writers[lane][cid]["endpoints"][arm])
                common = cut(entry["common"])
                require(describe(common.payload) == describe(source_state.payload), "Q common/source full payload differs")
                if arm == "Q":
                    require(describe(start.payload) == describe(common.payload), "Q selfcopy changed full payload")
            row_join(start.payload, rows[lane][0]["fields"])
            for stream in ("image", "text"):
                join(start.payload[stream], consumers[lane][key]["ports"][stream + "_in"])
                port = consumers[lane][key]["ports"][stream + "_out"]
                tensor = rows[lane][1]["fields"][stream]["tensor"]
                require(port["content_sha256"] == tensor["sha256"] and port["shape"] == tensor["shape"]
                    and port["logical_dtype"] == tensor["dtype"] and port["stride"] == tensor["stride"], "first-consumer output/boundary join differs")
            final = cut(case[lane + "_final"])
            require((final.payload["step"], final.payload["phase"], final.payload["cursor"]) == (4, "project", 0), "final horizon differs")
            row_join(final.payload, rows[lane][-1]["fields"])
            lane_values[lane] = {name: pub.array(decoders[lane][name], branch) for name in ("decoded", "pixels")}
            lane_values[lane]["latent"] = numpy_tensor(final.payload["latent"])
            png_record = decoders[lane]["png"]
            png = pub.member((branch / png_record["path"]).relative_to(root).as_posix())
            require(sha(png) == png_record["sha256"], "PNG/decoder binding differs")
            with Image.open(png) as image:
                require(image.mode == "RGB" and np.array_equal(np.asarray(image), lane_values[lane]["pixels"]), "PNG pixels differ from saved full RGB")
            target = output / "images" / (key.replace("/", "--") + "--" + lane + ".png")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(png.read_bytes())
            images[key + "/" + lane] = dict(path=str(target.relative_to(output)), sha256=sha(target), bytes=target.stat().st_size)
        latent = pair_account(lane_values["forecast"]["latent"], lane_values["native"]["latent"])
        decoder_comparisons["latent"] = latent
        full_array_fields += 1
        native_values[key] = lane_values["native"]
        cases[key] = dict(decoder=decoder_comparisons, consumer_ports=consumer_comparisons,
            entry_endpoint=endpoint_checks[cid][arm], boundary_misses=boundary_result,
            png_byte_exact=decoders["forecast"]["png"]["sha256"] == decoders["native"]["png"]["sha256"])
        equation_reports[key] = dict(writer_exit=compare_equations(
            {"image.exit": pub.array(writers["forecast"][cid]["endpoints"][arm])},
            {"image.exit": pub.array(writers["native"][cid]["endpoints"][arm])}, max_elements=4_000_000),
            continuation=compare_equations(lane_values["forecast"], lane_values["native"], max_elements=4_000_000))
    observations = {}
    for cid in roots:
        observations[cid] = {}
        for field in ("latent", "decoded", "pixels"):
            values = {arm: native_values[cid + "/" + arm][field] for arm in ARMS}
            account = axis_accounts(values)
            mixed = sum(sign * np.asarray(values[arm], dtype=np.float64) for sign, arm in ((1, "RA"), (-1, "R"), (-1, "A"), (1, "Q")))
            account["mixed_remainder"] = metric(mixed, np.zeros_like(mixed))
            observations[cid][field] = account
    numerical_misses = [dict(case=key, kind=kind, field=name) for key, case in cases.items()
        for kind in ("decoder", "consumer_ports") for name, account in case[kind].items() if not account["raw_bit_exact"]]
    result = dict(schema=SCHEMA, native_job_id=config["native_job_id"], native_request_sha256=config["native_request_sha256"],
        native_manifest_sha256=config["native_manifest_sha256"], source_worker_sha256=request["source_hashes"]["flux_parallel_family_campaign.py"],
        contexts=list(roots), context_panels={cid: value["panel"] for cid, value in contexts.items()},
        arms=list(ARMS), cases=cases, writer_checks=writer_checks, source_joins=source_joins,
        endpoint_checks=endpoint_checks, construction_checks=construction_checks, act_checks=act_checks,
        writer_layouts=layout_checks, writer_live_descriptors=live_checks, consumer_layouts=consumer_layouts,
        consumer_live_descriptors=consumer_hashes, boundary_counts=boundary_counts, observations=observations,
        images=images, equation_reports=equation_reports, transaction=transaction, source_transaction=prefix_tx,
        source_provenance=provenance, native_counts=expected, external_sealed_files=len(forecast["external_files"]),
        frozen_rule=rule,
        published_files_verified=len(pub.files), prefix_files_verified=len(original.files), full_array_fields_compared=full_array_fields,
        unique_arrays_read=len(pub.arrays), repeated_array_elements_read=pub.elements, numerical_bit_misses=numerical_misses,
        writer_bit_misses=[dict(context=cid, role=role, port=name) for cid, roles in writer_checks.items()
            for role, account in roles.items() for name, value in account["comparisons"].items() if not value["raw_bit_exact"]],
        endpoint_bit_misses=[dict(context=cid, arm=arm) for cid, arms in endpoint_checks.items()
            for arm, account in arms.items() if not account["raw_bit_exact"]],
        mechanics_status="verified", instrument_status="independent complete saved-array/CAS/hash reader", terminal_status="unassessed",
        model_forwards=0, cuda_initialized=False, original_parameters_exported=False,
        limitations=["Original weight projection products are paid and supplied; no weights or learned forwards in this reader.",
            "Boundary coverage is complete tensor hashes/layouts and all supplied RMS/means/up-to16 samples, not retained full arrays at every boundary.",
            "Hash-only ports cannot localize a numeric miss; live and retained snapshot layouts are separately classified.",
            "Native A is the attention-feature slice of the actual merge operand; its stride may differ from the functional flattened SDPA return.",
            "Signed-axis rank, source distance, orthogonal error and numerical exactness are separate observations; zero axes remain inapplicable.",
            "No semantic classifier or quality threshold; dependent arms and overlapping fields are not independent samples."])
    save(output / "audit.json", result)
    save(output / "observations.json", observations)
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        (output / name).write_bytes(pub.member(name).read_bytes())
    return result


audit = run


def worker(config_path):
    config = read(config_path)
    outer = Path(os.environ.get("SATURN_DEBUG_REQUEST", "saturn_debug_request.json"))
    if outer.exists():
        request = read(outer)
        require(request_identity(request) == request["request_sha256"] and request["config"] == config, "audit request/config differs")
        for name, expected in request["source_hashes"].items():
            require(sha(Path(config_path).parent / name) == expected, "audit payload source differs")
    config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / RESULT)
    result = run(config)
    from mdb.job_support import request_sha256

    publication = dict(output_root=config["output"], audit_sha256=sha(Path(config["output"]) / "audit.json"),
        source_job=config["native_job_id"], config_sha256=request_sha256(read(config_path)))
    print("MDB_FAMILY_AUDIT_READY " + json.dumps(publication, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
