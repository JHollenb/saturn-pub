"""Saved-only gauge/spectral campaign audit; no model or learned operation runs.

Complete retained arrays and hash-only intermediate coverage remain separate.
Numerical discrepancies are observations, never scientific acceptance gates.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
from collections import Counter
from pathlib import Path

import numpy as np

WORKER_SHA256 = "4493cd2cbf2ccf4b5d761540def7caa842871b172a3aa9881cc8be41daeb0551"
SCHEMA = "mdb-flux-gauge-spectral-v1"
RESULT = "astra-flux-audit-result"
HELPERS = ()
PINS = {}


def make_config(base):
    return dict(base, output_name=RESULT, audit_worker_sha256=sha(__file__), model_forwards=0,
        normalization_dependency="only two transient original BN stat payloads and authenticated archive header",
        max_output_bytes=10 << 20)


def validate_config(config):
    for key in ("native_root", "native_manifest_sha256", "native_request_sha256", "output_scratch"):
        require(bool(config.get(key)), "missing saved audit config: " + key)
    require(config.get("output_name") == RESULT and config.get("audit_worker_sha256") == sha(__file__)
        and config.get("model_forwards") == 0 and config.get("max_output_bytes") == 10 << 20, "saved audit execution contract differs")
    return config


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def member(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts and relative.parts, "unsafe evidence member")
    path = Path(root) / relative
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "missing/linked evidence member: " + name)
    return path


def array(root, record):
    path = member(root, record["path"])
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "saved array file differs")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == record["shape"] and str(value.dtype) == record.get("stored_dtype", record.get("dtype")), "saved array geometry differs")
    if "content_sha256" in record:
        require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == record["content_sha256"], "saved array payload differs")
    if record.get("encoding") == "bf16-bits-v1":
        require(value.dtype == np.uint16, "BF16 storage differs")
        value = (value.astype(np.uint32) << 16).view(np.float32)
    require(np.isfinite(value).all(), "nonfinite saved array")
    return value


def metric(left, right):
    left, right = np.ascontiguousarray(left), np.ascontiguousarray(right)
    require(left.shape == right.shape, "comparison shape differs")
    delta = left.astype(np.float64) - right.astype(np.float64)
    numeric = np.flatnonzero(delta.ravel() != 0)
    bit = np.flatnonzero(np.any(left.view(np.uint8).reshape(left.size, left.dtype.itemsize) != right.view(np.uint8).reshape(right.size, right.dtype.itemsize), axis=1)) if left.dtype == right.dtype else None
    return dict(shape=list(left.shape), elements=left.size, rms=float(np.sqrt(np.mean(delta * delta))),
        max_abs=float(np.max(np.abs(delta))), signed_mean=float(np.mean(delta)), changed=int(numeric.size),
        first_difference=None if not numeric.size else dict(index=int(numeric[0]), left=float(left.ravel()[numeric[0]]), right=float(right.ravel()[numeric[0]])),
        bit_exact=None if bit is None else not bool(bit.size), changed_bits=None if bit is None else int(bit.size),
        signed_zero_differences=int(np.count_nonzero((left == 0) & (right == 0) & (np.signbit(left) != np.signbit(right)))))


def scalar_rows(value):
    value = np.asarray(value, dtype=np.float64)
    return dict(shape=list(value.shape), elements=value.size, dtype="float64", values=value.ravel().tolist(),
        sha256=hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(), rms=float(np.sqrt(np.mean(value * value))),
        signed_mean=float(value.mean()), min=float(value.min()), max=float(value.max()))


def signed(left, right):
    value = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return dict(shape=list(value.shape), elements=value.size, dtype="float64", sha256=hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
        rms=float(np.sqrt(np.mean(value * value))), signed_mean=float(value.mean()), max_abs=float(np.abs(value).max()), changed=int(np.count_nonzero(value)))


def verify_signed_identity(actual, sealed):
    require(all(actual[key] == sealed[key] for key in ("shape", "elements", "dtype", "sha256")), "presealed complete signed-response bytes differ")
    return dict(sealed_reduction={k: sealed[k] for k in ("rms", "signed_mean", "max_abs", "changed")},
        independently_reduced={k: actual[k] for k in ("rms", "signed_mean", "max_abs", "changed")},
        scope="Exact FP64 delta byte hash; cross-library reduction rounding is an observation, not byte custody.")


def sealed_scalar(record):
    value = np.asarray(record["values"], dtype=np.float64).reshape(record["shape"])
    require(value.size == record["elements"] and record["dtype"] == "float64" and np.isfinite(value).all()
        and hashlib.sha256(value.astype("<f8").tobytes()).hexdigest() == record["sha256"], "presealed scalar row bytes differ")
    return value


def rne(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def tensor_array(value):
    return value.detach().float().numpy() if str(value.dtype) == "torch.bfloat16" else value.detach().numpy()


def leaves(value, prefix=""):
    if hasattr(value, "detach"):
        return {prefix or "/": value}
    if isinstance(value, dict):
        return {path: item for key, child in value.items() for path, item in leaves(child, prefix + "/" + str(key)).items()}
    if isinstance(value, (list, tuple)):
        return {path: item for key, child in enumerate(value) for path, item in leaves(child, prefix + "/" + str(key)).items()}
    return {}


def boundary_leaves(value, prefix=""):
    if isinstance(value, dict) and "tensor" in value:
        return {prefix or "/": value}
    if isinstance(value, dict):
        return {path: item for key, child in value.items() for path, item in boundary_leaves(child, prefix + "/" + str(key)).items()}
    if isinstance(value, list):
        return {path: item for key, child in enumerate(value) for path, item in boundary_leaves(child, prefix + "/" + str(key)).items()}
    return {}


def cut(store, record):
    value = store.load(record["cut"], device="cpu")
    require(value.fingerprint == record["cut"] and value.parent == record["parent"] and value.model_identity == record["model_identity"]
        and dict(value.execution_point) == record["execution_point"], "typed StateCut record differs")
    return value


def verify_boundary(payload, fields):
    from mdb.forward_recording import tensor_descriptor

    full, recorded = leaves(payload), boundary_leaves(fields)
    require(set(full) == set(recorded), "retained boundary field census differs")
    for name, value in full.items():
        entry, desc = recorded[name], tensor_descriptor(value)
        require(list(value.shape) == entry["tensor"]["shape"] and str(value.dtype) == entry["tensor"]["dtype"]
            and desc["sha256"] == entry["tensor"]["sha256"], "retained boundary hash differs: " + name)
        flat = tensor_array(value).ravel()
        require(flat[entry["witness_indices"]].tolist() == entry["witness_values"], "boundary witness differs")
    return full


def original_identity(plan, source, summary, request, wrapper, expected_request):
    """Actual persisted plan is request.config plus three runtime custody fields.

    The producer intentionally strips worker_sha256 from config. Its named
    request source map binds executed code; no invented plan worker key is used.
    """
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == expected_request == request["request_sha256"], "canonical original request differs")
    for document in (plan, source, summary):
        require(document["job_id"] == wrapper["job_id"] and document["request_sha256"] == identity and document["schema"] == SCHEMA, "core original identity differs")
    effective = {k: v for k, v in plan.items() if k not in ("job_id", "request_sha256", "output")}
    require(effective == request["config"] and request["source_hashes"].get("flux_gauge_spectral_campaign.py") == WORKER_SHA256, "executed config/source differs")
    return identity


def authenticate(root, expected_manifest, expected_request, expected_job=None):
    root = Path(root); wrapper = read(root / "files.json")
    require(sha(root / "files.json") == expected_manifest and wrapper["request_sha256"] == expected_request, "original wrapper/request differs")
    require(expected_job is None or wrapper["job_id"] == expected_job, "original job differs")
    inventory = wrapper["files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "files.json"}
    require(actual == set(inventory), "original publication file census differs")
    for name, record in inventory.items():
        path = member(root, name)
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "original publication member differs: " + name)
    require(sum(row["bytes"] for row in inventory.values()) == wrapper["expanded_bytes"], "expanded bytes differ")
    plan, source, summary, native = (read(root / name) for name in ("plan.json", "source.json", "summary.json", "native-contract.json"))
    request = read(root / "request.json")
    original_identity(plan, source, summary, request, wrapper, expected_request)
    require(source["helper_source_sha256"] == plan["helper_source_sha256"] and source["forecast_seal"] == summary["forecast_seal"], "source helper/seal differs")
    require(all(request["source_hashes"].get(name) == value for name, value in source["helper_source_sha256"].items()), "executed helper closure differs")
    for key, record in source["sources"].items():
        require(record["sha256"] == plan[key + "_sha256"] and sha(member(root, record["path"])) == record["sha256"], "original source metadata differs: " + key)
    require(source["parameter_recipe_counts"] == dict(qwen=398, flux=169, vae=142) and source["tied_alias"], "original recipe/alias census differs")
    for component in ("qwen", "flux", "vae"):
        before, after = native["identities_before"][component], native["identities_after"][component]
        require(set(before) == set(after) and after == source["parameter_identities"][component], "resident tensor census differs")
        for name in before:
            require(all(before[name][k] == after[name][k] for k in ("shape", "dtype", "sha256")), "resident tensor changed")
        manifest = read(member(root, source["sources"][component + "_manifest"]["path"]))
        require(len(manifest["artifacts"]) == source["parameter_recipe_counts"][component], "recipe manifest count differs")
        for entry in manifest["artifacts"]:
            name = entry["source"]["tensor"]
            require(name in after and after[name]["shape"] == entry["source"]["shape"] and after[name]["sha256"] == entry["source"]["payload_sha256"], "saved parameter identity/recipe differs: " + name)
            require(not (root / entry["path"]).exists(), "persistent parameter bank unexpectedly retained")
    require(native["counts"] == summary["counts"] and native["observed"] == summary["observed"], "actual census sidecars differ")
    require(all(not row.get("module_entries", 0) for phase, row in native["observed"].items() if phase.startswith("forecast")), "functional lane witnessed module entry")
    require([row["component"] for row in native["placements"] if row["device"] == "cuda"] == ["qwen", "flux", "vae", "flux", "vae"], "single-component GPU placement schedule differs")
    return plan, source, summary, native, inventory


def transaction(root, summary, inventory):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx = root / "forecast-transaction"
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"], expected_event_head=summary["transaction"]["event_head"])
    kinds = [read(p)["kind"] for p in sorted((tx / "events").glob("*.json"))]
    require(kinds == ["opened", "frontier_published", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and verified["phase"] == "closed" and not read(tx / "seal.json")["violations"], "transaction event chain incomplete/violated")
    prediction = load_forecast_tree(tx, read(tx / "forecasts/all-futures.json")["predictions"])
    require(prediction["targets_at_seal"] == 0 and len(prediction["paths"]) == 20, "global forecast panel differs")
    for name, record in prediction["external_files"].items():
        require(name in inventory and record == inventory[name], "pretarget sealed publication member differs: " + name)
    required = {"ff-forecasts.json", "reduced-law-predictions.json", "forecast-responses.json", "spectral-source.json", "ff-applicability.json"}
    require(required <= set(prediction["external_files"]), "pretarget law/response evidence absent")
    for name in inventory:
        if "/forecast-rows.json" in name or "/forecast-acts.json" in name or "/decoder/forecast/" in name:
            require(name in prediction["external_files"], "complete forecast absent from global seal")
    cuts = []
    for context in read(root / "source-frontiers.json")["roots"].values():
        cuts.extend((context["source"]["cut"], context["source"]["parent"]))
    for path in prediction["paths"].values():
        branch = root / path["root"]
        cuts.extend(row["retained"]["cut"] for row in read(branch / "forecast-rows.json") if "retained" in row)
        for record in read(branch / "forecast-acts.json"):
            cuts.extend(record[key]["cut"] for key in ("before", "after"))
    for identity in set(cuts):
        name = "state-store/cuts/" + identity + ".json"
        require(name in prediction["external_files"], "complete forecast cut not sealed")
        record = read(member(root, name))
        members = ["state-store/pages/" + page + ".json" for page in record["slots"].values()]
        members += ["state-store/blobs/" + blob + ".safetensors" for blob in record["physical_blobs"]]
        require(all(name in prediction["external_files"] for name in members), "forecast cut payload closure not sealed")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    require([row["number"] for row in timeline] == list(range(len(timeline))), "timeline order differs")
    sealed = [r for r in timeline if r["kind"] == "all_forecasts_globally_sealed"]
    released = [r for r in timeline if r["kind"] == "native_targets_released"]
    require(len(sealed) == len(released) == 1 and sealed[0]["seconds"] < released[0]["seconds"]
        and sealed[0]["seal_sha256"] == released[0]["seal_sha256"] == summary["forecast_seal"], "global release chronology differs")
    require(sum(r["kind"] == "forecast_suffix_completed" and r["number"] < sealed[0]["number"] for r in timeline) == 20
        and sum(r["kind"] == "native_suffix_completed" and r["number"] > released[0]["number"] for r in timeline) == 20, "completed future chronology differs")
    return dict(verification=verified, event_kinds=kinds, seal=sealed[0], release=released[0], sealed_files=len(prediction["external_files"]),
        sealed_bytes=sum(r["bytes"] for r in prediction["external_files"].values()))


def bn_buffers(root, plan, source):
    """Authorized two-buffer transient retrieval, never a whole-checkpoint scan."""
    manifest = read(member(root, source["sources"]["vae_manifest"]["path"]))
    rows = {r["source"]["tensor"]: r for r in manifest["artifacts"] if r["source"]["tensor"] in ("bn.running_mean", "bn.running_var")}
    require(len(rows) == 2, "two BN recipes unavailable")
    provenance = manifest["provenance"]; file = next(iter(rows.values()))["source"]["file"]
    path = member(plan["vae_parameter_root"], file); before = path.stat(); values, accounts = {}, {}
    with path.open("rb") as stream:
        prefix = stream.read(8); size = struct.unpack("<Q", prefix)[0]
        require(size == provenance["header_bytes"] and size <= 1 << 20, "BN archive header size differs")
        raw_header = stream.read(size)
        require(hashlib.sha256(raw_header).hexdigest() == provenance["header_sha256"], "BN archive header differs")
        header = json.loads(raw_header)
        for name, row in rows.items():
            spec = row["source"]; require(spec["dtype"] == "BF16" and spec["shape"] == [128] and header[name] == {k: spec[k] for k in ("dtype", "shape", "data_offsets")}, "BN header tensor differs")
            lo, hi = spec["data_offsets"]; require(hi - lo == 256, "BN payload extent differs")
            stream.seek(8 + size + lo); payload = stream.read(hi - lo)
            require(hashlib.sha256(payload).hexdigest() == spec["payload_sha256"]
                and hashlib.sha256(base64.b64decode(row["npy_prefix_b64"]) + payload).hexdigest() == row["sha256"], "BN payload/virtual NPY differs")
            values[name] = (np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16).view(np.float32).copy()
            accounts[name] = dict(recipe=row, payload_bytes=len(payload), header_tensor_verified=True)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "BN source mutated during transient read")
    return values, dict(status="independently-retrieved-two-stat-buffers", decoded_payload_bytes=512, archive_header_bytes=size,
        header_sha256=provenance["header_sha256"], buffers=accounts, whole_checkpoint_hash_recomputed=False,
        declared_checkpoint_sha256=provenance["checkpoint_sha256"], persistent_copies=0,
        retriever=source["sources"]["vae_retrieval_script"], scope="new authorized audit dependency; not originally retained producer buffer bytes")


def decoder_argument(payload, bn, epsilon, height, width, scale):
    z, ids = tensor_array(payload["latent"]), tensor_array(payload["img_ids"])
    h, w = height // (scale * 2), width // (scale * 2)
    indices = (ids[0, :, 1] * w + ids[0, :, 2]).astype(np.int64)
    require(z.shape[0] == 1 and sorted(indices.tolist()) == list(range(h * w)), "packed latent IDs do not cover grid")
    unpacked = np.empty((h * w, z.shape[-1]), dtype=np.float32); unpacked[indices] = z[0]
    unpacked = unpacked.reshape(h, w, -1).transpose(2, 0, 1)[None]
    mean, var = (bn["bn.running_" + name].reshape(1, -1, 1, 1) for name in ("mean", "var"))
    std = rne(np.sqrt(rne(var + np.float32(epsilon))))
    normalized = rne(rne(unpacked * std) + mean)
    b, c, h, w = normalized.shape
    return normalized.reshape(b, c // 4, 2, 2, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def ff_accounts(root):
    forecast, native = read(root / "ff-forecasts.json"), read(root / "ff-native.json")
    reduced, spectral, applicability = (read(root / name) for name in ("reduced-law-predictions.json", "spectral-source.json", "ff-applicability.json"))
    require(applicability["linear_in_bias_none"] and applicability["linear_out_bias_none"] and reduced["native_targets_at_formation"] == 0, "source FF applicability differs")
    c = array(root, spectral["coordinate"]); output = {}
    for context, directions in forecast.items():
        account = read(root / "writer" / context / "account.json"); x = array(root, account["original"]["x"])
        rows = {}
        for direction, records in directions.items():
            n = native[context][direction]; law = reduced["roots"][context]["directions"][direction]
            vector = array(root, spectral["vectors"][direction]); norm2 = np.square(x.astype(np.float64)).sum(-1)
            ideal = norm2 * law["rayleigh"]
            f_even, n_even = array(root, records["even"]), array(root, n["even"])
            f_scalar, n_scalar = (np.sum(value.astype(np.float64) * c, axis=-1) for value in (f_even, n_even))
            reductions = {key: metric(value, sealed_scalar(law[key])) for key, value in (("source_input_norm_squared", norm2), ("ideal_unrounded", ideal), ("full_finite_forecast", f_scalar), ("ideal_minus_full_finite", ideal - f_scalar))}
            # The original scalar arrays are the prospective predictions. NumPy
            # recomputation of FP64 reductions is independently reported, not
            # required to reproduce Torch/CUDA reduction rounding bit for bit.
            ideal, f_scalar = sealed_scalar(law["ideal_unrounded"]), sealed_scalar(law["full_finite_forecast"])
            operands = {}
            for sign in ("plus", "minus"):
                f = records[sign]["input"]; consumed = n[sign]["actual_input"]
                require(n[sign]["source_input"] == f and all(consumed[k] == f[v] for k, v in (("shape", "shape"), ("stride", "stride"), ("dtype", "logical_dtype"), ("device", "device_at_capture"), ("sha256", "content_sha256"))), "native FF consumed operand differs")
                operands[sign] = metric(array(root, records[sign]["output"]), array(root, n[sign]["output"]))
            plus, minus = (array(root, records[s]["output"]) for s in ("plus", "minus"))
            require(metric(rne(rne(plus + minus) * np.float32(.5)), f_even)["bit_exact"], "forecast even-write saved arithmetic differs")
            plus, minus = (array(root, n[s]["output"]) for s in ("plus", "minus"))
            require(metric(rne(rne(plus + minus) * np.float32(.5)), n_even)["bit_exact"], "native even-write saved arithmetic differs")
            positive, negative = (array(root, records[s]["input"]) for s in ("plus", "minus"))
            for key, value in (("actual_positive_operand_norm_squared", positive), ("actual_negative_operand_norm_squared", negative)):
                reductions[key] = metric(np.square(value.astype(np.float64)).sum(-1), sealed_scalar(law[key]))
            rows[direction] = dict(ideal=scalar_rows(ideal), forecast=scalar_rows(f_scalar), native=scalar_rows(n_scalar),
                ideal_minus_forecast=metric(ideal, f_scalar), forecast_minus_native=metric(f_scalar, n_scalar), ideal_minus_native=metric(ideal, n_scalar),
                sign_disagreements_ideal_native=int(np.count_nonzero(np.sign(ideal) != np.sign(n_scalar))),
                full_even_error=metric(f_even, n_even), original_ff_output_errors=operands, antipodal_operands=metric(negative, -positive),
                independent_scalar_reduction_errors=reductions, sealed_scalar_predictions=law,
                direction_norm_squared=float(np.square(vector).sum()), source_ritz=spectral["directions"][direction])
            arm = direction + "-even-add"
            if arm in account.get("endpoints", {}):
                residual, gate, base = (array(root, account["original"][key]) for key in ("R_attention", "gate_ff", "M_base"))
                formed = rne(residual + rne(gate * rne(base + rne(np.float32(.25) * f_even))))
                rows[direction]["sealed_endpoint_original_order"] = metric(formed, array(root, account["endpoints"][arm]))
        arrays = {key: np.stack([np.asarray(rows[d][key]["values"]) for d in ("top", "bottom", "random")], axis=-1) for key in ("ideal", "forecast", "native")}
        order = {key: np.argsort(-value, axis=-1, kind="stable").tolist() for key, value in arrays.items()}
        rows["ordering"] = dict(descending=order, ideal_native_rows_different=int(np.count_nonzero(np.any(np.asarray(order["ideal"]) != np.asarray(order["native"]), axis=-1))), scope="observed heterogeneous row ordering; no ordering gate")
        output[context] = rows
    return dict(roots=output, applicability=applicability, source_spectral_account=spectral,
        scope="ideal unrounded source-factor law versus original-order finite FF forecasts and actual native operand assays; no learned recomputation")


def lineage_accounts(root, store, roots, state_index, emissions):
    joined, splices = 0, []
    for context, source in roots.items():
        parent = cut(store, source["source"])
        prefix = [row for row in emissions if row["kind"] == "source-project" and row["context"] == context]
        require(len(prefix) == 1 and prefix[0]["receipt"]["result"] == parent.fingerprint and prefix[0]["receipt"]["parent"] == parent.parent, "actual source project cut differs")
        front = read(root / "inputs" / context / "frontend.json")
        payload = parent.payload
        for slot, field in (("latent", "packed_noise"), ("img_ids", "img_ids"), ("txt_ids", "txt_ids"), ("timesteps", "timesteps"), ("sigmas", "sigmas")):
            require(metric(tensor_array(payload[slot]), array(root, front[field]))["bit_exact"], "native project source collateral differs")
        memo = read(root / "conditioning-memo.json")[front["conditioning_memo"][1:]]
        require(metric(tensor_array(payload["conditioning"]), array(root, memo["conditioning"]))["bit_exact"], "immutable conditioning memo differs")
        for arm in source["arms"]:
            branch = root / state_index[context + "/" + arm]["root"]
            rows = read(branch / "native-rows.json"); previous = rows[0]["retained"]["cut"]
            occurrences = [row for row in emissions if row["kind"] == "native" and row["context"] == context and row["arm"] == arm]
            require(len(occurrences) == len(rows) - 1, "native physical occurrence count differs")
            for row, receipt in zip(rows[1:], occurrences):
                require(row["receipt"] == receipt["receipt"]["fingerprint"], "native occurrence receipt differs")
                actual_parent = receipt["receipt"]["parent"]
                if actual_parent != previous:
                    require(arm == "baseline" and row["number"] == 1, "native own-state receipt chain differs")
                    common = read(member(root, "state-store/cuts/" + previous + ".json"))
                    consumed = read(member(root, "state-store/cuts/" + actual_parent + ".json"))
                    proof = fork_clone_proof(common, consumed)
                    splices.append(dict(context=context, arm=arm, row=row["number"], native_receipt=receipt["receipt"]["fingerprint"], **proof))
                previous = receipt["receipt"]["result"]; joined += 1
                if "retained" in row:
                    require(row["retained"]["cut"] == previous, "retained native return differs from actual advance")
            require(previous == state_index[context + "/" + arm]["native_final"], "native final receipt/cut differs")
            if arm == "baseline":
                rewind = read(branch / "native-rewind.json"); receipt_valid(rewind["receipt"])
                require(rewind["before"]["cut"] == previous and rewind["after"]["cut"] == parent.fingerprint
                    and rewind["receipt"]["parent"] == previous and rewind["receipt"]["result"] == parent.fingerprint
                    and rewind["receipt"]["verified_exact"], "actual completed-history restore differs")
    return dict(native_successive_advances_joined=joined, source_projects_joined=4, completed_native_restores=4,
        typed_fork_splices=splices, typed_fork_splice_count=len(splices),
        scope="lane/context/arm/ordered physical occurrences and actual receipt parents/results; baseline first entry can use an exactly verified same-payload reparented public Session.fork cut; original row IDs/receipts are preserved; component forecasts create no native advance receipts")


def fork_clone_proof(common, consumed):
    """Verify public Session.fork's typed clone, not generic state similarity."""
    from saturn_pub.values import digest

    require(digest(common["manifest"]) == common["fingerprint"] and digest(consumed["manifest"]) == consumed["fingerprint"], "fork cut manifest fingerprint differs")
    require(consumed["logical_parent"] == consumed["manifest"]["parent"] == common["fingerprint"], "fork cut parent is not the shared common cut")
    require({k:v for k,v in common["manifest"].items() if k != "parent"} == {k:v for k,v in consumed["manifest"].items() if k != "parent"}, "fork cut changes typed state/execution closure")
    require(common["slots"] == consumed["slots"] and common["physical_blobs"] == consumed["physical_blobs"]
        and common["execution_point"] == consumed["execution_point"] and common["surface"] == consumed["surface"], "fork cut changes CAS closure/layout")
    return dict(common_cut=common["fingerprint"], consumed_fork_cut=consumed["fingerprint"], fork_parent=consumed["logical_parent"],
        full_manifest_except_parent_exact=True, slot_pages=common["slots"], physical_blobs=common["physical_blobs"],
        execution_point=common["execution_point"], scope="public Session.fork(cut) creates a child cut with the same complete typed payload/execution/surface; fingerprint differs because parent differs")


def receipt_statistics(emissions):
    fingerprints = Counter(row["receipt"]["fingerprint"] for row in emissions)
    return dict(physical_occurrences=len(emissions), distinct_fingerprints=len(fingerprints),
        repeated_fingerprint_occurrences=sum(n - 1 for n in fingerprints.values()),
        scope="Physical emission count is independent of fingerprint equality; identical same-parent Acts/replays remain separate occurrences.")


def receipt_valid(record):
    body = {k: v for k, v in record.items() if k != "fingerprint"}
    require(hashlib.sha256(canonical(body)).hexdigest() == record["fingerprint"], "public receipt fingerprint differs")


def act_accounts(root, store, branch, arm, lane):
    from saturn_pub.values import describe

    output = []
    for record in read(branch / (lane + "-acts.json")):
        receipt_valid(record["receipt"])
        before, after = (cut(store, record[key]) for key in ("before", "after")); slot = record["slot"]
        require(record["receipt"]["parent"] == before.fingerprint and record["receipt"]["result"] == after.fingerprint, "Act before/after receipt differs")
        a, b = before.payload, after.payload
        require(describe({k: v for k, v in a.items() if k != slot}) == describe({k: v for k, v in b.items() if k != slot}) and record["all_other_slots_unchanged"], "Act collateral slot changed")
        old, new = tensor_array(a[slot]), tensor_array(b[slot])
        if arm in ("gauge", "mean-zero-control"):
            offset = np.where(np.arange(old.shape[1]) % 2 == 0, .5, -.5).reshape(1, -1, 1)
            if arm == "mean-zero-control":
                offset = offset * np.where(np.arange(old.shape[-1]) % 2 == 0, 1, -1)
            expected = rne(old + offset)
        elif arm == "sham":
            expected = old
        else:
            account = read(root / "writer" / branch.parent.name / "account.json")
            expected = array(root, account["endpoints"][arm])
        require(metric(new, expected)["bit_exact"], "declared public endpoint replacement differs")
        delta = new.astype(np.float64) - old.astype(np.float64)
        output.append(dict(slot=slot, before=before.fingerprint, after=after.fingerprint, receipt=record["receipt"]["fingerprint"],
            change=metric(new, old), centered_leakage_rms=float(np.sqrt(np.mean(np.square(delta - delta.mean(-1, keepdims=True))))), other_slots_exact=True))
    return output


def state_accounts(root, store, branch):
    frows, nrows = (read(branch / (lane + "-rows.json")) for lane in ("forecast", "native"))
    require(len(frows) == len(nrows) == 111 and [r["number"] for r in frows] == list(range(111)) == [r["number"] for r in nrows], "complete logical boundary census differs")
    output = []
    for f, n in zip(frows, nrows):
        require(f.get("event") == n.get("event"), "functional/native logical clocks differ")
        fl, nl = boundary_leaves(f["fields"]), boundary_leaves(n["fields"])
        require(set(fl) == set(nl), "boundary field sets differ")
        require(all(fl[k]["tensor"]["shape"] == nl[k]["tensor"]["shape"] and fl[k]["tensor"]["dtype"] == nl[k]["tensor"]["dtype"] for k in fl), "boundary comparison geometry differs")
        hashes = {k: dict(elements=fl[k]["elements"], value_hash_exact=fl[k]["tensor"]["sha256"] == nl[k]["tensor"]["sha256"],
            layout_exact=fl[k]["tensor"]["stride"] == nl[k]["tensor"]["stride"]) for k in fl}
        for k in hashes:
            if not hashes[k]["layout_exact"]:
                hashes[k].update(forecast_stride=fl[k]["tensor"]["stride"], native_stride=nl[k]["tensor"]["stride"])
        full = None
        if "retained" in f and "retained" in n:
            a, b = cut(store, f["retained"]), cut(store, n["retained"])
            fa, na = verify_boundary(a.payload, f["fields"]), verify_boundary(b.payload, n["fields"])
            full = {k: metric(tensor_array(fa[k]), tensor_array(na[k])) for k in fa}
        output.append(dict(number=f["number"], event=f.get("event"), hashes=hashes, full_errors=full,
            scope="complete retained arrays" if full is not None else "complete tensor hashes/layout; no full numeric norm on differing unretained tensors"))
    return output


def image_record(branch, record):
    from PIL import Image

    decoded, pixels = (array(branch, record[key]) for key in ("decoded", "pixels"))
    expected = np.round(np.clip(rne(rne(decoded * np.float32(.5)) + np.float32(.5)), 0, 1).transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)
    require(np.array_equal(expected, pixels), "raw RGB postprocess/pixel bytes differ")
    png = member(branch, record["png"]["path"])
    require(sha(png) == record["png"]["sha256"] and png.stat().st_size == record["png"]["bytes"], "original PNG custody differs")
    with Image.open(png) as image:
        require(np.array_equal(np.asarray(image), pixels), "PNG pixel consumer differs")
    return decoded, pixels


def gauge_ports(root, context, arm, inventory):
    if arm not in ("gauge", "mean-zero-control"):
        return None
    branch = root / "cases" / context / arm
    selected = read(branch / "joint0-account-forecast.json")
    base = read(root / "writer" / context / "account.json")["original"]
    native_base = read(root / "writer" / context / "native-account.json")
    result = {}
    for name in ("image_ln", "text_ln", "x", "M_base", "R_attention"):
        relative = "writer/" + context + "/native-" + arm + "/" + name + ".npy"
        # The frozen passive writer archives these arrays without a JSON sidecar.
        # Their file identity is authenticated by the full original wrapper.
        path = member(root, relative); stored = np.load(path, allow_pickle=False)
        require(stored.dtype == np.uint16 and sha(path) == inventory[relative]["sha256"], "passive native writer array differs")
        actual = (stored.astype(np.uint32) << 16).view(np.float32)
        f, fbase, nbase = array(branch, selected[name]), array(root, base[name]), array(root, native_base[name])
        result[name] = dict(forecast_vs_native=metric(f, actual), forecast_change_from_own_base=metric(f, fbase), native_change_from_own_base=metric(actual, nbase),
            native_array=relative, scope="LayerNorm gauge ideal zero applies to pre-modulation LayerNorm only; finite BF16 and later ports observed")
    return result


def audit(root, output, *, expected_manifest, expected_request, expected_job=None, publication_id=None):
    from saturn_pub.store import LocalStore

    root, output = Path(root), Path(output)
    plan, source, summary, native, inventory = authenticate(root, expected_manifest, expected_request, expected_job)
    chronology = transaction(root, summary, inventory)
    store = LocalStore(root / "state-store")
    state_index, roots = read(root / "state-index.json"), read(root / "source-frontiers.json")["roots"]
    require(len(roots) == 4 and len(state_index) == 20, "context/arm panel differs")
    try:
        bn, bn_account = bn_buffers(root, plan, source)
    except (OSError, ValueError, KeyError) as error:
        bn, bn_account = None, dict(status="evidence-incomplete", reason=str(error), scope="only independent latent normalization subcheck; other evidence retained")
    spec = read(member(root, source["sources"]["vae_config"]["path"]))
    responses = read(root / "forecast-responses.json")
    cases = {}
    for identifier, context in roots.items():
        base = root / state_index[identifier + "/baseline"]["root"]
        bf, bn_record = (read(base / name) for name in ("decoder-forecast.json", "decoder-native-reference.json"))
        base_f, base_n = image_record(base, bf), image_record(base, bn_record)
        for arm in context["arms"]:
            key = identifier + "/" + arm; path = state_index[key]; branch = root / path["root"]
            f, n = (read(branch / name) for name in ("decoder-forecast.json", "decoder-native-reference.json"))
            f_images, n_images = image_record(branch, f), image_record(branch, n)
            images = {name: metric(a, b) for name, a, b in zip(("decoded", "pixels"), f_images, n_images)}
            effects = {name: dict(forecast=signed(a, fa), native=signed(b, na), response_error=metric(a.astype(np.float64) - fa, b.astype(np.float64) - na))
                for name, a, b, fa, na in zip(("decoded", "pixels"), f_images, n_images, base_f, base_n)}
            if arm != "baseline":
                response_reductions = {}
                for name in ("decoded", "pixels"):
                    effects[name]["sealed_response_reductions"] = verify_signed_identity(effects[name]["forecast"], responses[key]["decoder"][name])
                for entry in responses[key]["full_retained_states"]:
                    a, b = (store.load(entry[k], device="cpu").payload for k in ("left_cut", "right_cut"))
                    al, bl = leaves(a), leaves(b)
                    actual = {k: signed(tensor_array(al[k]), tensor_array(bl[k])) for k in al}
                    require(set(actual) == set(entry["fields"]), "presealed signed-response field census differs")
                    response_reductions[str(entry["number"])] = {k: verify_signed_identity(actual[k], entry["fields"][k]) for k in al}
            else:
                response_reductions = {}
            joins = {}
            for lane, record in (("forecast", f), ("native", n)):
                payload = store.load(path[lane + "_final"], device="cpu").payload
                require(payload["step"] == 4 and payload["phase"] == "project" and payload["cursor"] == 0, "final state clock differs")
                argument = array(branch, record["decoder_input"])
                if bn is not None:
                    rebuilt = decoder_argument(payload, bn, spec.get("batch_norm_eps", 1e-5), plan["height"], plan["width"], 2 ** (len(spec["block_out_channels"]) - 1))
                    joins[lane + "_latent_to_argument"] = metric(rebuilt, argument)
                if lane == "native":
                    joins["actual_before_post_quant"] = metric(argument, array(branch, record["actual_before_post_quant"]))
                    joins["actual_after_post_quant"] = dict(record=record["actual_after_post_quant"], scope="actual consumed decoder entry; no independent learned postquant projection in saved reader")
            repeat = None
            if (branch / "decoder-native-repeat.json").exists():
                r = read(branch / "decoder-native-repeat.json")
                require(r["origin"] == "same_native_final_latent_repeat" and r["source_native_final_cut"] == path["native_final"], "same-native-latent repeat source differs")
                repeat_images = image_record(branch, r)
                repeat = dict(errors={name: metric(a, b) for name, a, b in zip(("decoded", "pixels"), repeat_images, n_images)},
                    argument_joins={name: metric(array(branch, r[name]), array(branch, n[name])) for name in ("decoder_input", "actual_before_post_quant", "actual_after_post_quant")},
                    origin=r["origin"], native_final=path["native_final"])
            cases[key] = dict(states=state_accounts(root, store, branch), acts={lane: act_accounts(root, store, branch, arm, lane) for lane in ("forecast", "native")},
                images=images, effects_from_own_baseline=effects, decoder_joins=joins, native_same_latent_repeat=repeat,
                sealed_state_response_reductions=response_reductions,
                gauge_writer_ports=gauge_ports(root, identifier, arm, inventory),
                gauge_ideal_zero=(arm in ("gauge", "sham")), scope="conditional own-state suffix; spectral intervention is public endpoint Act plus separate native FF assay")
    emissions = [json.loads(line) for line in (root / "receipt-emissions.jsonl").read_text().splitlines()]
    require([r["number"] for r in emissions] == list(range(len(emissions))), "receipt emission order differs")
    for row in emissions:
        receipt_valid(row["receipt"])
    census = dict(Counter(r["kind"] for r in emissions))
    require(census == {"source-project": 4, "act": 44, "native-joint0": 8, "native": 2200, "restore": 4}, "actual receipt census differs")
    lineage = lineage_accounts(root, store, roots, state_index, emissions)
    for key in ("functional_transitions", "native_target_transitions", "functional_ff_assays", "native_ff_assays", "functional_vae_calls", "native_vae_repeat_calls", "native_vae_reference_calls"):
        require(summary["counts"][key] == summary["contract"][key], "declared/observed physical census differs")
    report = dict(schema="mdb-astra-flux-saved-audit-v1", native_job_id=summary["job_id"], native_request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, publication_id=publication_id, source_worker_sha256=WORKER_SHA256,
        chronology=chronology, reduced_law=ff_accounts(root), cases=cases, receipt_census=census, receipt_identity_statistics=receipt_statistics(emissions), native_lineage=lineage,
        complete_census=summary["counts"], observed_primitive_work=summary["observed"], independent_BN_retrieval=bn_account,
        source=dict(parameter_recipe_counts=source["parameter_recipe_counts"], final_resident_parameter_checks=source["final_resident_parameter_checks"],
            sources=source["sources"], helper_source_sha256=source["helper_source_sha256"], original_members=len(inventory), original_bytes=sum(r["bytes"] for r in inventory.values())),
        mechanics_status="saved evidence authenticated", trend_status="observations; numerical misses preserved", terminal_status="unassessed",
        limitations=["Intermediate unretained tensors have complete hashes, layout and witnesses, not reconstructed full norms on a miss.",
            "Source-factor Ritz extrema are strongest found, not certified global extrema; scalar ideal is real arithmetic, not BF16 regrouping.",
            "Spectral endpoint Act does not establish that an edited native full block internally produced the endpoint.",
            "Four same-native-latent repeats measure native decoder repeatability; forecast-versus-native images can combine latent and decoder differences, without a general error decomposition.",
            "Shared functional primitive kernels are not independent kernels; the reader performs no learned operation."])
    full = [value for case in cases.values() for row in case["states"] if row["full_errors"] is not None for value in row["full_errors"].values()]
    hashes = [value for case in cases.values() for row in case["states"] for value in row["hashes"].values()]
    report["coverage"] = dict(image_pairs=len(cases), same_native_latent_repeats=sum(case["native_same_latent_repeat"] is not None for case in cases.values()),
        whole_boundary_tensor_hash_checks=len(hashes), whole_boundary_hash_positions=sum(v["elements"] for v in hashes),
        hash_mismatches=sum(not v["value_hash_exact"] for v in hashes), layout_mismatches=sum(not v["layout_exact"] for v in hashes),
        complete_retained_field_comparisons=len(full), complete_retained_positions=sum(v["elements"] for v in full),
        complete_retained_nonzero_comparisons=sum(bool(v["changed"]) for v in full),
        scope="Whole-boundary and retained-full counts overlap; FF, scalar-law, image, response and control accounts reported separately, not added as independent fields.")
    require(len(canonical(report)) < 10 << 20, "compact audit bound exceeded")
    output.mkdir(parents=True, exist_ok=False); save(output / "audit.json", report)
    save(output / "evidence.json", dict(native_root=str(root), manifest_sha256=expected_manifest, request_sha256=expected_request,
        source=source, cases={k: v["root"] for k, v in state_index.items()}, raw_arrays_copied=0, weights_copied=0))
    return report


def worker(config_path):
    config = validate_config(read(config_path)); job = os.environ["MRUN_JOB_ID"]
    require(job.startswith("job-") and "/" not in job, "invalid audit job identity")
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "audit request/config differs")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "audit executed source differs")
    output = Path(config["output_scratch"]) / job / RESULT
    report = audit(config["native_root"], output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"],
        expected_job=config.get("native_job_id"), publication_id=config.get("publication_id"))
    save(output / "summary.json", dict(job_id=job, request_sha256=identity, native_job_id=report["native_job_id"], mechanics_status=report["mechanics_status"], trend_status=report["trend_status"]))
    files = {str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "files.json", dict(schema="mdb-image-tail-files-v1", job_id=job, request_sha256=identity, files=files, expanded_bytes=sum(r["bytes"] for r in files.values())))
    ready = dict(output_root=str(output), files_manifest_sha256=sha(output / "files.json"), file_count=len(files), expanded_bytes=sum(r["bytes"] for r in files.values()),
        summary_sha256=sha(output / "summary.json"), job_id=job, request_sha256=identity)
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
