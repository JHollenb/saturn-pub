"""Audit saved binding-generalization evidence without model hydration.

Usage: python flux_binding_generalization_audit.py ROOT OUTPUT.json
ROOT is the completed ``binding-minimal-result`` directory (or its collected
parent). NumPy performs the independent rounded writer calculation; Pillow
decodes existing PNGs. The optional installed MDB verifier checks transactions.
No Torch, Diffusers, model weights, model forward or replay is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path


ARMS = ("base-replay", "I", "C", "IC.early", "IC.late", "IC.control-rows",
        "IC.restoreI", "IC.restoreT", "IC.restoreIT")
NATURAL = ("natural.base", "natural.source")
EXPECTED = {"images": 88, "native_transitions": 6008, "blocks": 5368,
            "acts": 184, "encodes": 16, "noise_draws": 2}
COUNT_ALIASES = {
    "images": ("images",),
    "native_transitions": ("native_transitions", "transitions", "nav_steps"),
    "blocks": ("blocks", "native_blocks", "block_entries"),
    "acts": ("acts", "typed_acts"),
    "encodes": ("encodes", "prompt_encodes"),
    "noise_draws": ("noise_draws", "draws"),
}
HEX = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    """Broken evidence, identity, geometry or execution contract."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    def bad(value):
        raise AuditError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=bad)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def untag(value):
    """Decode scalar-only forecast trees or Saturn tagged metadata."""
    require(isinstance(value, dict), "malformed tagged tree")
    kind = value.get("kind")
    if kind == "scalar":
        return value["value"]
    if kind == "mapping":
        items = value["items"]
        if isinstance(items, dict):
            return {key: untag(child) for key, child in items.items()}
        keys = [untag(item["key"]) for item in items]
        require(len(set(keys)) == len(keys), "duplicate tagged mapping key")
        return {key: untag(item["value"]) for key, item in zip(keys, items)}
    if kind in ("tuple", "list"):
        return [untag(item) for item in value["items"]]
    if kind == "tensor":
        return value
    raise AuditError("unsupported tagged tree kind: " + str(kind))


def safe_path(root, relative):
    require(isinstance(relative, str) and relative, "artifact path must be text")
    relative_path = Path(relative)
    require(not relative_path.is_absolute() and ".." not in relative_path.parts,
            "artifact path escapes evidence root: " + relative)
    path = root / relative_path
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent)
            and path.resolve().is_relative_to(root), "symlink/escaped artifact: " + relative)
    require(path.is_file(), "missing artifact: " + relative)
    return path


def find_root(root):
    root = Path(root).resolve(strict=True)
    candidates = [p for p in (root, root / "binding-minimal-result",
                              root / "extracted/binding-minimal-result")
                  if (p / "plan.json").is_file()]
    require(len(candidates) == 1, "ROOT must identify one completed evidence directory")
    return candidates[0]


def inventory(root):
    """Authenticate every declared file and reject silently uninventoried data."""
    path = root / "files.json"
    declared = read(path)
    require(isinstance(declared, dict) and declared, "empty/malformed files.json")
    hashes = {}
    total = 0
    for relative, reference in declared.items():
        require(relative != "files.json", "inventory must not hash itself")
        expected = reference.get("sha256") if isinstance(reference, dict) else reference
        require(isinstance(expected, str) and HEX.fullmatch(expected),
                "invalid inventory hash: " + relative)
        member = safe_path(root, relative)
        if isinstance(reference, dict) and "bytes" in reference:
            require(member.stat().st_size == reference["bytes"], "inventory extent differs: " + relative)
        actual = sha(member)
        require(actual == expected, "inventory hash differs: " + relative)
        hashes[relative] = actual
        total += member.stat().st_size
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(files == set(hashes) | {"files.json"},
            "inventory omits/adds files: " + repr(sorted(files ^ (set(hashes) | {"files.json"}))))
    return hashes, {"files": len(hashes), "bytes": total,
                    "files_manifest_sha256": sha(path), "all_files_verified": True}


def decode_bf16(bits):
    import numpy as np
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_bf16(value):
    """Independent finite float32 -> BF16 round-to-nearest, ties-to-even bits."""
    import numpy as np
    value = np.asarray(value, dtype=np.float32)
    require(bool(np.isfinite(value).all()), "nonfinite intermediate in writer equation")
    bits = value.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
    require(bool(np.isfinite(decode_bf16(rounded)).all()), "BF16 writer cast overflow")
    return rounded


def sequential_writer(residual, attention, mlp, gate_attention, gate_mlp):
    """B(B(R + B(gA*A)) + B(gM*M)), retaining all four BF16 rounds."""
    import numpy as np
    r, a, m, ga, gm = [decode_bf16(x) for x in
                       (residual, attention, mlp, gate_attention, gate_mlp)]
    require(r.shape == a.shape == m.shape, "writer operand shapes differ")
    try:
        attention_write = decode_bf16(round_bf16(np.multiply(ga, a, dtype=np.float32)))
        after_attention = decode_bf16(round_bf16(np.add(r, attention_write, dtype=np.float32)))
        mlp_write = decode_bf16(round_bf16(np.multiply(gm, m, dtype=np.float32)))
        result = round_bf16(np.add(after_attention, mlp_write, dtype=np.float32))
    except AuditError:
        raise
    except ValueError as exc:
        raise AuditError("writer gates are not broadcast-compatible") from exc
    require(result.shape == r.shape, "gate broadcasting expanded writer geometry")
    return result


def comparison(predicted_bits, actual_bits):
    import numpy as np
    require(predicted_bits.shape == actual_bits.shape, "comparison shapes differ")
    p, a = [decode_bf16(x).astype(np.float64) for x in (predicted_bits, actual_bits)]
    diff = p - a
    rms = float(np.sqrt(np.mean(diff * diff)))
    reference_rms = float(np.sqrt(np.mean(a * a)))
    return {"raw_bits_exact": bool(np.array_equal(predicted_bits, actual_bits)),
            "different_bits": int(np.count_nonzero(predicted_bits != actual_bits)),
            "positions": int(a.size), "rms_error": rms,
            "max_abs_error": float(np.max(np.abs(diff))),
            "reference_rms": reference_rms,
            "relative_rms_error": None if reference_rms == 0 else rms / reference_rms}


class Evidence:
    def __init__(self, root, hashes):
        self.root, self.hashes, self.cuts = root, hashes, {}

    def path(self, relative):
        require(relative in self.hashes, "reference absent from inventory: " + relative)
        return safe_path(self.root, relative)

    def bits(self, reference):
        import numpy as np
        require(isinstance(reference, dict), "array reference must be a mapping")
        relative = reference.get("path")
        path = self.path(relative)
        require(reference.get("sha256") == self.hashes[relative], "array reference hash differs: " + relative)
        require(reference.get("dtype") == "bfloat16" and reference.get("storage_dtype") == "uint16",
                "array must declare BF16 values stored as uint16")
        shape = reference.get("shape")
        require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape)
                and 0 < math.prod(shape) <= 20_000_000, "invalid bounded array geometry")
        value = np.load(path, allow_pickle=False)
        require(value.dtype.kind == "u" and value.dtype.itemsize == 2 and list(value.shape) == shape,
                "NPY storage/shape differs: " + relative)
        require(bool(np.isfinite(decode_bf16(value)).all()), "nonfinite BF16 evidence: " + relative)
        return value

    def cut(self, fingerprint):
        if isinstance(fingerprint, dict):
            fingerprint = fingerprint.get("fingerprint")
        require(isinstance(fingerprint, str) and HEX.fullmatch(fingerprint), "invalid cut fingerprint")
        if fingerprint not in self.cuts:
            cut = read(self.path("store/cuts/" + fingerprint + ".json"))
            manifest = cut.get("manifest", {})
            require(cut.get("schema") == "saturn-pub-local-cut-v2"
                    and manifest.get("schema") == "saturn-pub-statecut-v2"
                    and cut.get("fingerprint") == fingerprint
                    and hashlib.sha256(canonical(manifest)).hexdigest() == fingerprint,
                    "cut manifest identity differs: " + fingerprint)
            require(cut.get("logical_parent") == manifest.get("parent"), "cut parent differs")
            require(untag(manifest["execution_point"]) == cut.get("execution_point"),
                    "cut execution point differs from sealed manifest")
            payload = manifest.get("payload", {})
            require(payload.get("kind") == "mapping", "cut payload is not a mapping")
            pairs = payload["items"]
            descriptors = {p["key"]["value"]: p["value"] for p in pairs}
            require(len(descriptors) == len(pairs) and set(cut["slots"]) == set(descriptors),
                    "cut slot descriptors are incomplete/duplicated")
            for name, page_id in cut["slots"].items():
                relative = "store/pages/" + page_id + ".json"
                page = read(self.path(relative))
                require(self.hashes[relative] == page_id
                        and page.get("schema") == "saturn-pub-slot-page-v1"
                        and page.get("value") == descriptors[name], "cut/page descriptor join differs")
                blob = page.get("tensor_blob")
                if blob is not None:
                    blob_path = "store/blobs/" + blob + ".safetensors"
                    self.path(blob_path)
                    require(self.hashes[blob_path] == blob, "tensor blob CAS identity differs")
            self.cuts[fingerprint] = (cut, descriptors)
        return self.cuts[fingerprint]

    def witness_join(self, reference, fingerprint, slot):
        """Join an NPY field to a model-bound native cut descriptor."""
        value = self.bits(reference)
        _, descriptors = self.cut(fingerprint)
        descriptor = descriptors[slot]
        raw = value.astype("<u2", copy=False).tobytes(order="C")
        require(descriptor.get("kind") == "tensor" and descriptor.get("dtype") == "torch.bfloat16"
                and descriptor.get("shape") == list(value.shape)
                and descriptor.get("sha256") == hashlib.sha256(raw).hexdigest(),
                "native cut/NPY witness differs: " + slot)
        return True

    def latent_join(self, row):
        """Join the retained final NPY witness to actual native cut tensor bytes."""
        import numpy as np
        cut, described = self.cut(row["final_cut"])
        descriptor = described["latent"]
        require(descriptor.get("kind") == "tensor" and descriptor.get("dtype") == "torch.bfloat16",
                "final cut latent is not BF16")
        page = read(self.path("store/pages/" + cut["slots"]["latent"] + ".json"))
        encoded = page["encoded"]
        require(encoded.get("kind") == "tensor", "latent page is not a tensor")
        blob = self.path("store/blobs/" + page["tensor_blob"] + ".safetensors")
        with blob.open("rb") as stream:
            raw_size = stream.read(8)
            require(len(raw_size) == 8, "truncated safetensors header")
            size = struct.unpack("<Q", raw_size)[0]
            require(0 < size <= 8 << 20, "invalid safetensors header size")
            header = json.loads(stream.read(size))
            tensor = header[encoded["name"]]
            start, stop = tensor["data_offsets"]
            require(tensor["dtype"] == "BF16" and tensor["shape"] == descriptor["shape"]
                    and type(start) is int and type(stop) is int and 0 <= start <= stop
                    and stop <= blob.stat().st_size - 8 - size
                    and stop - start == 2 * math.prod(tensor["shape"]), "latent tensor geometry differs")
            stream.seek(8 + size + start)
            raw = stream.read(stop - start)
        require(hashlib.sha256(raw).hexdigest() == descriptor["sha256"], "latent tensor raw hash differs")
        witness = self.bits(row["final_latent"])
        require(list(witness.shape) == tensor["shape"]
                and np.array_equal(witness, np.frombuffer(raw, dtype="<u2").reshape(tensor["shape"])),
                "final latent witness differs from actual cut")
        point = cut["execution_point"]
        require(point.get("logical_step") == 4 and point.get("phase") == "project",
                "final cut has wrong denoising clock")
        return {"cut": cut["fingerprint"], "latent_raw_sha256": descriptor["sha256"],
                "witness_exact": True, "execution_point": point}


def rgb_diagnostics(image, base, source):
    """Signed source-axis projections/RMS, never semantic classifications."""
    import numpy as np
    require(image.shape == base.shape == source.shape and image.ndim == 3 and image.shape[2] == 3,
            "RGB comparison geometry differs")
    result = {}
    width = image.shape[1]
    require(width >= 2, "image too narrow for half diagnostics")
    for name, selection in (("whole", slice(None)), ("left", slice(0, width // 2)),
                             ("right", slice(width // 2, None))):
        delta = image[:, selection].astype(np.float64) - base[:, selection].astype(np.float64)
        direction = source[:, selection].astype(np.float64) - base[:, selection].astype(np.float64)
        power = float(np.sum(direction * direction))
        result[name] = {
            "signed_projection_on_natural_source_minus_base": None if power == 0 else float(np.sum(delta * direction)) / power,
            "response_rms_0_255": float(np.sqrt(np.mean(delta * delta))),
            "natural_source_contrast_rms_0_255": float(np.sqrt(np.mean(direction * direction))),
            "mae_to_natural_source_0_255": float(np.mean(np.abs(image[:, selection].astype(np.float64) - source[:, selection].astype(np.float64)))),
            "positions": int(delta.size),
        }
    return result


def check_counts(summary):
    counts = summary["counts"]
    normalized = {}
    for canonical_name, expected in EXPECTED.items():
        present = [counts[name] for name in COUNT_ALIASES[canonical_name] if name in counts]
        require(present and all(type(v) is int and v == expected for v in present),
                "physical count differs/missing: " + canonical_name)
        normalized[canonical_name] = expected
    return normalized


def transaction_reports(root, summary, plan, fields):
    directories = sorted({p.parent for p in root.rglob("seal.json")
                          if (p.parent / "events").is_dir()})
    require(directories == [root / "transaction"], "expected one global forecast transaction")
    expected_prefix = {"loads": 1, "encodes": 16, "noise_draws": 2,
                       "native_transitions": 1976, "native_blocks": 1768,
                       "acts": 8, "receipts": 1984, "images": 16}
    require(summary.get("counts_at_seal") == expected_prefix, "preseal physical work differs")
    frozen_record = read(root / "transaction/forecasts/all_future_predictions.json")
    frozen = untag(frozen_record["predictions"])
    require(frozen["counts_at_seal"] == expected_prefix and frozen["semantic"] == plan["hypotheses"],
            "sealed prediction context differs from original plan/prefix")
    projections = frozen["projections"]
    require(set(projections) == {item["case"] + "." + item["arm"] for item in fields},
            "sealed projection case inventory differs")
    for item in fields:
        require(projections[item["case"] + "." + item["arm"]][item["field"]] == item["prediction"],
                "projection reference is not the presealed forecast")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seals = [i for i, row in enumerate(timeline) if row["kind"] == "sealed"]
    require(len(seals) == 1 and timeline[seals[0]]["counts"] == expected_prefix, "global seal chronology differs")
    before = [r for r in timeline[:seals[0]] if r["kind"] == "image_complete"]
    after = [r for r in timeline[seals[0]+1:] if r["kind"] == "image_complete"]
    require(len(before) == 16 and all(r["arm"] in NATURAL for r in before)
            and len(after) == 72 and all(r["arm"] in ARMS for r in after),
            "permitted natural-prefix versus added-future chronology differs")
    try:
        from mdb.forecast_transaction import verify_transaction
    except ModuleNotFoundError as exc:
        if exc.name not in {"mdb", "mdb.forecast_transaction"}:
            raise
        return {"status": "installed-verifier-unavailable", "directories": [str(p.relative_to(root)) for p in directories],
                "projection_references_match_sealed_forecast": True, "prefix_census_and_chronology_exact": True,
                "limitation": "Inventory authenticates transaction bytes; journal/seal semantics were not independently verified."}
    reports = {}
    for directory in directories:
        # The campaign inventory pins these seal bytes; use that pin explicitly.
        report = verify_transaction(directory, expected_seal_sha256=sha(directory / "seal.json"))
        require(report.get("verified") and report.get("phase") == "closed", "forecast transaction is not closed")
        reports[directory.relative_to(root).as_posix()] = report
    return {"status": "verified", "reports": reports,
            "projection_references_match_sealed_forecast": True, "prefix_census_and_chronology_exact": True,
            "pin_scope": "Seal bytes pinned by the independently verified campaign inventory; external request custody remains separate."}


def analyze(root, output):
    import numpy as np
    from PIL import Image
    root, output = find_root(root), Path(output).resolve()
    require(not output.exists(), "OUTPUT already exists; use a new additive audit path")
    require(not output.is_relative_to(root), "write audit outside the immutable evidence root")
    hashes, custody = inventory(root)
    evidence = Evidence(root, hashes)
    plan, request, summary, rows, fields, writes = [read(evidence.path(n)) for n in
                                                  ("plan.json", "request.json", "summary.json", "rows.json", "fields.json", "writes.json")]
    require(request.get("config") == plan, "request/config binding differs")
    if "request_sha256" in summary:
        require(summary["request_sha256"] == request.get("request_sha256"), "summary/request identity differs")
    counts = check_counts(summary)
    contexts = plan["contexts"]
    require(isinstance(contexts, list) and len(contexts) == 8, "expected eight declared contexts")
    cases = [c["id"] if isinstance(c, dict) else c for c in contexts]
    require(len(set(cases)) == 8 and all(isinstance(c, str) and c for c in cases), "duplicate/invalid context IDs")
    arms = plan.get("arms", plan.get("arm_specs"))
    require(isinstance(arms, list) and [a["id"] if isinstance(a, dict) else a for a in arms] == list(ARMS),
            "declared suffix arms/order differ")
    expected_rows = {(c, arm) for c in cases for arm in (*NATURAL, *ARMS)}
    indexed = {(r["case"], r["arm"]): r for r in rows}
    require(len(rows) == 88 and len(indexed) == len(rows) and set(indexed) == expected_rows,
            "row inventory is incomplete or duplicated")
    projection_index = {(r["case"], r["arm"], r["field"]): r for r in fields}
    require(len(fields) == 144 and len(projection_index) == len(fields)
            and set(projection_index) == {(c, a, f) for c in cases for a in ARMS for f in ("image", "text")},
            "projection inventory is incomplete or duplicated")
    writer_index = {(r["case"], r["arm"]): r for r in writes}
    require(len(writes) == 72 and len(writer_index) == len(writes)
            and set(writer_index) == {(c, a) for c in cases for a in ARMS},
            "writer inventory is incomplete or duplicated")
    projections = []
    for key, item in projection_index.items():
        predicted, actual = evidence.bits(item["prediction"]), evidence.bits(item["actual"])
        expected_shape = [1, 1024 if item["field"] == "image" else 512, 3072]
        require(list(actual.shape) == expected_shape, "projection is not a complete declared field")
        evidence.witness_join(item["actual"], indexed[key[0], key[1]]["marks"]["joint0_entry"], item["field"])
        projections.append({"case": key[0], "arm": key[1], "field": key[2],
                            **comparison(predicted, actual),
                            "archive_bytes_exact": item["prediction"]["sha256"] == item["actual"]["sha256"]})
    writer_reports = []
    required_ports = {"residual", "attention", "mlp", "gate_attention", "gate_mlp", "actual"}
    for key, item in writer_index.items():
        require(required_ports <= set(item["fields"]), "writer operand inventory incomplete")
        ports = {name: evidence.bits(item["fields"][name]) for name in required_ports}
        require(list(ports["actual"].shape) == [1, 1024, 3072], "writer exit is not a complete image field")
        marks = indexed[key]["marks"]
        evidence.witness_join(item["fields"]["residual"], marks["joint0_entry"], "image")
        evidence.witness_join(item["fields"]["actual"], marks["joint0_exit"], "image")
        predicted = sequential_writer(*(ports[name] for name in
                                        ("residual", "attention", "mlp", "gate_attention", "gate_mlp")))
        writer_reports.append({"case": key[0], "arm": key[1], **comparison(predicted, ports["actual"])})
    endpoint_joins, controls, diagnostics, restorations, qkv = [], {}, {}, {}, {}
    for case in cases:
        images = {}
        for arm in (*NATURAL, *ARMS):
            row = indexed[case, arm]
            image_path = evidence.path(row["png"])
            require(row["png_sha256"] == hashes[row["png"]], "PNG reference hash differs")
            with Image.open(image_path) as image:
                require(image.mode == "RGB" and image.size == (512, 512), "native PNG geometry/mode differs")
                images[arm] = np.asarray(image).copy()
            endpoint_joins.append({"case": case, "arm": arm, **evidence.latent_join(row)})
        controls[case] = {}
        for label, a, b in (("baseline_replay", "base-replay", "natural.base"),
                            ("joint_restore_closure", "IC.restoreIT", "I")):
            left, right = indexed[case, a], indexed[case, b]
            lc, _ = evidence.cut(left["final_cut"])
            rc, _ = evidence.cut(right["final_cut"])
            controls[case][label] = {
                "arms": [a, b],
                "complete_payload_descriptors_exact": lc["manifest"]["payload"] == rc["manifest"]["payload"],
                "execution_points_exact": lc["execution_point"] == rc["execution_point"],
                "png_bytes_exact": left["png_sha256"] == right["png_sha256"],
                "png_pixels_exact": bool(np.array_equal(images[a], images[b])),
                "final_latent_bits_exact": bool(np.array_equal(evidence.bits(left["final_latent"]), evidence.bits(right["final_latent"]))),
                "lineage_fingerprints_equal": lc["fingerprint"] == rc["fingerprint"],
                "scope": "Payload equality is required; parent/lineage equality is not required.",
            }
        restorations[case] = {}
        source_fingerprint = indexed[case, "I"]["marks"]["joint0_exit"]
        _, source_payload = evidence.cut(source_fingerprint)
        for arm, slots in (("IC.restoreI", ("image",)), ("IC.restoreT", ("text",)),
                           ("IC.restoreIT", ("image", "text"))):
            marks = indexed[case, arm]["marks"]
            require(marks["restoration_source"] == source_fingerprint, "restoration uses wrong matched I-only source")
            _, before_payload = evidence.cut(marks["joint0_exit"])
            _, after_payload = evidence.cut(marks["post_restore"])
            expected_payload = {**before_payload, **{slot: source_payload[slot] for slot in slots}}
            require(after_payload == expected_payload, "restoration changes undeclared payload slot")
            _, reset = evidence.cut(marks["project3_reset"])
            _, source_reset = evidence.cut(indexed[case, "I"]["marks"]["project3_reset"])
            restorations[case][arm] = {
                "selected_slots": list(slots), "typed_footprint_exact": True,
                "project3_reset_full_payload_exact_to_I": reset == source_reset,
                "post_restore_different_from_I_slots": sorted(k for k in after_payload if after_payload[k] != source_payload[k]),
            }
        qkv[case] = []
        for anchor, targets in (("base-replay", ("C",)),
                                ("I", tuple(a for a in ARMS if a not in ("base-replay", "C", "I")))):
            _, anchor_payload = evidence.cut(indexed[case, anchor]["marks"]["joint0_entry"])
            for arm in targets:
                _, payload = evidence.cut(indexed[case, arm]["marks"]["joint0_entry"])
                same_receiver = all(anchor_payload[k] == payload[k] for k in ("image", "context", "img_ids"))
                left = writer_index[case, anchor].get("qkv_descriptors", {})
                right = writer_index[case, arm].get("qkv_descriptors", {})
                require(set(left) == set(right) == {"q", "k", "v"}, "actual QKV descriptor inventory incomplete")
                qkv[case].append({"arms": [anchor, arm], "fixed_image_context_ids": same_receiver,
                                  "actual_projection_descriptor_equal": {k: left[k] == right[k] for k in ("q", "k", "v")},
                                  "role": "Actual native tensor descriptors; not independent numeric QKV forecasts or semantic gate."})
        diagnostics[case] = {arm: rgb_diagnostics(image, images["natural.base"], images["natural.source"])
                             for arm, image in images.items()}
    transactions = transaction_reports(root, summary, plan, fields)
    control_keys = ("complete_payload_descriptors_exact", "execution_points_exact", "png_bytes_exact",
                    "png_pixels_exact", "final_latent_bits_exact")
    numerical_exact = (all(r["raw_bits_exact"] for r in projections)
                       and all(r["raw_bits_exact"] for r in writer_reports))
    controls_exact = (all(v[k] for case in controls.values() for v in case.values() for k in control_keys)
                      and all(v["IC.restoreIT"]["project3_reset_full_payload_exact_to_I"] for v in restorations.values()))
    result = {
        "schema": "mdb-flux-binding-generalization-audit-v1", "mechanics_status": "verified",
        "instrument_status": "replay-and-closure-qualified" if controls_exact else "replay-or-closure-not-established",
        "numerical_status": "all-compared-bits-exact" if numerical_exact else "nonzero-discrepancies-observed",
        "replay_and_closure_exact": controls_exact,
        "trend_status": "observations-retained", "terminal_status": "unassessed",
        "source_root": str(root), "source_job": summary.get("job_id"), "custody": custody,
        "counts": counts, "contexts": cases, "projection_comparisons": len(projections),
        "exact_projection_comparisons": sum(r["raw_bits_exact"] for r in projections),
        "writer_comparisons": len(writer_reports), "exact_writer_comparisons": sum(r["raw_bits_exact"] for r in writer_reports),
        "projections": projections, "sequential_writers": writer_reports, "endpoint_joins": endpoint_joins,
        "controls": controls, "restoration_footprints": restorations,
        "qkv_invariance_diagnostics": qkv, "rgb_diagnostics": diagnostics, "transactions": transactions,
        "worker_closure_declarations": [{"case": r["case"], "arm": r["arm"], "closures": r.get("closures", {})} for r in rows],
        "new_model_calls": 0, "new_model_hydrations": 0,
        "limitations": [
            "Projected I/T are sealed numerical predictions; writer equations are conditional reconstructions from captured native operands, and RGB is observed native output.",
            "Signed projections and left/right image halves are diagnostics, not attribute-binding classifiers or causal percentages; zero source contrast yields null projection.",
            "Context counts and repeated tensors are dependent; see the frozen plan for seeds, wording, attribute class and token alignment.",
            "Inventory and local transaction consistency do not replace external scheduler/request/source custody.",
            "Worker closure booleans are retained as declarations; selected restoration footprints and project3/final payload joins are independently checked from actual cut descriptors.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(args.root, args.output)
    except (AuditError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, "saved-evidence audit refused: " + str(exc) + "\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "projection_comparisons",
                                            "exact_projection_comparisons", "writer_comparisons",
                                            "exact_writer_comparisons", "new_model_calls")}, sort_keys=True))
    return 0 if result["mechanics_status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
