"""Saved-array Klein write audit and ungated image-color diagnostics.

No model, equation module, native replay or Torch execution. RGB is a native
consumer observation, never an image forecast. Color regions are uncalibrated
proxies; they do not certify cube/sphere identity, count or semantic alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import struct
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from mdb.forward_ports import NativePortCatalog
from mdb.forward_symbols import verify_trace

ARMS = ("native", "sham", "forecastself", "donorfull", "donorleft", "donorright")
COLOR_POLICY = {"red_hue_degrees": [[0, 25], [335, 360]], "blue_hue_degrees": [[190, 260]],
                "minimum_saturation": .30, "minimum_value": .25, "minimum_alpha": .5,
                "component_connectivity": 4, "reported_component_minimum_pixels": 16,
                "calibration": "none; descriptive diagnostic only; no execution/scientific gate"}


@lru_cache(maxsize=1)
def helpers():
    return runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))


def quantize_bf16(value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite BF16 cast input")
    bits = value.view(np.uint32)
    result = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BF16 cast overflow")
    return result


def descriptor(value, *, bf16=False):
    value = np.asarray(value)
    if bf16:
        if value.dtype != np.float32 or not np.array_equal(quantize_bf16(value), value):
            raise ValueError("native BF16 witness must contain exactly widened BF16 values")
        raw = (value.view(np.uint32) >> 16).astype("<u2").tobytes()
        dtype = "torch.bfloat16"
    else:
        raw, dtype = value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes(), "torch." + str(value.dtype)
    return {"shape": list(value.shape), "dtype": dtype, "sha256": hashlib.sha256(raw).hexdigest()}


def matches(value, expected, *, bf16=False):
    return all(expected.get(key) == item for key, item in descriptor(value, bf16=bf16).items())


def unpack(value):
    if value["kind"] == "mapping":
        return {item["key"]["value"]: unpack(item["value"]) for item in value["items"]}
    if value["kind"] in {"tuple", "list"}:
        return [unpack(item) for item in value["items"]]
    return value["value"] if value["kind"] == "scalar" else value


def cut_id(manifest):
    return helpers()["sha"](helpers()["canonical"](manifest))


def audit_store(root):
    """Verify CAS metadata, raw safetensor hashes and typed tensor descriptors."""
    h, cuts, pages, blobs = helpers(), {}, {}, {}
    for path in (root / "blobs").glob("*.safetensors"):
        raw = path.read_bytes()
        h["require"](h["sha"](raw) == path.stem, "typed tensor blob CAS differs")
        h["require"](len(raw) >= 8, "truncated safetensors header")
        size = struct.unpack("<Q", raw[:8])[0]
        h["require"](size <= 8 << 20 and 8 + size <= len(raw), "invalid bounded safetensors header")
        header = json.loads(raw[8:8 + size])
        blobs[path.stem] = (path, header, 8 + size)
    types = {"BF16": "torch.bfloat16", "F32": "torch.float32", "F64": "torch.float64", "F16": "torch.float16",
             "I64": "torch.int64", "I32": "torch.int32", "BOOL": "torch.bool", "U8": "torch.uint8"}

    def verify_encoded(encoded, described, blob, used):
        kind = encoded["kind"]
        h["require"](kind == ("tensor" if described["kind"] == "tensor" else described["kind"]), "typed page kind differs")
        if kind == "tensor":
            path, header, offset = blobs[blob]
            name = encoded["name"]
            h["require"](name not in used, "repeated tensor reference")
            used.add(name)
            item = header[name]
            start, stop = item["data_offsets"]
            widths = {"BF16": 2, "F32": 4, "F64": 8, "F16": 2, "I64": 8, "I32": 4, "BOOL": 1, "U8": 1}
            h["require"](type(start) is int and type(stop) is int and 0 <= start <= stop <= path.stat().st_size - offset
                         and stop - start == int(np.prod(item["shape"], dtype=np.int64)) * widths[item["dtype"]], "typed tensor storage extent differs")
            with path.open("rb") as stream:
                stream.seek(offset + start)
                raw = stream.read(stop - start)
            h["require"](encoded["shape"] == item["shape"] == described["shape"]
                         and encoded["stride"] == described["stride"]
                         and types[item["dtype"]] == described["dtype"] and h["sha"](raw) == described["sha256"],
                         "typed page tensor bytes/geometry differ")
        elif kind == "mapping":
            mapping = {item["key"]["value"]: item["value"] for item in described["items"]}
            h["require"](set(encoded["items"]) == set(mapping), "typed page mapping inventory differs")
            for name, child in encoded["items"].items():
                verify_encoded(child, mapping[name], blob, used)
        elif kind in {"tuple", "list"}:
            h["require"](len(encoded["items"]) == len(described["items"]), "typed page sequence length differs")
            for left, right in zip(encoded["items"], described["items"], strict=True):
                verify_encoded(left, right, blob, used)
        else:
            h["require"](encoded == described, "typed page scalar differs")

    for path in (root / "pages").glob("*.json"):
        h["require"](h["sha"](path.read_bytes()) == path.stem, "typed slot page CAS differs")
        page = h["read"](path)
        h["require"](page["schema"] == "saturn-pub-slot-page-v1", "typed page schema differs")
        used = set()
        verify_encoded(page["encoded"], page["value"], page["tensor_blob"], used)
        if page["tensor_blob"] is not None:
            h["require"](used == set(blobs[page["tensor_blob"]][1]) - {"__metadata__"}, "unreferenced typed tensor data")
        pages[path.stem] = page
    for path in (root / "cuts").glob("*.json"):
        item = h["read"](path)
        manifest = item["manifest"]
        h["require"](item["schema"] == "saturn-pub-local-cut-v2" and manifest["schema"] == "saturn-pub-statecut-v2"
                     and cut_id(manifest) == path.stem == item["fingerprint"] and item["logical_parent"] == manifest["parent"]
                     and item["execution_point"] == unpack(manifest["execution_point"])
                     and item["surface"] == unpack(manifest["surface"]), "typed cut seal/metadata differs")
        payload = {row["key"]["value"]: row["value"] for row in manifest["payload"]["items"]}
        h["require"](set(item["slots"]) == set(payload), "typed cut slot inventory differs")
        for name, page in item["slots"].items():
            h["require"](pages[page]["value"] == payload[name], "typed cut descriptor differs from saved page")
        h["require"](sorted({pages[page]["tensor_blob"] for page in item["slots"].values() if pages[page]["tensor_blob"]}) == item["physical_blobs"], "typed cut physical blob inventory differs")
        cuts[path.stem] = manifest
    return {"cuts": cuts, "counts": {"cuts": len(cuts), "pages": len(pages), "tensor_blobs": len(blobs)}}


def _regions(mask):
    visited, components = np.zeros(mask.shape, dtype=bool), []
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        pending, positions = deque([(int(y), int(x))]), []
        visited[y, x] = True
        while pending:
            py, px = pending.popleft()
            positions.append((py, px))
            for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    pending.append((ny, nx))
        points = np.asarray(positions)
        components.append({"pixels": len(points), "centroid_xy": [float(points[:, 1].mean()), float(points[:, 0].mean())],
                           "bbox_xyxy": [int(points[:, 1].min()), int(points[:, 0].min()), int(points[:, 1].max()), int(points[:, 0].max())]})
    components.sort(key=lambda item: (-item["pixels"], item["centroid_xy"][0], item["centroid_xy"][1]))
    reported = [item for item in components if item["pixels"] >= COLOR_POLICY["reported_component_minimum_pixels"]]
    return {"threshold_pixel_count": int(mask.sum()), "components": len(components), "reported_components": reported[:5],
            "largest": reported[0] if reported else None, "status": "color-region-observed" if reported else "unknown-under-diagnostic-threshold"}


def image_diagnostics(pixels):
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("expected native uint8 RGB/RGBA pixels")
    rgb = pixels[:, :, :3].astype(np.float64) / 255.
    maximum, minimum = rgb.max(-1), rgb.min(-1)
    difference = maximum - minimum
    saturation = np.divide(difference, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    hue = np.zeros_like(maximum)
    active = difference > 0
    for channel, base in ((0, 0), (1, 2), (2, 4)):
        selected = active & (rgb[:, :, channel] == maximum)
        numerator = rgb[:, :, (channel + 1) % 3] - rgb[:, :, (channel + 2) % 3]
        ratio = np.divide(numerator, difference, out=np.zeros_like(hue), where=active)
        hue[selected] = ((ratio[selected] + base) % 6) * 60
    eligible = (saturation >= COLOR_POLICY["minimum_saturation"]) & (maximum >= COLOR_POLICY["minimum_value"])
    if pixels.shape[2] == 4:
        eligible &= pixels[:, :, 3] / 255. >= COLOR_POLICY["minimum_alpha"]
    red = _regions(eligible & ((hue <= 25) | (hue >= 335)))
    blue = _regions(eligible & (hue >= 190) & (hue <= 260))
    separation = None
    if red["largest"] is not None and blue["largest"] is not None:
        separation = (red["largest"]["centroid_xy"][0] - blue["largest"]["centroid_xy"][0]) / pixels.shape[1]
    return {"red": red, "blue": blue, "red_minus_blue_normalized_x": separation,
            "shape_identity": "unassessed; color regions do not identify cubes/spheres",
            "semantic_gate": False, "policy": COLOR_POLICY}


def pixel_difference(actual, baseline):
    if actual.shape != baseline.shape:
        return {"status": "different-native-image-geometry"}
    delta = actual[:, :, :3].astype(np.float64) - baseline[:, :, :3]
    middle = actual.shape[1] // 2
    return {"status": "observed", "changed_pixel_fraction": float(np.any(delta != 0, axis=-1).mean()),
            "mean_absolute_channel_difference": float(np.abs(delta).mean()), "signed_mean": float(delta.mean()),
            "left_half_MAE": float(np.abs(delta[:, :middle]).mean()), "right_half_MAE": float(np.abs(delta[:, middle:]).mean()),
            "scope": "rendered pixel collateral; local latent mask preservation does not imply unchanged pixels"}


def audit(run_dir):
    from PIL import Image

    h = helpers()
    run = h["source_run"](run_dir, "flux-mlp-result", max_archive_bytes=512 << 20, max_member_bytes=256 << 20)
    data, tx, tree = run["data"], run["transaction"], run["tree"]
    source, summary = h["read"](data / "source.json"), h["read"](data / "summary.json")
    h["require"](source["request"] == run["request"] and run["plan"] == run["request"]["config"], "source/plan/request binding differs")
    h["require"](h["sha"](h["canonical"](source["checkpoint_files"])) == run["plan"]["expected_checkpoint_sha256"] == source["model_identity"]["checkpoint_sha256"], "pipeline checkpoint inventory differs")
    prefixes = tree.decode(h["read"](tx / "frontiers/prefixes.json")["value"])
    declarations = h["read"](tx / "declarations.json")
    h["require"](declarations["model_identity"] == source["model_identity"]
                 and declarations["source_identity"] == run["request"]["source_hashes"]
                 and declarations["input_identity"] == cut_id({name: value["parent_fingerprint"] for name, value in prefixes.items()})
                 and tree.decode(declarations["actual_weight_manifest"]) == {}, "transaction declared model/source/prefix/parameterless writer binding differs")
    external = h["read"](data / "external-weights.json")
    sealed_external = tree.decode(h["read"](tx / "frontiers/external_weights.json")["value"])
    h["require"](external["checkpoint_sha256"] == sealed_external["checkpoint_sha256"] == source["model_identity"]["checkpoint_sha256"]
                 and external["weights"] == sealed_external["weights"], "external FF weight binding differs")
    checks = {}
    for stage in ("pre-prediction", "at-release", "post-execution"):
        check = h["read"](data / ("weight-check-" + stage + ".json"))
        for name, binding in external["weights"].items():
            h["require"](not binding["external_matrix_serialized"] and check[name]["exact"]
                         and check[name]["native"] == binding["native"] and check[name]["external_fp32_sha256"] == binding["external_fp32_sha256"], "external native/copy weights changed at " + stage)
        checks[stage] = check
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    h["require"]([event["sequence"] for event in timeline] == list(range(len(timeline))), "worker timeline is not contiguous")
    seals = [event for event in timeline if event["kind"] == "global_forecast_sealed"]
    releases = [event for event in timeline if event["kind"] == "native_suffix_released"]
    h["require"](len(seals) == len(releases) == 1 and seals[0]["sequence"] < releases[0]["sequence"]
                 and seals[0]["seal_sha256"] == run["transaction_report"]["seal_sha256"], "native suffix release differs from global forecast seal")
    pre = seals[0]["pre_release_counters"]
    h["require"](pre == summary["pre_release_counters"] and pre["prefix_attempts"] == pre["prefix_stops"] == len(prefixes)
                 and not any(pre[name] for name in ("target_ff_entries", "vae_entries", "denied_entries")), "genuine prefix/suffix exclusion counters differ")
    h["require"](all(event["sequence"] < seals[0]["sequence"] for event in timeline if event["kind"] == "prefix_complete"), "prefix completed after seal")
    h["require"]({event["scene"] for event in timeline if event["kind"] == "prefix_complete"} == set(prefixes)
                 and all(event["sequence"] > releases[0]["sequence"] for event in timeline
                         if event["kind"] in {"branch_start", "RGB_decode_start", "branch_complete"}), "native branch/prefix timeline differs from release")
    exclusion = tree.decode(h["read"](tx / "frontiers/exclusion.json")["value"])
    h["require"](exclusion["pre_release_counters"] == pre and exclusion["target_ff_entries"] == exclusion["VAE_entries"] == 0, "sealed exclusion differs")
    typed = audit_store(data / "typed-store")
    observed = tree.decode(h["read"](tx / "observations/native.json")["value"])
    rows, native = observed["rows"], observed["native_audits"]
    forecasts, payloads, numerical, prefix_reports = {}, {}, {}, {}
    for scene, prefix in prefixes.items():
        errors = []
        doc = h["read"](tx / ("forecasts/" + scene + ".json"))
        predicted = tree.decode(doc["predictions"])
        forecasts[scene] = predicted["values"]
        payloads[scene] = tree.decode(doc["payloads"]["write"]["value"])
        expected_write = quantize_bf16(predicted["values"]["write"])
        if not np.array_equal(payloads[scene], expected_write) or payloads[scene].dtype != np.float32:
            errors.append("sealed BF16 payload differs from FP64→FP32→BF16-RNE forecast write")
        before, after = prefix["typed_prefix_before"], prefix["typed_prefix_after"]
        parent = prefix["parent_manifest"]
        if before != after or cut_id(parent) != prefix["parent_fingerprint"] or typed["cuts"].get(prefix["parent_fingerprint"]) != parent:
            errors.append("genuine partial block changed parent or typed parent custody differs")
        if prefix["prefix_contract"]["native_target_FF_computed"] or not prefix["prefix_contract"]["typed_parent_unchanged"]:
            errors.append("prefix boundary claims a computed FF or changed parent")
        actual = native[scene]
        if actual["parent_fingerprint"] != prefix["parent_fingerprint"]:
            errors.append("native block starts from another typed parent")
        for name in ("r", "x"):
            if not np.array_equal(actual[name], prefix[name]) or not matches(actual[name], actual["native_descriptors"][name], bf16=True):
                errors.append("fresh native/prefix boundary join differs: " + name)
        for name in ("y", "write", "image"):
            if not matches(actual[name], actual["native_descriptors"][name], bf16=True):
                errors.append("native BF16 witness descriptor differs: " + name)
        if not np.array_equal(quantize_bf16(prefix["gate"] * actual["y"]), actual["write"]) or not np.array_equal(quantize_bf16(actual["r"] + actual["write"]), actual["image"]):
            errors.append("actual BF16 gated write/residual arithmetic differs")
        columns = np.unique(prefix["img_ids"][0, :, 2])
        left = np.isin(prefix["img_ids"][:, :, 2], columns[:len(columns) // 2])[:, :, None]
        if not np.array_equal(left, prefix["masks"]["left"]) or not np.array_equal(~left, prefix["masks"]["right"]):
            errors.append("spatial masks differ from fixed native coordinate halves")
        numerical[scene] = {"mechanics_errors": errors, "y": h["difference"](actual["y"], forecasts[scene]["y"]),
            "write": h["difference"](actual["write"], forecasts[scene]["write"]),
            "residual": h["difference"](actual["image"], forecasts[scene]["residual"]),
            "scope": "independent FP64 vs actual CUDA BF16; residual magnitudes do not assign scientific pass/fail"}
        prefix_reports[scene] = {"parent_fingerprint": prefix["parent_fingerprint"], "prompt": prefix["prompt"],
                                 "prefix_contract": prefix["prefix_contract"], "shape": list(prefix["r"].shape), "mask_description": prefix["masks"]["description"]}
    transitions = defaultdict(list)
    for line in (data / "transition-receipts.jsonl").read_text().splitlines():
        item = json.loads(line)
        receipt = item["receipt"]
        h["public_receipt"](receipt)
        h["require"](h["read"](data / "typed-store/receipts" / (receipt["fingerprint"] + ".json")) == receipt, "typed transition CAS differs")
        h["require"](receipt["operation"] == "continue" and receipt["steps"] == len(receipt["transitions"]), "typed transition geometry differs")
        transitions[item["branch"]].append(receipt)
    reports, pixels = {}, {}
    expected_rows = {scene + "." + arm for scene in prefixes for arm in ARMS}
    global_errors = [] if set(rows) == expected_rows else ["planned scene/arm observation inventory is incomplete"]
    for row_id, row in rows.items():
        errors = []
        try:
            scene, arm = row["recipient"], row["arm"]
            base_image = native[scene]["image"]
            post = native[scene]["postblock_manifest"]
            post_id = cut_id(post)
            if typed["cuts"].get(post_id) != post or row["native_postblock_fingerprint"] != post_id:
                errors.append("native postblock typed cut differs")
            if arm in {"native", "sham"}:
                candidate = base_image
                writer_report = None
            else:
                origin = scene if arm == "forecastself" else ("right" if scene == "left" else "left")
                if row["donor"] != origin:
                    errors.append("branch forecast donor differs from declared recipe")
                writer_output = np.asarray(prefixes[scene]["r"] + payloads[origin], dtype=np.float32)
                candidate = quantize_bf16(writer_output)
                trace = h["read"](data / "writer-traces" / (row_id + ".json"))
                verify_trace(trace)
                traced_inputs = sorted((item["shape"], item["dtype"], item["sha256"]) for item in trace["values"].values() if item["origin"] == "input")
                expected_inputs = sorted((item["shape"], item["dtype"], item["sha256"]) for item in map(descriptor, (prefixes[scene]["r"], payloads[origin])))
                if traced_inputs != expected_inputs or not matches(writer_output, trace["values"][trace["outputs"]["$value"]]):
                    errors.append("parameterless CPU writer inputs/output differ")
                port = NativePortCatalog(trace).resolve("module:<root>:return")
                if row["Act"]["writer_trace"] != trace["fingerprint"] or row["Act"]["writer_output_port"] != port:
                    errors.append("typed branch CPU writer port differs")
                if arm in {"donorleft", "donorright"}:
                    side = "left" if arm == "donorleft" else "right"
                    candidate = np.where(prefixes[scene]["masks"][side], candidate, base_image)
                    if row["mask"]["side"] != side or row["mask"]["rows"] != int(prefixes[scene]["masks"][side].sum()):
                        errors.append("declared fixed spatial mask differs")
                writer_report = {"trace_fingerprint": trace["fingerprint"], "CPU_FP32_to_BF16_rounding": h["difference"](candidate, writer_output)}
            if not matches(candidate, row["supplied_image_descriptor"], bf16=True):
                errors.append("applied native image is not the declared quantized/masked candidate")
            applied = row["applied_manifest"]
            supplied_slots = unpack(applied["payload"])
            native_slots = unpack(post["payload"])
            text_descriptor = native[scene]["native_descriptors"]["text"]
            if (not matches(base_image, native_slots["image"], bf16=True)
                    or any(native_slots["text"].get(key) != text_descriptor[key] for key in ("shape", "dtype", "sha256"))):
                errors.append("native postblock image/text descriptor differs from retained native witness")
            if not matches(candidate, supplied_slots["image"], bf16=True) or supplied_slots["text"] != native_slots["text"]:
                errors.append("applied cut image bytes or recipient text preservation differs")
            if not row["text_preserved_exact"] or not row["unmasked_native_exact"]:
                errors.append("worker observed text/unmasked local-state corruption")
            act = row["Act"]
            if arm == "native":
                if act is not None:
                    errors.append("native branch has an unexpected Act")
            else:
                receipt = act["receipt"]
                h["public_receipt"](receipt)
                if (receipt["fingerprint"] != act["receipt_id"] or receipt["operation"] != "apply" or receipt["act"] != act["act"]
                        or act["act"]["writes"] != ["image"] or act["act"]["operation"]["id"] != "saturn.builtin.replace"
                        or receipt["result"] != cut_id(applied) or receipt["parent"] != cut_id({**post, "parent": post_id})):
                    errors.append("public image Act/receipt identity differs")
                if not matches(candidate, unpack(receipt["writes"])["image"], bf16=True):
                    errors.append("public Act write bytes differ from predicted recipe")
                if not matches(candidate, unpack(act["act"]["parameters"])["value"], bf16=True):
                    errors.append("public Act replacement parameter differs from predicted recipe")
                if h["read"](data / "typed-store/receipts" / (receipt["fingerprint"] + ".json")) != receipt:
                    errors.append("public Act receipt CAS differs")
            suffix = transitions[row_id + ".native_suffix"]
            if len(suffix) != 1 or suffix[0]["parent"] != cut_id(applied) or suffix[0]["result"] != cut_id(row["final_manifest"]):
                errors.append("native continuation receipt does not join applied/final cuts")
            if not matches(row["final_latent"], unpack(row["final_manifest"]["payload"])["latent"], bf16=True):
                errors.append("final native latent differs from final typed cut")
            image_path = h["file_path"](data, row["RGB"]["path"])
            if h["sha"](image_path.read_bytes()) != row["RGB"]["png_sha256"]:
                errors.append("native PNG file hash differs")
            with Image.open(image_path) as image:
                array = np.asarray(image).copy()
            if list(array.shape) != row["RGB"]["shape"] or h["sha"](array.tobytes()) != row["RGB"]["pixel_sha256"]:
                errors.append("decoded native pixel bytes differ")
            pixels[row_id] = array
            reports[row_id] = {"mechanics_errors": errors, "scene": scene, "arm": arm, "donor": row["donor"],
                "RGB": row["RGB"], "image_diagnostic": image_diagnostics(array), "CPU_writer": writer_report,
                "final_latent_descriptor": descriptor(row["final_latent"], bf16=True),
                "Act_fingerprint": None if act is None else act["receipt_id"], "future_scope": row["future_scope"]}
        except (KeyError, ValueError, TypeError, IndexError, OSError) as error:
            reports[row_id] = {"mechanics_errors": [*errors, f"row instrument: {type(error).__name__}: {error}"],
                               "scene": row.get("recipient"), "arm": row.get("arm"), "semantic_status": "unknown-incomplete-evidence"}
    for row_id, array in pixels.items():
        scene = rows[row_id]["recipient"]
        baseline = pixels.get(scene + ".native")
        if baseline is not None:
            reports[row_id]["pixel_collateral_vs_native"] = pixel_difference(array, baseline)
            if row_id.endswith(".sham") and not np.array_equal(array, baseline):
                reports[row_id]["mechanics_errors"].append("native/sham rendered image differs")
            if row_id.endswith(".sham"):
                controls = native[scene]["sham_control"]
                latent_equal = np.array_equal(rows[row_id]["final_latent"], rows[scene + ".native"]["final_latent"])
                if not latent_equal or controls != {"final_latent_exact": latent_equal, "RGB_exact": bool(np.array_equal(array, baseline))}:
                    reports[row_id]["mechanics_errors"].append("native/sham final latent or retained control differs")
            actual_x = reports[row_id]["image_diagnostic"]["red_minus_blue_normalized_x"]
            native_x = reports[scene + ".native"]["image_diagnostic"]["red_minus_blue_normalized_x"]
            reports[row_id]["red_blue_relation_effect"] = None if actual_x is None or native_x is None else actual_x - native_x
        analytic_self = pixels.get(scene + ".forecastself")
        if analytic_self is not None:
            reports[row_id]["pixel_collateral_vs_forecastself"] = pixel_difference(array, analytic_self)
            actual_x = reports[row_id]["image_diagnostic"]["red_minus_blue_normalized_x"]
            self_x = reports[scene + ".forecastself"]["image_diagnostic"]["red_minus_blue_normalized_x"]
            reports[row_id]["red_blue_relation_effect_vs_forecastself"] = None if actual_x is None or self_x is None else actual_x - self_x
            reports[row_id]["final_latent_vs_forecastself"] = h["difference"](rows[row_id]["final_latent"], rows[scene + ".forecastself"]["final_latent"])
        reports[row_id]["contrast_scope"] = ("donorfull vs forecastself holds recipient residual/text and full-grid analytic installation fixed"
            if row_id.endswith(".donorfull") else "half-grid transfers lack a footprint-matched analytic self control"
            if rows[row_id]["arm"] in {"donorleft", "donorright"} else "native contrasts include approximation and donor-content effects where applicable")
    counts = summary["counts"]
    if (counts["images"] != len(pixels) or counts["acts"] != sum(row["Act"] is not None for row in rows.values())
            or counts["completed_adapter_steps"] != sum(receipt["steps"] for group in transitions.values() for receipt in group)):
        global_errors.append("native work/image/Act/transition accounting differs")
    left, right = prefixes["left"], prefixes["right"]
    cross_scene = {"initial_latent_descriptor_exact": unpack(left["parent_manifest"]["payload"])["latent"] == unpack(right["parent_manifest"]["payload"])["latent"],
        "img_ids_exact": bool(np.array_equal(left["img_ids"], right["img_ids"])),
        "gate_exact": bool(np.array_equal(left["gate"], right["gate"])),
        **{name: h["difference"](left[name], right[name]) for name in ("r", "x", "gate")},
        "forecast_write": h["difference"](forecasts["left"]["write"], forecasts["right"]["write"]),
        "native_write": h["difference"](native["left"]["write"], native["right"]["write"]),
        "scope": "observed same-time gate and shared-grid contrasts; no object-per-token or object-per-MLP claim"}
    scales = {}
    for scene in prefixes:
        donor = "right" if scene == "left" else "left"
        scales[scene] = {}
        for name in ("y", "write"):
            error = numerical[scene][name]
            contrast = h["difference"](native[donor][name], native[scene][name])
            scales[scene][name] = {"forecast_error": error, "native_donor_minus_recipient": contrast,
                "error_to_native_contrast_l2": None if contrast["l2"] == 0 else error["l2"] / contrast["l2"]}
        cast_error = h["difference"](payloads[scene], native[scene]["write"])
        contrast = h["difference"](payloads[donor], payloads[scene])
        scales[scene]["staged_write"] = {"payload_minus_native_write": cast_error, "donor_minus_recipient_payload": contrast,
            "error_to_payload_contrast_l2": None if contrast["l2"] == 0 else cast_error["l2"] / contrast["l2"]}
    valid = not global_errors and not any(row["mechanics_errors"] for row in reports.values()) and not any(row["mechanics_errors"] for row in numerical.values())
    return {"schema": "mdb-flux-mlp-forecast-audit-v1", "source": run["source"], "model_identity": source["model_identity"],
        "native_execution": source["native_execution"], "implementation_sources": source["implementation_sources"],
        "mechanics_status": "valid" if valid else "instrument-errors-retained", "mechanics_errors": global_errors,
        "counts": counts, "pre_release_counters": pre, "typed_store": typed["counts"], "prefixes": prefix_reports,
        "external_weights": external, "weight_checks": checks, "numerical_forecasts": numerical, "rows": reports,
        "cross_scene": cross_scene, "error_contrast_scales": scales,
        "audit_sources": {"summarize_context_window.py": h["sha"](Path(__file__).with_name("summarize_context_window.py").read_bytes())},
        "semantic_hypothesis": h["read"](data / "semantic-hypothesis.json"), "semantic_status": "ungated color/centroid diagnostics; visual shape and relation review separate",
        "terminal_status": "unassessed", "RGB_forecast": False,
        "limits": [*summary["limits"], "No model, equation module or native replay was executed by this saved-array audit.",
            "External full FF matrices are source/checkpoint-bound worker provenance, not independently reloaded here.",
            "Hue thresholds and component centroids are uncalibrated diagnostics, not proof of object identity or semantic alignment.",
            "Donorfull vs forecastself is the matched full-grid donor-content contrast; half transfers lack footprint-matched analytic self controls.",
            "Fixed coordinate halves are not learned object masks; all native RGB outcomes remain observations."],
        "references": {name: {"path": str(path.relative_to(Path(run_dir).resolve())), "sha256": h["sha"](path.read_bytes())} for name, path in
            {"summary": data / "summary.json", "source": data / "source.json", "timeline": data / "timeline.jsonl",
             "external_weights": data / "external-weights.json", "semantic_hypothesis": data / "semantic-hypothesis.json"}.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result["auditor_source_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "rows": len(result["rows"]), "model_calls": 0}))


if __name__ == "__main__":
    main()
