"""Saved-only full-macro / committed-micro audit of decoder replication.

No decoder, module forward, learned recurrence, or native VAE call is executed.
Full macro comparisons are distinct from primitive hashes/fixed witnesses and
post-observation, hash-matched reconstruction of a first differing primitive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "196b206b547488d26c6659e4c6d1f8a2cb75822ace3f4033ca5e26660e1f7a23"
RECOVERY_WORKER_SHA256 = "8de957590606dfbe39fdd5500075151025f266401fdac51238f891bfe13f9067"
REMAINING_AUDITOR_SHA256 = "81f77f5f8e6a24915095875f1373097616c03b62c4f4983a588f0bc1fd957ea2"
DECODER_HELPER_SHA256 = "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f"
PRIOR_HELPER_SHA256 = "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61"
CASES = ("sham", "HH", "HL", "new-3over8")
HELPERS = ("audit_flux_functional_remaining.py", "audit_flux_functional_pass.py", "audit_flux_single_stack_trace.py",
           "audit_flux_image_tail_microscope.py", "retrieve_reader_evidence.py", "flux_family_custody.py")
BLOCK = 1 << 18


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            h.update(value)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def endpoint_comparison(predicted, observed):
    from mdb_lanes.equation_comparison import compare_equations

    return compare_equations(predicted, observed, max_elements=sum(int(value.size) for value in predicted.values()))


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def pure_helpers():
    require(sha(HERE / HELPERS[0]) == REMAINING_AUDITOR_SHA256, "qualified saved-only helper differs")
    d = runpy.run_path(str(HERE / HELPERS[0]))
    return d, d["prior"]()


def verify_external(root, records, prefixes):
    paths = [path for prefix in prefixes for path in
             ([root / prefix] if (root / prefix).is_file() else (root / prefix).rglob("*")) if path.is_file()]
    require(set(records) == {p.relative_to(root).as_posix() for p in paths}, "sealed external inventory scope differs")
    for name, row in records.items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve()) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "sealed external evidence differs")


def expected_counts(spec):
    channels = tuple(reversed(spec.get("decoder_block_out_channels") or spec["block_out_channels"]))
    levels, attention = len(channels), int(spec["mid_block_add_attention"])
    resnets, up = 2 + levels * (spec["layers_per_block"] + 1), levels - 1
    shortcuts = sum(a != b for a, b in zip(channels, channels[1:], strict=False))
    one = dict(conv2d=3 + 2 * resnets + up + shortcuts, group_norm=2 * resnets + 1 + attention,
        silu=2 * resnets + 1, linear=4 * attention, sdpa=attention, dropout_identity=resnets + attention,
        resnet=resnets, attention=attention, upsample=up, interpolate_nearest=up, mid_block=1, up_block=levels, decoder=1)
    return {key: value for key, value in one.items() if value}, sum(value for key, value in one.items() if key not in {"sdpa", "interpolate_nearest"})


class Macro:
    """One verified chunked artifact; no giant widened/full FP64 copy."""

    def __init__(self, cut, record):
        import numpy as np
        self.cut, self.record = cut, record
        payload = cut.payload
        require(payload["schema"] == "mdb-decoder-macro-chunks-v1"
                and payload["name"] == record["name"] and plain(payload["descriptor"]) == record["descriptor"]
                and payload["original_source_cut"] == record["original_source_cut"], "macro record/payload differs")
        require(plain(payload["chunk_keys"]) == record["chunk_keys"]
                and record["chunk_keys"] == [f"chunk_{i:05d}" for i in range(record["chunk_count"])], "top-level macro chunk keys differ")
        self.descriptor, self.chunks = payload["descriptor"], tuple(payload[key] for key in record["chunk_keys"])
        self.shape = tuple(self.descriptor["shape"])
        self.size = math.prod(self.shape)
        require(self.size == record["elements"] > 0 and len(self.chunks) == record["chunk_count"] > 0,
                "macro chunk geometry differs")
        digest, total, offsets = hashlib.sha256(), 0, []
        for index, tensor in enumerate(self.chunks):
            require(tensor.ndim == 1 and tensor.is_contiguous() and str(tensor.dtype) == self.descriptor["dtype"]
                    and tensor.numel() * tensor.element_size() <= 32 << 20, "bounded native macro chunk required")
            require(tensor.numel() == min(record["chunk_elements"], self.size - total), "macro chunks are not complete ordered flats")
            values = tensor.view(__import__("torch").uint16).numpy() if str(tensor.dtype) == "torch.bfloat16" else tensor.numpy()
            digest.update(values.tobytes())
            offsets.append(total)
            total += tensor.numel()
        require(total == self.size and digest.hexdigest() == self.descriptor["sha256"], "full macro raw content differs")
        self.offsets = offsets
        self.dtype = np.dtype("uint16") if self.descriptor["dtype"] == "torch.bfloat16" else self.chunks[0].numpy().dtype
        if "preview" in self.descriptor:
            require(self.pieces(0, min(8, self.size)).tolist() == list(self.descriptor["preview"]), "macro preview differs from full values")

    def pieces(self, start, stop, *, bits=False):
        import bisect

        import numpy as np
        values = []
        while start < stop:
            index = bisect.bisect_right(self.offsets, start) - 1
            offset = start - self.offsets[index]
            size = min(stop - start, self.chunks[index].numel() - offset)
            tensor = self.chunks[index][offset:offset + size]
            if self.descriptor["dtype"] == "torch.bfloat16":
                array = tensor.view(__import__("torch").uint16).numpy()
                if not bits:
                    array = (array.astype(np.uint32) << 16).view(np.float32)
            else:
                array = tensor.numpy()
            values.append(array)
            start += size
        return values[0] if len(values) == 1 else np.concatenate(values)

    def small(self, *, limit=64 << 20):
        import numpy as np
        require(self.size * 4 <= limit, "full widening exceeds small-macro audit bound")
        return np.concatenate([self.pieces(i, min(i + BLOCK, self.size)) for i in range(0, self.size, BLOCK)]).reshape(self.shape)


def stream_metric(predicted, native, *, p_sham=None, n_sham=None):
    """Full signed/RMS/max accounts, bounded temporary reductions; no quantiles."""
    import numpy as np
    require(predicted.shape == native.shape, "macro pair geometry differs")
    require((p_sham is None) == (n_sham is None), "both finite-response baselines required")
    for item in (p_sham, n_sham):
        require(item is None or item.shape == predicted.shape, "finite response geometry differs")
    sums = dict(error2=0., native2=0., predicted2=0., dot=0., error=0., native=0., predicted=0.)
    changed, maxabs, bitwise = 0, 0., predicted.descriptor["dtype"] == native.descriptor["dtype"]
    for start in range(0, predicted.size, BLOCK):
        stop = min(start + BLOCK, predicted.size)
        p = predicted.pieces(start, stop).astype(np.float64)
        n = native.pieces(start, stop).astype(np.float64)
        if p_sham is not None:
            p -= p_sham.pieces(start, stop)
            n -= n_sham.pieces(start, stop)
        else:
            bitwise = bitwise and predicted.pieces(start, stop, bits=True).tobytes() == native.pieces(start, stop, bits=True).tobytes()
        require(np.isfinite(p).all() and np.isfinite(n).all(), "nonfinite macro evidence")
        e = p - n
        sums["error2"] += float(np.dot(e, e))
        sums["native2"] += float(np.dot(n, n))
        sums["predicted2"] += float(np.dot(p, p))
        sums["dot"] += float(np.dot(p, n))
        sums["error"] += float(e.sum())
        sums["native"] += float(n.sum())
        sums["predicted"] += float(p.sum())
        changed += int(np.count_nonzero(e))
        maxabs = max(maxabs, float(abs(e).max()))
    size = predicted.size
    return dict(orientation="functional-minus-native" if p_sham is None else "functional case-minus-sham response minus native case-minus-sham response",
        rms=math.sqrt(sums["error2"] / size), native_rms=math.sqrt(sums["native2"] / size), predicted_rms=math.sqrt(sums["predicted2"] / size),
        relative_rms=None if sums["native2"] == 0 else math.sqrt(sums["error2"] / sums["native2"]), signed_mean=sums["error"] / size,
        native_signed_mean=sums["native"] / size, predicted_signed_mean=sums["predicted"] / size,
        cosine=None if sums["native2"] * sums["predicted2"] == 0 else sums["dot"] / math.sqrt(sums["native2"] * sums["predicted2"]),
        changed_values=changed, sample_count=size, maxabs=maxabs, exact=changed == 0,
        bitwise_exact=bitwise if p_sham is None else None,
        layout_exact=predicted.descriptor["stride"] == native.descriptor["stride"],
        reduction_scope="Complete macro arrays, bounded FP64 blocks; no full-error quantiles computed.")


def compare_micro(predicted, native, witness_count):
    import numpy as np
    _, a = pure_helpers()
    require(len(predicted) == len(native), "primitive occurrence coverage differs")
    rows, first = [], None
    for ordinal, (p, n) in enumerate(zip(predicted, native, strict=True)):
        require(p["ordinal"] == n["ordinal"] == ordinal and p["id"] == n["id"]
                and p["name"] == n["name"] and p["kind"] == n["kind"]
                and len(p["inputs"]) == len(n["inputs"]), "primitive occurrence/order differs")
        fields = {}
        for name, pp, nn in [(f"input.{i}", x, y) for i, (x, y) in enumerate(zip(p["inputs"], n["inputs"], strict=True))] + [("output", p["output"], n["output"])]:
            for value in (pp, nn):
                descriptor = value["descriptor"]
                require(descriptor["dtype"] in {"torch.bfloat16", "torch.float32"} and all(type(i) is int and i > 0 for i in descriptor["shape"])
                        and len(descriptor["stride"]) == len(descriptor["shape"]) and value["full_values_retained"] is False, "primitive descriptor/coverage differs")
                size = math.prod(descriptor["shape"])
                indices = np.unique(np.linspace(0, size - 1, min(witness_count, size)).astype(np.int64)).tolist()
                require(value["indices"] == indices and len(value["values"]) == len(indices), "fixed primitive witness coordinates differ")
            comparable = pp["descriptor"]["dtype"] == nn["descriptor"]["dtype"] and pp["descriptor"]["shape"] == nn["descriptor"]["shape"]
            require(comparable, "primitive native geometry/dtype differs")
            x, y = np.asarray(pp["values"], np.float32), np.asarray(nn["values"], np.float32)
            fields[name] = dict(full_descriptor_exact={k: v for k, v in pp["descriptor"].items() if k != "stride"} == {k: v for k, v in nn["descriptor"].items() if k != "stride"},
                layout_exact=pp["descriptor"]["stride"] == nn["descriptor"]["stride"], witness=a["metric"](x, y),
                witness_bitwise_exact=x.tobytes() == y.tobytes(), scope="Full content hash commitment plus fixed-coordinate samples; no unretained full primitive norm.")
        record = dict(ordinal=ordinal, id=p["id"], name=p["name"], kind=p["kind"], fields=fields)
        rows.append(record)
        if first is None and p["output"]["descriptor"] != n["output"]["descriptor"]:
            first = dict(ordinal=ordinal, id=p["id"], name=p["name"], kind=p["kind"], field="output")
    return rows, first


def audit(root, output, *, expected_manifest, expected_request, metadata_root=None):
    import numpy as np
    from PIL import Image
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.equation_comparison import EquationCatalog
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(root), Path(output)
    d, a = pure_helpers()
    recovery = None
    if metadata_root is None:
        inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
        require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(inventory["files"]) | {"files.json"}, "complete raw inventory differs")
    else:
        metadata_root = Path(metadata_root)
        recovery = read(metadata_root / "recovery.json")
        require(recovery["schema"] == "mdb-failed-decoder-recovery-metadata-v1" and recovery["original_job_state"] == "failed"
            and recovery["final_resident_parameter_check"] == "unestablished; original check after failing catalog was not executed", "failed-source recovery scope differs")
        require(sha(metadata_root / "retained-files.json") == expected_manifest == recovery["retained_manifest_sha256"], "fresh retained inventory differs")
        inventory = read(metadata_root / "retained-files.json")
        require(inventory["schema"] == "mdb-failed-decoder-retained-files-v1" and inventory["original_job_state"] == "failed", "fresh inventory is not a runtime success manifest")
        require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(inventory["files"]), "retained failed-root inventory differs")
        for name, row in inventory["files"].items():
            p = root / name
            require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "retained failed evidence differs")
        for name, digest in recovery["derived_files"].items():
            require(Path(name).name == name and sha(metadata_root / name) == digest, "derived recovery metadata differs")
    def artifact(name):
        return metadata_root / name if recovery is not None and name in recovery["derived_files"] else root / name
    config, source, summary, contract = [read(artifact(name)) for name in ("plan.json", "source.json", "summary.json", "native-contract.json")]
    require(all(v["request_sha256"] == expected_request for v in (config, source, summary, inventory))
            and len({source["job_id"], summary["job_id"], inventory["job_id"]}) == 1, "actual job/request differs")
    expected_worker = RECOVERY_WORKER_SHA256 if recovery is not None else WORKER_SHA256
    require(source["worker_sha256"] == expected_worker and source["helper_source_sha256"] == config["helper_source_sha256"]
            and source["source_provenance"] == config.get("source_provenance")
            and config["helper_source_sha256"]["flux_functional_decoder.py"] == DECODER_HELPER_SHA256
            and config["helper_source_sha256"]["flux_functional_vae.py"] == PRIOR_HELPER_SHA256, "qualified executed source differs")
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        require(sha(artifact(Path(config[key]).name)) == config[key + "_sha256"], "copied decoder metadata differs")
    spec = read(artifact(Path(config["vae_config"]).name))
    one, native_entries = expected_counts(spec)
    require(set(summary["counts"]) == {"forecast", "native", "diagnostic"} and summary["cases"] == list(CASES), "four-case census scope differs")
    for lane in ("forecast", "native"):
        require(summary["counts"][lane] == dict.fromkeys(CASES, one), "actual functional/native primitive census differs")
    totals = summary["totals"]
    require(totals["model_loads"] == totals["decoder_loads"] == 1
            and totals["functional_decode_calls"] == totals["native_decode_calls"] == 4
            and totals["native_module_entries"] == 4 * native_entries
            and all(totals[k] == 0 for k in ("transformer_loads", "forecast_module_entries", "encodes", "noise_draws", "denoiser_transitions"))
            and 0 <= totals["diagnostic_prefix_calls"] <= 4 and summary["act_receipts"] == summary["session_continue_receipts"] == 0, "declared physical work differs")
    require(type(totals["diagnostic_module_entries"]) is type(totals["guard_refusals"]) is int
            and totals["diagnostic_module_entries"] >= 0 and totals["guard_refusals"] >= 0, "diagnostic attempt counters malformed")
    require(contract["decoder"]["prior_helper_sha256"] == PRIOR_HELPER_SHA256
            and contract["upstream_source_sha256"] == config["upstream_source_sha256"]
            and contract["dtype"] == "torch.bfloat16" and contract["tf32"] is False
            and contract["autocast"] is False and contract["inference_mode"] is True
            and contract["source_reduction_policy"] == config["source_reduction_policy"] and contract["replication"] is True, "original decoder/profile differs")
    require(contract["frontend"] == dict(input_state_device="cpu", unpack_device="cpu", bn_statistics_device="cpu", bn_sqrt_device="cpu",
            bn_multiply_device="cpu", bn_add_device="cpu", unpatch_device="cpu", decoder_argument_device=config["device"],
            transfer_boundary="After CPU unpatch, before post_quant_conv; applies to forecast, native and diagnostic lanes."),
            "original CPU frontend/decoder transfer profile differs")
    member = runpy.run_path(str(HERE / "audit_flux_single_stack_trace.py"))["cut_members"]
    original_root = Path(config["input_root"])
    original_members = set()
    for item in config["inputs"].values():
        original_members.update(member(original_root, item["store"], item["cut"], config["input_files"]))
    require(original_members == set(config["input_files"]), "input whitelist includes current decoder/future evidence")
    for name, row in config["input_files"].items():
        path = original_root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(original_root.resolve())
                and sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "source cut/page/blob differs")
    verifies, trees = {}, {}
    for name, prediction, seal in (("input", "inputs", "input_seal"), ("forecast", "cohort", "forecast_seal")):
        tx = root / (name + "-transaction")
        verifies[name] = verify_transaction(tx, expected_seal_sha256=summary[seal]["seal_sha256"], max_bytes=256 << 20,
            **({"expected_event_head": summary["transaction"]["event_head"]} if name == "forecast" else {}))
        require(verifies[name]["phase"] == "closed" and source[seal] == summary[seal], "closed transaction/seal differs")
        trees[name] = plain(load_forecast_tree(tx, read(tx / "forecasts" / (prediction + ".json"))["predictions"]))
    inp, cohort = trees["input"], trees["forecast"]
    require(inp["cases"] == source["input_records"] and cohort["input_seal"] == summary["input_seal"]
            and cohort["native_decode_calls_at_seal"] == 0 and summary["transaction"]["failure"] is None
            and not summary["transaction"]["violations"], "input/cohort exposure joins differ")
    verify_external(root, inp["external_files"], ("input-store",))
    verify_external(root, cohort["external_files"], ("forecast-store", "forecast-images", "forecast-micro", "forecast-macros.json"))
    origins = {}
    for case in CASES:
        item = inp["cases"][case]
        member(root, "input-store", item["cut"], inventory["files"])
        cut = LocalStore(root / "input-store").load(item["cut"])
        p, component = cut.payload, cut.execution["component_pass"]
        require(cut.model_identity == config["expected_source_model_identity"] and cut.fingerprint == config["inputs"][case]["cut"]
                and (p["step"], p["phase"], p["cursor"]) == (4, "project", 0)
                and all(p[k] is None for k in a["WORKING"]) and len(p["timesteps"]) == 4
                and (component["id"], component["lane"], component["case"]) == ("mdb-flux-functional-remaining-v1", "forecast", case)
                and plain(describe(p)) == item["payload"] and dict(cut.execution_point) == item["point"], "unchanged prior predicted S4 input differs")
        origins[case] = cut
    manifest = read(artifact(Path(config["parameter_manifest"]).name))
    restoration = a["restore_parameters"](manifest, config["parameter_root"], source["original_decoder_tensor_sha256"], expected_module_sha256=config["retrieval_script_sha256"])
    expected_names = set(contract["decoder"]["parameter_names"]) | {"bn.running_mean", "bn.running_var"}
    require(set(restoration["rows"]) == expected_names == set(contract["parameter_names"])
            and restoration["payload_bytes"] == contract["original_parameter_bytes"], "140 decoder / 2 upstream BN recipe closure differs")
    if config.get("profile") != "tiny-cpu":
        require(restoration["tensors"] == 142 and len(contract["decoder"]["parameter_names"]) == 140, "original production recipe coverage differs")
    bn = d["small_checkpoint_tensors"](manifest, config["parameter_root"], ("bn.running_mean", "bn.running_var"))
    macros = {lane: read(root / (lane + "-macros.json")) for lane in ("forecast", "native")}
    require(macros["forecast"] == cohort["macros"], "sealed full macro records differ")
    names = ["latent_unpacked", "latent_denormalized", "latent_unpatched", "post_quant_conv", "decoder_input", "decoder_mid",
             *[f"decoder_up_{i}" for i in range(len(spec["block_out_channels"]))], "decoded", "pixels"]
    count = len(names)
    require(summary["complete_macro_cuts"] == {"forecast": 4 * count, "native": 4 * count}, "full macro coverage differs")

    def load_macro(record, lane, case, *, ordinal=None, parent=None):
        store = lane + "-store"
        member(root, store, record["cut"], inventory["files"])
        cut = LocalStore(root / store).load(record["cut"])
        execution = cut.execution["component_decoder"]
        require(cut.model_identity == origins[case].model_identity and execution["id"] == "mdb-functional-decoder-campaign-v1"
                and execution["lane"] == lane and execution["case"] == case and execution["source_cut"] == origins[case].fingerprint
                and execution["not_session_act"] is True and execution["complete_flux_advance"] is False
                and record["original_source_cut"] == origins[case].fingerprint and cut.parent == record["parent"], "component macro lineage/source differs")
        point = dict(cut.execution_point)
        require(point["family"] == "flux2-klein-decoder" and point["logical_step"] == 4 and point["phase"] == "decoder"
                and point["operator"] == record["name"] and point["edge"] == "after", "component checkpoint point differs")
        if ordinal is not None:
            require(point["index"] == point["local_clock"] == ordinal, "component macro ordinal differs")
        if parent is not None:
            require(cut.parent == parent, "macro parent chain differs")
        result = Macro(cut, record)
        require(result.descriptor["dtype"] == ("torch.uint8" if record["name"] == "pixels" else "torch.bfloat16"), "original macro dtype differs")
        require(record["chunk_elements"] == max(1, config.get("chunk_bytes", 32 << 20) // result.chunks[0].element_size()), "declared chunk budget differs")
        return result

    for case in CASES:
        for lane in ("forecast", "native"):
            require(set(macros[lane][case]) == set(names), "declared legacy macro coverage differs")
            parent = origins[case].fingerprint
            for ordinal, name in enumerate(names):
                item = load_macro(macros[lane][case][name], lane, case, ordinal=ordinal, parent=parent)
                parent = item.cut.fingerprint
    comparisons, responses, first_macro = {}, {}, {}
    for case in CASES:
        comparisons[case], first_macro[case] = {}, None
        for name in names:
            p, n = (load_macro(macros[lane][case][name], lane, case) for lane in ("forecast", "native"))
            value = stream_metric(p, n)
            comparisons[case][name] = value
            if first_macro[case] is None and not value["bitwise_exact"]:
                first_macro[case] = name
    for case in CASES[1:]:
        responses[case] = {}
        for name in names:
            p, ps, n, ns = (load_macro(macros[lane][role][name], lane, role)
                for lane, role in (("forecast", case), ("forecast", "sham"), ("native", case), ("native", "sham")))
            responses[case][name] = stream_metric(p, n, p_sham=ps, n_sham=ns)
    micro_rows, first_micro, micros = {}, {}, {lane: {} for lane in ("forecast", "native")}
    for case in CASES:
        require(cohort["micro_files"][case] == "forecast-micro/" + case + ".json", "sealed micro file occurrence differs")
        for lane in ("forecast", "native"):
            micros[lane][case] = read(root / (lane + "-micro") / (case + ".json"))
        micro_rows[case], first_micro[case] = compare_micro(micros["forecast"][case], micros["native"][case], config.get("witness_values", 16))
        macro_micro = {"post_quant_conv.0": "post_quant_conv", "decoder.mid_block.0": "decoder_mid", "decoder.0": "decoded",
                       **{f"decoder.up_blocks.{i}.0": f"decoder_up_{i}" for i in range(len(spec["block_out_channels"]))}}
        for lane in ("forecast", "native"):
            by_id = {row["id"]: row for row in micros[lane][case]}
            for identifier, name in macro_micro.items():
                require(by_id[identifier]["output"]["descriptor"] == macros[lane][case][name]["descriptor"], "macro/actual primitive checkpoint output join differs")
            require(macros[lane][case]["post_quant_conv"]["descriptor"] == macros[lane][case]["decoder_input"]["descriptor"], "after-post-quant actual decoder entry differs")
            expected_input = d["decoder_preinput"](origins[case].payload, bn, spec, config["height"], config["width"])
            before_quant = load_macro(macros[lane][case]["latent_unpatched"], lane, case).small()
            require(expected_input.tobytes() == before_quant.tobytes() and expected_input.shape == before_quant.shape, "own latent/IDs/BN/unpatch decoder argument differs")
            decoded = load_macro(macros[lane][case]["decoded"], lane, case).small()
            pixels = load_macro(macros[lane][case]["pixels"], lane, case).small()
            image = np.clip(a["rne_bf16"](a["rne_bf16"](decoded * np.float32(.5)) + np.float32(.5)), 0, 1).transpose(0, 2, 3, 1)
            require(np.array_equal(np.round(image * 255).astype(np.uint8), pixels), "decoded output/pixel coupling differs")
            png = root / (lane + "-images") / (case + ".png")
            require(pixels.shape == (1, config["height"], config["width"], 3) and pixels.dtype == np.uint8
                    and np.array_equal(pixels[0], np.asarray(Image.open(png).convert("RGB"))), "raw pixels/original PNG differs")
    observation = plain(load_forecast_tree(root / "forecast-transaction", read(root / "forecast-transaction/observations/native_decoders_and_optional_diagnostics.json")["value"]))
    sidecars = read(root / "comparisons.json")
    require(observation["native_macros"] == macros["native"] and observation["counts"] == summary["counts"]
            and observation["totals"] == totals and observation["diagnostics"] == summary["diagnostics"] == sidecars["diagnostics"]
            and observation["first_micro_difference"] == summary["first_micro_difference"] == sidecars["first_micro_difference"]
            and summary["first_macro_difference"] == sidecars["first_macro_difference"] == first_macro, "closed observation/sidecar lineage differs")
    diagnostics = {}
    for case in CASES:
        observed = summary["first_micro_difference"][case]
        diag = summary["diagnostics"][case]
        if first_micro[case] is None:
            require(observed is None and diag == dict(status="not-needed", prefix_calls=0)
                    and not summary["counts"]["diagnostic"][case], "unneeded diagnostic work differs")
            diagnostics[case] = diag
            continue
        first = first_micro[case]
        require(all(observed[k] == first[k] for k in ("id", "ordinal", "name", "kind"))
                and observed["sealed_forecast"] == micros["forecast"][case][first["ordinal"]]["output"]
                and observed["actual_native"] == micros["native"][case][first["ordinal"]]["output"], "first differing primitive join differs")
        n = load_macro(observed["native_full"], "diagnostic", case, ordinal=first["ordinal"], parent=origins[case].fingerprint)
        require(plain(n.descriptor) == observed["actual_native"]["descriptor"] and diag["native_full"] == observed["native_full"]
                and diag["prefix_calls"] == 1 and diag["unchanged_input"] == origins[case].fingerprint
                and diag["helper_sha256"] == DECODER_HELPER_SHA256 and diag["original_sealed"] == observed["sealed_forecast"], "post-observation reconstruction source differs")
        restored = dict(diag)
        prefix_counts = summary["counts"]["diagnostic"][case]
        counter_valid = all(type(value) is int and 0 <= value <= one.get(kind, -1) for kind, value in prefix_counts.items())
        restored["prefix_census_valid"] = counter_valid
        restored["unmatched_count_kinds"] = sorted(set(prefix_counts) - set(one))
        if diag["status"] == "hash-matched-reconstruction":
            p = load_macro(diag["forecast_full"], "diagnostic", case, ordinal=first["ordinal"], parent=origins[case].fingerprint)
            sealed = observed["sealed_forecast"]["descriptor"]
            require(all(plain(p.descriptor[k]) == sealed[k] for k in ("sha256", "shape", "dtype"))
                    and plain(p.descriptor) == diag["replay_descriptor"], "reconstructed tensor does not match original pre-target hash")
            restored["independently_reduced_full_pair"] = stream_metric(p, n) if counter_valid else None
            restored["diagnostic_integrity_status"] = "verified-hash-matched-reconstruction" if counter_valid else "instrument-broken-prefix-census"
        else:
            require("forecast_full" not in diag and "full_pair_metric" not in diag, "unmatched reconstructed tensor claimed as original full forecast")
            restored["independently_reduced_full_pair"] = None
            restored["diagnostic_integrity_status"] = "reconstruction-unavailable-or-unmatched"
        diagnostics[case] = restored
    require(sum(v["prefix_calls"] for v in summary["diagnostics"].values()) == totals["diagnostic_prefix_calls"], "conditional diagnostic census differs")
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    positions = lambda kind: [(i, row) for i, row in enumerate(events) if row["kind"] == kind]
    require(len(positions("inputs_sealed")) == len(positions("cohort_globally_sealed")) == 1, "unique global seal events required")
    i0, seal0 = positions("inputs_sealed")[0]
    ic, seal = positions("cohort_globally_sealed")[0]
    require(seal0["seal"] == summary["input_seal"] and seal["seal"] == summary["forecast_seal"]
            and seal["native_decode_calls"] == 0, "global forecast chronology differs")
    forecasts = positions("forecast_case_completed")
    require([row["case"] for _, row in forecasts] == list(CASES)
            and all(i0 < i < ic and row["module_entries"] == 0 and row["primitive_outputs"] == len(micros["forecast"][row["case"]]) for i, row in forecasts), "forecast before native/module coverage differs")
    entries, after_quant, native_done = [positions(kind) for kind in ("native_decode_entry", "native_decoder_entry", "native_case_completed")]
    require(all([row["case"] for _, row in panel] == list(CASES) for panel in (entries, after_quant, native_done)), "four actual native calls differ")
    for number, ((before_i, entry), (after_i, post), (done_i, done)) in enumerate(zip(entries, after_quant, native_done, strict=True)):
        case = CASES[number]
        require(ic < before_i < after_i < done_i and done["native_decode_calls"] == number + 1
                and entry["actual_before_post_quant_descriptor"] == macros["native"][case]["latent_unpatched"]["descriptor"]
                and post["actual_after_post_quant_descriptor"] == macros["native"][case]["decoder_input"]["descriptor"], "actual native pre/post-quant operands/chronology differ")
    starts, finishes = positions("diagnostic_prefix_started"), positions("diagnostic_prefix_completed")
    cases_with_misses = [case for case in CASES if first_micro[case] is not None]
    require([row["case"] for _, row in starts] == [row["case"] for _, row in finishes] == cases_with_misses, "conditional prefix occurrence order differs")
    for (i, start), (j, finish) in zip(starts, finishes, strict=True):
        case = start["case"]
        require(i > native_done[-1][0] and i < j and start["native_decode_calls"] == finish["native_decode_calls"] == 4
                and finish["counts"] == summary["counts"]["diagnostic"][case] and finish["status"] == summary["diagnostics"][case]["status"], "post-observation replay chronology/census differs")
    result = dict(schema="mdb-functional-decoder-audit-v1", job=summary["job_id"], native_request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, worker_sha256=expected_worker, auditor_sha256=sha(__file__), source_provenance=source["source_provenance"],
        source_model_identity=config["expected_source_model_identity"], input_records=source["input_records"], transactions=verifies,
        native_contract=contract, totals=totals, counts=summary["counts"], parameter_restoration=restoration,
        macro_records=macros, macro_comparisons=comparisons, signed_pair_responses=responses,
        primitive_witnesses=micro_rows, first_macro_difference=first_macro, first_micro_descriptor_difference=first_micro, diagnostic_reconstruction=diagnostics,
        diagnostic_instrument=dict(module_entry_attempts=totals["diagnostic_module_entries"], guard_refusals=totals["guard_refusals"],
            scope="Optional post-observation prefix refusals are separate from primary forecast mechanics. Frozen global pre-hook refuses before original module.forward; aggregate attempts cannot be assigned to individual cases without per-case evidence.",
            zero_attempts=totals["diagnostic_module_entries"] == 0, primary_evidence_preserved=True),
        coverage=dict(full_macros_per_case=count, cases=4, full_micro_arrays=False, primitive_scope="Full hashes and fixed witnesses; first native differing output plus conditional post-observation hash-matched forecast reconstruction only.",
                      chunks_max_bytes=32 << 20, full_error_quantiles=False),
        scope="Four previously observed endpoint replications using the unchanged original Torch primitive graph. No first functional VAE, new prompt, independent kernel arithmetic, family induction or denoiser forecast claim.",
        mechanics_status="authenticated-full-macro-saved-evidence-audit", trend_status="observed", terminal_status="unassessed", new_model_calls=0, new_reference_forward_calls=0)
    if recovery is not None:
        result.update(recovery=recovery, producer_status="failed", mechanics_status="authenticated-recovered-full-macro-saved-evidence-audit",
            final_resident_parameter_check=recovery["final_resident_parameter_check"], declared_profile_coverage="Derived from authenticated executed source/config; missing original final metadata export is not a fresh runtime observation.")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "macro-index.json", macros)
    catalog, queries = EquationCatalog(output / "catalog/equations"), []
    for case in CASES:
        context = dict(case=case, source_cut=origins[case].fingerprint, clock="step4/project0", model=config["expected_source_model_identity"],
            helper_sha256=DECODER_HELPER_SHA256, prior_helper_sha256=PRIOR_HELPER_SHA256, parameter_manifest_sha256=config["parameter_manifest_sha256"], forecast_seal=summary["forecast_seal"]["seal_sha256"])
        values = {lane: {name: load_macro(macros[lane][case][name], lane, case).small() for name in ("decoded", "pixels")} for lane in ("forecast", "native")}
        record = catalog.record(law_id="law.flux.functional-decoder-replication.v1", route_id="route.flux.shared-native-primitives.decode.v1", alias="cmp.flux.decoder-audit." + case,
            binding=dict(model_identity=context["model"], execution=dict(id="mdb-functional-decoder-campaign-v1", dtype="torch.bfloat16", replication=True),
                input_identity=context["source_cut"], source_identity=expected_request, frontier=context["clock"], horizon=dict(decoder_passes=1)),
            report=endpoint_comparison(values["forecast"], values["native"]), conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()], tested_context=context,
            readers=["same-latent-native-VAE"], evidence=dict(audit_sha256=sha(output / "audit.json"), replication=True, independent_kernel_arithmetic=False, dependent_cases=4))
        exact, wrong = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to={**context, "clock": "step3/project0"}))
        require(exact == 1 and wrong == 0, "catalog exact-context applicability differs")
        queries.append(dict(case=case, fingerprint=record["fingerprint"], exact=exact, wrong_clock=wrong))
    save(output / "catalog/provenance.json", dict(records=4, queries=queries, dependent_cases=4, family_induction=False,
        signed_responses=3, full_original_micro_retention=False, scope="Four endpoint replication comparisons; signed pair accounts remain complete in audit and are not independent tasks."))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged audit source differs")
    root, started = Path(config["native_root"]), time.monotonic()
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    original = read(root / "plan.json")
    for name in ("plan.json", "source.json", "summary.json", "native-contract.json", "files.json", "comparisons.json", "signed-responses.json", "forecast-macros.json", "native-macros.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        shutil.copyfile(root / Path(original[key]).name, output / Path(original[key]).name)
    for directory in ("forecast-images", "native-images"):
        shutil.copytree(root / directory, output / directory)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots={"native": str(root)}, manifests={"native": {name: row["sha256"] for name, row in inventory["files"].items()}}, scope="Complete macro chunks and primitive witnesses remain near data; individual checksum retrieval, no raw bulk collection."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-functional-decoder-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], source_job=result["job"],
        records=4, original_pngs=8, output_root=str(output), seconds=time.monotonic() - started, model_calls=0, mechanics_status="verified-saved-evidence"))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output), files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


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
        print(json.dumps(launch["submit"](args.run, worker=Path(__file__)) if args.submit else launch["stage"](args.run, args.study, worker=Path(__file__), helpers=HELPERS,
            experiment="mdb-functional-decoder-audit", operations=4,
            scope="Four decoder replications, full macro chunks and primitive commitments, explicit diagnostic reconstruction and exact142 recipes; no model or neural forward.")))
