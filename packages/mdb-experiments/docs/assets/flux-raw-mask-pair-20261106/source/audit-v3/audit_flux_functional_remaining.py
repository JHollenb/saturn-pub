"""Saved-only audit of four sealed two-pass continuations and real VAE consumers.

No learned recurrence, model construction, native forward or independent decoder
prediction is performed. Earlier forecast lineage and observed decoder work are
kept distinct from the newly sealed remaining-pass horizon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import struct
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "71e2a82f47f3a29cc8195589ee20018f74e880cc6696fa47ebc1e04e78aa993e"
PASS_AUDITOR_SHA256 = "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80"
CASES = ("sham", "HH", "HL", "new-3over8")
HELPERS = ("audit_flux_functional_pass.py", "audit_flux_single_stack_trace.py",
           "audit_flux_image_tail_microscope.py", "retrieve_reader_evidence.py", "flux_family_custody.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prior():
    require(sha(HERE / HELPERS[0]) == PASS_AUDITOR_SHA256, "qualified pure audit helper differs")
    return runpy.run_path(str(HERE / HELPERS[0]))


def schedule(joints, singles):
    a = prior()
    before, after = a["schedule"](joints, singles)
    return ([{**row, "step": step} for step in (2, 3) for row in before],
            [{**row, "step": step + int(row["phase"] == "project")} for step in (2, 3) for row in after])


def counts(joints, singles):
    value = prior()["expected_counts"](joints, singles)
    return {lane: {key: 2 * n for key, n in value[lane].items()} for lane in ("forecast", "native")}


def archived_array(root, record, *, bf16=False):
    import numpy as np
    p = Path(record["path"])
    require(not p.is_absolute() and ".." not in p.parts, "unsafe decoder archive")
    path = root / p
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
            and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "decoder archive differs")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"], "decoder archive geometry differs")
    if bf16:
        require(value.dtype == np.uint16 and record["encoding"] == "bf16-bits-v1", "lossless decoder BF16 bits required")
        descriptor = record["native_descriptor"]
        require(descriptor["dtype"] == "torch.bfloat16" and descriptor["shape"] == record["shape"]
                and hashlib.sha256(value.astype("<u2", copy=False).tobytes()).hexdigest() == descriptor["sha256"], "actual decoder operand bits/descriptor differ")
        decoded = (value.astype(np.uint32) << 16).view(np.float32)
        require(np.isfinite(decoded).all(), "nonfinite decoder archive")
        return decoded
    return value


def small_checkpoint_tensors(manifest, checkpoint, names):
    """Load only already fully verified small BN tensors, never a decoder bank."""
    import numpy as np
    result = {}
    for entry in manifest["artifacts"]:
        source = entry["source"]
        if source["tensor"] not in names:
            continue
        require(source["dtype"] == "BF16", "normalization tensor dtype differs")
        with (Path(checkpoint) / source["file"]).open("rb") as stream:
            start = 8 + struct.unpack("<Q", stream.read(8))[0]
            lo, hi = source["data_offsets"]
            stream.seek(start + lo)
            raw = stream.read(hi - lo)
        require(hashlib.sha256(raw).hexdigest() == source["payload_sha256"], "normalization bytes differ")
        result[source["tensor"]] = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(source["shape"])
    require(set(result) == set(names), "normalization coverage differs")
    return result


def decoder_preinput(payload, bn, spec, height, width):
    """Pure saved-array ID scatter, BF16 BN inverse and 2x2 unpatchification."""
    import numpy as np
    a = prior()
    z, ids = a["numpy"](payload["latent"]), a["numpy"](payload["img_ids"])
    scale = 2 ** (len(spec["block_out_channels"]) - 1)
    h, w = height // (scale * 2), width // (scale * 2)
    require(z.shape[0] == 1 and ids.shape[:2] == z.shape[:2], "decoder B1 ID geometry differs")
    flat = ids[0, :, 1] * w + ids[0, :, 2]
    require(len(set(flat.tolist())) == len(flat) and len(flat) == h * w, "decoder ID scatter is not complete")
    unpacked = np.zeros((h * w, z.shape[-1]), np.float32)
    unpacked[flat] = z[0]
    unpacked = unpacked.reshape(h, w, z.shape[-1]).transpose(2, 0, 1)[None]
    mean = bn["bn.running_mean"].reshape(1, -1, 1, 1)
    variance = bn["bn.running_var"].reshape(1, -1, 1, 1)
    std = a["rne_bf16"](np.sqrt(a["rne_bf16"](variance + np.float32(spec["batch_norm_eps"]))))
    normalized = a["rne_bf16"](a["rne_bf16"](unpacked * std) + mean)
    b, c, h, w = normalized.shape
    return normalized.reshape(b, c // 4, 2, 2, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np
    from PIL import Image
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    a, root, output = prior(), Path(root), Path(output)
    inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(inventory["files"]) | {"files.json"}, "raw manifest coverage differs")
    config, source, summary, contract = [read(root / n) for n in ("plan.json", "source.json", "summary.json", "native-contract.json")]
    require(all(row["request_sha256"] == expected_request for row in (config, source, summary, inventory))
            and len({source["job_id"], summary["job_id"], inventory["job_id"]}) == 1, "raw request/job differs")
    require(source["worker_sha256"] == WORKER_SHA256 and source["helper_source_sha256"] == config["helper_source_sha256"]
            and source["source_provenance"] == config.get("source_provenance"), "executed source/provenance differs")
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script", "vae_config", "vae_parameter_manifest"):
        require(sha(root / Path(config[key]).name) == config[key + "_sha256"], "copied metadata differs: " + key)
    spec = read(root / Path(config["transformer_config"]).name)
    joints, singles = spec["num_layers"], spec["num_single_layers"]
    before, after = schedule(joints, singles)
    length, expected = len(before), counts(joints, singles)
    require(summary["counts"] == expected and summary["cases"] == list(CASES), "exact primitive/native-module census differs")
    require(summary["totals"] == dict(model_loads=1, decoder_loads=1, reference_cut_reads=4, encodes=0, noise_draws=0,
            vae_calls=8, predicted_decoder_calls=4, native_reference_decoder_calls=4, guard_refusals=0)
            and summary["complete_forecast_cuts"] == summary["complete_native_cuts"] == 4 * length
            and summary["act_receipts"] == summary["session_continue_receipts"] == 0, "loads/guards/coverage differ")
    require(contract["source_model_identity"] == config["expected_source_model_identity"] and contract["verification_adapter_identity"]
            and contract["actual_attention_backend"] == "native" and contract["forecast_helper_sha256"] == config["helper_source_sha256"]["flux_functional_denoiser.py"]
            and contract["dtype"] == "torch.bfloat16" and contract["source_reduction_policy"] == config["source_reduction_policy"]
            and contract["tf32"] is False and contract["autocast"] is False and contract["deterministic"] is True, "original numerical contract differs")
    member = runpy.run_path(str(HERE / "audit_flux_single_stack_trace.py"))["cut_members"]

    def load(base, store, identifier, records):
        member(base, store, identifier, records)
        cut = LocalStore(base / store).load(identifier)
        require(cut.model_identity == config["expected_source_model_identity"], "source identity replaced by verification adapter")
        return cut

    def original_closure(base, sources, records):
        members = set()
        for item in sources.values():
            members.update(member(base, item["store"], item["cut"], records))
        require(members == set(records), "original closure includes intermediate/future evidence")
        for name, row in records.items():
            path = base / name
            require(not path.is_symlink() and path.resolve().is_relative_to(base.resolve())
                    and sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "original closure bytes differ")

    original_closure(Path(config["input_root"]), config["inputs"], config["input_files"])
    trees, verifies = {}, {}
    for name, forecast, seal_key in (("input", "inputs", "input_seal"), ("latent-forecast", "cohort", "latent_seal"), ("forecast", "cohort", "forecast_seal")):
        tx = root / (name + "-transaction")
        verifies[name] = verify_transaction(tx, expected_seal_sha256=summary[seal_key]["seal_sha256"],
            **({"expected_event_head": summary["transaction"]["event_head"]} if name == "forecast" else {}), max_bytes=256 << 20)
        require(verifies[name]["phase"] == "closed", "transaction not closed")
        trees[name] = a["plain"](load_forecast_tree(tx, read(tx / "forecasts" / (forecast + ".json"))["predictions"]))
        require(source[seal_key] == summary[seal_key], "source seal differs")
    inp, latent, cohort = (trees[k] for k in ("input", "latent-forecast", "forecast"))
    require(inp["cases"] == source["input_records"] and inp["target_entries"] == 0
            and cohort["input_seal"] == summary["input_seal"] and cohort["latent_seal"] == summary["latent_seal"]
            and cohort["native_full_blocks_at_seal"] == latent["native_full_blocks_at_seal"] == 0
            and cohort["reference_cut_reads"] == latent["reference_cut_reads"] == 0
            and cohort["forecast_counts"] == latent["forecast_counts"] == expected["forecast"], "sealed input/cohort joins differ")
    a["verify_external"](root, inp["external_files"], ("input-store",))
    a["verify_external"](root, latent["external_files"], ("prediction-store", "forecast-ports", "forecast-rows"))
    a["verify_external"](root, cohort["external_files"], ("prediction-store", "forecast-ports", "forecast-rows", "predicted-decoder"))
    require(cohort["cases"] == latent["cases"] and summary["transaction"]["failure"] is None
            and not summary["transaction"]["violations"], "latent/image immutable program differs")
    origins, references = {}, {}
    for case in CASES:
        for lane, store, originals, key, records in (("forecast", "input-store", origins, "inputs", source["input_records"]),
                ("native", "reference-input-store", references, "references", source["reference_records"])):
            cut = load(root, store, records[case]["cut"], inventory["files"])
            p = cut.payload
            require((p["step"], p["phase"], p["cursor"]) == (2, "project", 0) and all(p[k] is None for k in a["WORKING"])
                    and dict(cut.execution_point) == a["point"](before[0]) and a["plain"](describe(p)) == records[case]["payload"], "complete source payload/reset differs")
            require(str(p["latent"].dtype) == str(p["conditioning"].dtype) == "torch.bfloat16"
                    and str(p["img_ids"].dtype) == str(p["txt_ids"].dtype) == "torch.int64", "source dtype differs")
            c = cut.execution["component_pass"]
            require(c["id"] == "mdb-flux-functional-pass-v1" and c["lane"] == lane and c["case"] == case
                    and cut.fingerprint == config[key][case]["cut"], "archived prior forecast/reference lineage differs")
            originals[case] = cut
        require(all(describe(origins[case].payload[k]) == describe(references[case].payload[k]) for k in a["CARRIED"]), "forecast/reference nuisance closure differs")
    original_closure(Path(config["reference_root"]), config["references"], config["reference_files"])
    restorations = {}
    for label, manifest_key, checkpoint_key, hashes_key in (("transformer", "parameter_manifest", "parameter_root", "original_parameter_sha256"),
            ("decoder", "vae_parameter_manifest", "vae_parameter_root", "original_decoder_tensor_sha256")):
        manifest = read(root / Path(config[manifest_key]).name)
        restorations[label] = a["restore_parameters"](manifest, config[checkpoint_key], source[hashes_key], expected_module_sha256=config["retrieval_script_sha256"])
    decoder_contract = contract["decoder_contract"]
    require(decoder_contract == source["decoder_contract"] and decoder_contract["source_sha256"] == config["decoder_source_sha256"]
            and decoder_contract["independent_vae_prediction"] is False
            and restorations["transformer"]["tensors"] == contract["original_parameter_count"]
            and restorations["transformer"]["payload_bytes"] == contract["original_parameter_bytes"]
            and restorations["decoder"]["tensors"] == decoder_contract["tensors"]
            and restorations["decoder"]["payload_bytes"] == decoder_contract["parameter_bytes"], "parameter/decoder contract differs")
    if config.get("profile") != "tiny-cpu":
        require((joints, singles) == (5, 20) and restorations["transformer"]["tensors"] == 169
                and restorations["decoder"]["tensors"] == 142, "production tensor coverage differs")
    rows = {lane: {case: read(root / (lane + "-rows") / (case + ".json")) for case in CASES} for lane in ("forecast", "native")}
    require(rows["forecast"] == cohort["cases"], "sealed rows differ")
    comparisons, first, states = {}, {}, {}
    for case in CASES:
        comparisons[case], first[case], states[case] = [], None, {}
        for lane, store, parent_cut in (("forecast", "prediction-store", origins[case]), ("native", "native-store", references[case])):
            require(len(rows[lane][case]) == length, "remaining pass coverage differs")
            previous = parent_cut
            for number, row in enumerate(rows[lane][case]):
                require(row["number"] == number and row["before"] == before[number], "own-state cursor sequence differs")
                cut = load(root, store, row["state"]["cut"], inventory["files"])
                p, old, c = cut.payload, previous.payload, cut.execution["component_pass"]
                require(cut.parent == row["state"]["parent"] == previous.fingerprint and cut.boundary == row["state"]["boundary"]
                        and dict(cut.execution_point) == row["state"]["point"] == a["point"](after[number])
                        and {k: p[k] for k in ("step", "phase", "cursor")} == after[number], "complete lineage/clock differs")
                require(c["id"] == "mdb-flux-functional-remaining-v1" and c["lane"] == lane and c["case"] == case
                        and c["source_cut"] == parent_cut.fingerprint and c["verification_adapter_identity"] == contract["verification_adapter_identity"], "component source identity differs")
                require(row["state"]["fields"] == {k: describe(p[k]) for k in ("image", "text", "latent", "context", "noise_prediction")}, "state sidecar differs")
                require(all(describe(p[k]) == describe(parent_cut.payload[k]) for k in a["CARRIED"]), "carried closure refreshed")
                phase = before[number]["phase"]
                if phase != "scheduler":
                    require(describe(p["latent"]) == describe(old["latent"]), "latent changed before commit")
                if phase in ("joint", "single", "readout"):
                    require(describe(p["context"]) == describe(old["context"]), "own context refreshed")
                if phase == "readout":
                    require(all(describe(p[k]) == describe(old[k]) for k in ("image", "text")), "readout changed streams")
                require((p["noise_prediction"] is not None) == (after[number]["phase"] == "scheduler")
                        and all((p[k] is None) == (after[number]["phase"] == "project") for k in ("image", "text", "context")), "working field lifetime differs")
                previous = cut
            states[case][lane] = [r["state"] for r in rows[lane][case]]
        for number in range(length):
            p = LocalStore(root / "prediction-store").load(rows["forecast"][case][number]["state"]["cut"])
            n = LocalStore(root / "native-store").load(rows["native"][case][number]["state"]["cut"])
            fields = a["compare_states"](p.payload, n.payload)
            pp, np_ = [read(root / rows[lane][case][number]["ports"]) for lane in ("forecast", "native")]
            require(pp["before"] == np_["before"] == before[number], "primitive clock differs")
            ports = a["witness_comparison"](pp["ports"], np_["ports"], config.get("witness_values", 16))
            semantics = {k: "committed-latent" if k == "latent" and before[number]["phase"] == "scheduler"
                else "carried-latent" if k == "latent" else "readout-noise" if k == "noise_prediction"
                else "own-project-context-carried" if k.startswith("context.") else "working-stream" for k in fields}
            comparisons[case].append(dict(number=number, before=before[number], fields=fields, field_semantics=semantics, primitive_witnesses=ports))
            bad_fields = [k for k, v in fields.items() if not v["bitwise_exact"]]
            bad_ports = [k for k, v in ports["ports"].items() if not v["full_descriptor_exact"] or not v["witness_bitwise_exact"]]
            if first[case] is None and (bad_fields or bad_ports or ports["unmatched_native_ports"]):
                first[case] = dict(number=number, before=before[number], fields=bad_fields, primitive_ports=bad_ports)
    finite = []
    for number in range(length):
        sham = {lane: a["fields"](LocalStore(root / store).load(rows[lane]["sham"][number]["state"]["cut"]).payload)
                for lane, store in (("forecast", "prediction-store"), ("native", "native-store"))}
        panel = {}
        for case in CASES[1:]:
            values = {lane: a["fields"](LocalStore(root / store).load(rows[lane][case][number]["state"]["cut"]).payload)
                      for lane, store in (("forecast", "prediction-store"), ("native", "native-store"))}
            panel[case] = {}
            for key in ("image", "text", "latent", "noise_prediction"):
                if key in values["native"]:
                    d = {lane: values[lane][key].astype(np.float64) - sham[lane][key] for lane in values}
                    panel[case][key] = dict(a["metric"](d["forecast"], d["native"]), native_signed_mean=float(d["native"].mean()), predicted_signed_mean=float(d["forecast"].mean()))
        finite.append(dict(number=number, before=before[number], responses=panel))
    consumers, consumer_index = {}, {}
    vae_spec = read(root / Path(config["vae_config"]).name)
    vae_manifest = read(root / Path(config["vae_parameter_manifest"]).name)
    bn = small_checkpoint_tensors(vae_manifest, config["vae_parameter_root"], ("bn.running_mean", "bn.running_var"))
    observed = a["plain"](load_forecast_tree(root / "forecast-transaction", read(root / "forecast-transaction/observations/native_targets_and_reference_decoders.json")["value"]))
    predicted_observed = a["plain"](load_forecast_tree(root / "latent-forecast-transaction", read(root / "latent-forecast-transaction/observations/real_predicted_latent_decoders.json")["value"]))
    require(observed["final_cuts"] == {case: rows["native"][case][-1]["state"]["cut"] for case in CASES}, "actual native final output join differs")
    final_arrays = {lane: {} for lane in ("forecast", "native")}
    for case in CASES:
        consumers[case], consumer_index[case] = {}, {}
        decoded_arrays = {}
        for origin, lane, store, prefix in (("predicted", "forecast", "prediction-store", "predicted-decoder"), ("native_reference", "native", "native-store", "native-reference-decoder")):
            record = read(root / prefix / case / "consumer.json")
            require(record == observed["consumers"][case][origin] and (origin != "predicted" or record == cohort["decoded_consumers"][case][origin] == predicted_observed[case][origin]), "sealed decoder consumer sidecar differs")
            cut = LocalStore(root / store).load(rows[lane][case][-1]["state"]["cut"])
            require(record["origin"] == origin and record["input_cut"] == cut.fingerprint
                    and record["consumed_state"] == describe(cut.payload) and record["independent_decoder_prediction"] is False
                    and record["native_future_transitions"] == (0 if origin == "predicted" else 4 * length)
                    and record["decoder_contract_sha256"] == a["identity"](decoder_contract), "actual decoder source/state join differs")
            values = {key: archived_array(root, record[key], bf16=True) for key in ("decoder_input", "actual_decoder_input", "decoded")}
            rebuilt = decoder_preinput(cut.payload, bn, vae_spec, config["height"], config["width"])
            require(rebuilt.shape == values["decoder_input"].shape and rebuilt.tobytes() == values["decoder_input"].tobytes(), "own committed latent decoder pre-input differs")
            pixels = archived_array(root, record["pixels"])
            png = root / record["png"]["path"]
            require(sha(png) == record["png"]["sha256"] and png.stat().st_size == record["png"]["bytes"]
                    and pixels.dtype == np.uint8 and list(pixels.shape) == [config["height"], config["width"], 3]
                    and np.array_equal(pixels, np.asarray(Image.open(png).convert("RGB"))), "actual PNG/pixels coupling differs")
            # This is only deterministic saved decoded-output postprocessing,
            # never a reconstruction of the native decoder.
            image = np.clip(a["rne_bf16"](a["rne_bf16"](values["decoded"] * np.float32(.5)) + np.float32(.5)), 0, 1)[0].transpose(1, 2, 0)
            require(np.array_equal(np.round(image * 255).astype(np.uint8), pixels), "observed decoded output/pixel postprocessing differs")
            decoded_arrays[origin] = {**values, "pixels": pixels}
            consumers[case][origin] = dict(record=record, pre_decoder_input_join="Own committed latent/IDs plus original CPU BN statistics and 2x2 unpatch exact BF16 bytes",
                post_quant_conv_scope="Actual decoder args[0] full BF16 archive and live descriptor captured by None-returning prehook; post_quant_conv is observed native work, not independently reconstructed.")
            consumer_index[case][origin] = record
            final_arrays[lane][case] = a["numpy"](cut.payload["latent"])
        consumers[case]["comparison"] = {key: a["metric"](decoded_arrays["predicted"][key], decoded_arrays["native_reference"][key]) for key in ("decoder_input", "actual_decoder_input", "decoded", "pixels")}
    rgb_responses = {}
    for case in CASES[1:]:
        d = {}
        for origin, prefix in (("predicted", "predicted-decoder"), ("native_reference", "native-reference-decoder")):
            records = [read(root / prefix / name / "consumer.json") for name in (case, "sham")]
            d[origin] = archived_array(root, records[0]["pixels"]).astype(np.float64) - archived_array(root, records[1]["pixels"]).astype(np.float64)
        rgb_responses[case] = dict(a["metric"](d["predicted"], d["native_reference"]), native_signed_mean=float(d["native_reference"].mean()), predicted_signed_mean=float(d["predicted"].mean()), scope="Observed native VAE response to each lane's own final latents; no mathematical decoder prediction.")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    unique = lambda kind: [i for i, event in enumerate(timeline) if event["kind"] == kind]
    positions = {}
    for kind in ("inputs_globally_sealed", "latents_globally_sealed", "cohort_globally_sealed"):
        require(len(unique(kind)) == 1, "global seal event count differs")
        positions[kind] = unique(kind)[0]
    i0, il, ic = (positions[k] for k in ("inputs_globally_sealed", "latents_globally_sealed", "cohort_globally_sealed"))
    require(i0 < il < ic and timeline[il]["seal"] == summary["latent_seal"] and timeline[ic]["seal"] == summary["forecast_seal"], "global seal order differs")
    require(timeline[i0]["seal"] == summary["input_seal"] and timeline[i0]["full_target_entries"] == 0
            and timeline[il]["native_full_blocks"] == timeline[il]["reference_cut_reads"] == 0
            and timeline[ic]["native_full_blocks"] == timeline[ic]["native_scheduler_calls"] == timeline[ic]["reference_cut_reads"] == 0, "zero pre-seal target census differs")
    for lane in ("forecast", "native"):
        events = [(i, event) for i, event in enumerate(timeline) if event["kind"] == "boundary_completed" and event["lane"] == lane]
        require(len(events) == 4 * length, "boundary event coverage differs")
        for (i, event), (case, number) in zip(events, ((c, n) for c in CASES for n in range(length)), strict=True):
            require(event["case"] == case and event["number"] == number and event["before"] == before[number]
                    and event["cut"] == rows[lane][case][number]["state"]["cut"] and (i0 < i < il if lane == "forecast" else i > ic), "boundary event/order join differs")
    require(all(i > ic for kind in ("reference_cut_read", "native_entry", "native_scheduler_entry") for i in unique(kind)), "native reference/target accessed before image seal")
    require([(timeline[i]["case"], timeline[i]["cut"]) for i in unique("reference_cut_read")] == [(case, references[case].fingerprint) for case in CASES], "native reference read order differs")
    entry_program = [("project", 0, "time_guidance_embed"), *[("joint", k, f"transformer_blocks.{k}") for k in range(joints)],
                     *[("single", k, f"single_transformer_blocks.{k}") for k in range(singles)]]
    require([(timeline[i]["case"], timeline[i]["phase"], timeline[i]["index"], timeline[i]["module"]) for i in unique("native_entry")]
            == [(case, phase, index, module) for case in CASES for _ in range(2) for phase, index, module in entry_program]
            and [timeline[i]["case"] for i in unique("native_scheduler_entry")] == [case for case in CASES for _ in range(2)], "actual native guarded program differs")
    decoded_events = [(i, event) for i, event in enumerate(timeline) if event["kind"] == "real_decoder_completed"]
    require([(e["origin"], e["case"]) for _, e in decoded_events] == [(origin, case) for origin in ("predicted", "native_reference") for case in CASES], "decoder completion ledger differs")
    for i, event in decoded_events:
        origin, case = event["origin"], event["case"]
        record = consumer_index[case][origin]
        require((il < i < ic if origin == "predicted" else i > max(unique("boundary_completed")))
                and event["input_cut"] == record["input_cut"] and event["pixels"] == record["pixels"]
                and event["native_future_transitions"] == record["native_future_transitions"], "real decoder chronology/consumer differs")
    decoder_entries = [(i, timeline[i]) for i in unique("native_decoder_entry")]
    require([(event["origin"], event["case"]) for _, event in decoder_entries] == [(origin, case) for origin in ("predicted", "native_reference") for case in CASES], "actual decoder entry census differs")
    for (entry_i, entry), (exit_i, done) in zip(decoder_entries, decoded_events, strict=True):
        require(entry_i < exit_i and entry["native_future_transitions"] == done["native_future_transitions"]
                and (il < entry_i < ic if entry["origin"] == "predicted" else entry_i > max(unique("boundary_completed"))), "actual decoder entry before own seal/complete native future")
    dc = summary["decoder_counts"]
    require(dc == contract["decoder_counts"] and dc["predicted"] == dc["native_reference"]
            and dc["predicted"]["decoder_entries"] == 4 and dc["predicted"]["module_entries"] > 4, "real decoder module census differs")
    result = dict(schema="mdb-functional-remaining-audit-v1", job=summary["job_id"], native_request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, worker_sha256=WORKER_SHA256, auditor_sha256=sha(__file__), source_provenance=source["source_provenance"],
        source_model_identity=config["expected_source_model_identity"], verification_adapter_identity=contract["verification_adapter_identity"],
        native_contract=contract, counts=expected, totals=summary["totals"], decoder_counts=dc, transactions=verifies,
        parameter_restoration=restorations, boundaries=comparisons, first_covered_bitwise_mismatch=first,
        signed_finite_responses=finite, consumers=consumers, observed_rgb_finite_responses=rgb_responses,
        forecast_horizon_per_case=length, dependent_cases=4, earlier_pass_lineage="Archived prior sealed forecast commits; stitched 84-stage lineage is not a single prospectively sealed 84-stage forecast.",
        known_control="step2/project had already been observed; first unobserved future occurrence step2/joint0",
        scope="Four complete own-state two-pass continuations sharing original primitives; conditional images use actual native VAE consumers sealed before reference futures. No independent decoder/kernel or semantic claim.",
        mechanics_status="authenticated-full-state-saved-evidence-audit", trend_status="observed", terminal_status="unassessed", new_model_calls=0, new_reference_forward_calls=0)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "state-index.json", states)
    save(output / "consumer-index.json", consumer_index)
    catalog, queries = EquationCatalog(output / "catalog/equations"), []
    for case in CASES:
        context = dict(case=case, input_cut=origins[case].fingerprint, native_reference_cut=references[case].fingerprint,
            clock=before[0], horizon=length, model=config["expected_source_model_identity"], source=expected_request, forecast_seal=summary["forecast_seal"]["seal_sha256"])
        item = catalog.record(law_id="law.flux.functional-remaining-pass.v1", route_id="route.flux.two-pass-commit.v1", alias="cmp.flux.remaining." + case,
            binding=dict(model_identity=config["expected_source_model_identity"], input_identity=context["input_cut"], source_identity=expected_request, frontier=before[0],
                         execution=dict(native_contract_sha256=a["identity"](contract), verification_adapter_identity=contract["verification_adapter_identity"]), horizon=length),
            report=compare_equations({"latent": final_arrays["forecast"][case]}, {"latent": final_arrays["native"][case]}),
            conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()], tested_context=context,
            readers=["observed-native-VAE"], evidence=dict(audit_sha256=sha(output / "audit.json"), independent_kernel_arithmetic=False, independent_decoder_prediction=False, dependent_cases=4))
        exact, wrong = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to={**context, "clock": {**before[0], "step": 3}}))
        require(exact == 1 and wrong == 0, "catalog applicability differs")
        queries.append(dict(case=case, fingerprint=item["fingerprint"], exact=exact, wrong_clock=wrong))
    save(output / "catalog/provenance.json", dict(records=4, queries=queries, dependent_cases=4, family_induction=False,
        observed_rgb_accounts=3, independent_vae_equation_records=0, scope="Terminal latent equations only; conditional RGB finite responses retained separately as observed consumer accounts."))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged audit source differs")
    root, start = Path(config["native_root"]), time.monotonic()
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    original = read(root / "plan.json")
    for name in ("source.json", "summary.json", "plan.json", "files.json", "comparisons.json", "response-accounts.json", "native-contract.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script", "vae_config", "vae_parameter_manifest"):
        shutil.copyfile(root / Path(original[key]).name, output / Path(original[key]).name)
    for prefix in ("predicted-decoder", "native-reference-decoder"):
        for case in CASES:
            target = output / prefix / case
            target.mkdir(parents=True)
            for name in ("image.png", "consumer.json"):
                shutil.copyfile(root / prefix / case / name, target / name)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots={"native": str(root)}, manifests={"native": {name: row["sha256"] for name, row in inventory["files"].items()}}, scope="All full states, decoder operands and native outputs retained near data; individual checksum retrieval."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-functional-remaining-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], source_job=result["job"],
        records=4, observed_rgb_accounts=3, output_root=str(output), seconds=time.monotonic() - start, model_calls=0, mechanics_status="verified-saved-evidence"))
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
            experiment="mdb-functional-remaining-audit", operations=224,
            scope="Four sealed two-pass graphs and eight actual VAE consumers, full saved-state/operand audits and exact 169+142 recipe restoration; no neural forwards.")))
