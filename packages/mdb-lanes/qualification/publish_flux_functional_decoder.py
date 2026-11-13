"""Publish authenticated saved-only decoder replication accounts and original PNGs.

No scheduler, model, raw activation loading, parameter hydration or learned
recurrence is used. The external compact manifest and native job/request pins
must come from the authenticated production audit collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

AUDITOR_SHA256 = "ba40ce9954219b9b561553cb66a5f1e16dcca5113335fe0b3d06b11d77c7c999"
RECOVERY_SHA256 = "52faec91105872f15fd3114d541dd72ff76ccb4c7c7c61de25372372787e2aa9"
WORKER_SHA256 = "8de957590606dfbe39fdd5500075151025f266401fdac51238f891bfe13f9067"
DECODER_SHA256 = "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f"
VAE_SHA256 = "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61"
PRIOR_JOB = "job-f7d249b6d3bb"
PRIOR_REQUEST = "00d08d3eb058fa50656b0f3deb23238e9ad38a69a2266751462b80ee2ddce93a"
PRIOR_MANIFEST = "c6baff2a5aae8c03651399226f526794e7c9df3535748f365f101586bbfab41d"
PRIOR_SEAL = "46a1266921696c0b927d1d9394f515565dd00e8077eb2e6d68e1ebc1403ba19e"
CASES = ("sham", "HH", "HL", "new-3over8")
LABELS = {"sham": "Older sham control", "HH": "Older HH control", "HL": "Older HL control", "new-3over8": "Fixed numerical 3/8 input"}
COLORS = {"sham": "#777777", "HH": "#b34525", "HL": "#287bb2", "new-3over8": "#8758a7"}
INPUT_CUTS = {"sham": "a73123009248f068f7d1571a4931c51c09ae42cfa411d5e5e479eae5907235b0",
    "HH": "8d6ece943e857c3237b66c9d649e2519507abdfb14da79925077d7a948edda80",
    "HL": "2f492798b39423581c4cfbb381fbfe8fc676e26ab434c5e8ccbd2f54b057c7e5",
    "new-3over8": "13997d6daf30ff4a069dfcd7bcd32c4941f859a19817ec948c9ace71aa259ff0"}
SCOPE = ("Four dependent decoder endpoint replications from unchanged sealed predicted final S4 latents. "
    "Left images use the explicit functional graph of original Torch primitives; right images use native SAME-LATENT VAE decode. "
    "All endpoints were already observed in the remaining-schedule campaign. No new prompt, semantic replication or independent kernel arithmetic claim. "
    "The CPU front end unpacks, applies BN sqrt/multiply/add and unpatches, then transfers the decoder argument to CUDA. "
    "The prior denoiser lineage remains 28+56=84 separately sealed transitions; macro indices here are retention clocks, including aliases, not additional denoiser or physical transitions.")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name, "unsafe compact path")
    result = root / path
    require(result.is_file() and not result.is_symlink() and all(not p.is_symlink() for p in result.parents if p != root.parent), "nonregular compact member")
    return result


def macro_names(spec):
    return ["latent_unpacked", "latent_denormalized", "latent_unpatched", "post_quant_conv", "decoder_input", "decoder_mid",
        *["decoder_up_" + str(i) for i in range(len(spec["block_out_channels"]))], "decoded", "pixels"]


def authenticate(root, expected_manifest, expected_request, expected_job):
    require(sha(root / "files.json") == expected_manifest, "authenticated compact manifest differs")
    manifest = read(root / "files.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == {*manifest, "files.json"}, "compact file inventory differs")
    for name, digest in manifest.items():
        require(sha(member(root, name)) == digest, "compact member differs: " + name)
    audit, request, plan, summary = [read(member(root, name)) for name in ("audit.json", "request.json", "plan.json", "summary.json")]
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "audit request seal differs")
    recovered = "recovery" in audit
    require(audit["schema"] == "mdb-functional-decoder-audit-v1" and audit["mechanics_status"] == ("authenticated-recovered-full-macro-saved-evidence-audit" if recovered else "authenticated-full-macro-saved-evidence-audit")
        and audit["auditor_sha256"] == AUDITOR_SHA256 and audit["worker_sha256"] == WORKER_SHA256
        and audit["new_model_calls"] == audit["new_reference_forward_calls"] == 0, "qualified saved-only source differs")
    require(request["source_hashes"]["audit_flux_functional_decoder.py"] == AUDITOR_SHA256 and (not recovered or request["source_hashes"]["recover_flux_functional_decoder.py"] == RECOVERY_SHA256) and request["config"] == plan
        and plan["native_manifest_sha256"] == audit["native_manifest_sha256"] and plan["native_request_sha256"] == audit["native_request_sha256"]
        and plan["source_job"] == summary["source_job"] == audit["job"] and summary["request_sha256"] == request["request_sha256"], "audit request/native identity differs")
    native_files = read(member(root, "source-files.json"))
    require(sha(root / "source-files.json") == audit["native_manifest_sha256"] and native_files["job_id"] == expected_job == audit["job"]
        and native_files["request_sha256"] == expected_request == audit["native_request_sha256"], "externally supplied native request/job differs")
    inventory = native_files["files"]
    if recovered:
        recovery = audit["recovery"]
        require(recovery == read(member(root, "recovery-metadata/recovery.json")) and recovery["original_job_state"] == audit["producer_status"] == summary["original_producer_status"] == native_files["original_job_state"] == "failed"
            and recovery["final_resident_parameter_check"] == audit["final_resident_parameter_check"] == "unestablished; original check after failing catalog was not executed", "recovered failed status/coverage differs")
        packet = root / "failed-source-authentication"
        auth, job = read(member(root, "failed-source-authentication/authentication.json")), read(member(root, "failed-source-authentication/job.json"))
        require(sha(packet / "authentication.json") == recovery["failed_authentication_sha256"] and sha(packet / "job.json") == recovery["failed_job_sha256"] == auth["job_sha256"]
            and sha(member(root, "failed-source-authentication/worker.log")) == recovery["failed_log_sha256"] == auth["log_sha256"]
            and job["state"] == auth["original_job_state"] == "failed" and job["result"]["returncode"] == 1
            and auth["job_id"] == job["job_id"] == expected_job and auth["request_sha256"] == expected_request
            and recovery["retained_manifest_sha256"] == audit["native_manifest_sha256"], "failed source packet/retained inventory differs")
        custody = job["payload_custody"]
        require(custody["declared"]["sha256"] == custody["sealed"]["sha256"] == custody["executed"]["sha256"] == job["result"]["executed_payload"]["sha256"] == auth["executed_payload_sha256"] == recovery["failed_executed_payload_sha256"], "failed executed payload custody differs")
    copied = ("plan.json", "source.json", "summary.json", "native-contract.json", "comparisons.json", "signed-responses.json", "forecast-macros.json", "native-macros.json", "timeline.jsonl")
    for name in copied:
        p = member(root, "source-" + name)
        if recovered and name in ("source.json", "summary.json", "native-contract.json"):
            require(sha(p) == recovery["derived_files"][name] == sha(member(root, "recovery-metadata/" + name)), "derived recovery sidecar differs: " + name)
        else:
            require(sha(p) == inventory[name]["sha256"] and p.stat().st_size == inventory[name]["bytes"], "copied raw sidecar differs: " + name)
    source, native_plan, native_summary, contract = [read(root / ("source-" + name)) for name in ("source.json", "plan.json", "summary.json", "native-contract.json")]
    require(source["job_id"] == native_summary["job_id"] == expected_job and source["request_sha256"] == native_plan["request_sha256"] == native_summary["request_sha256"] == expected_request
        and source["worker_sha256"] == WORKER_SHA256 and source["helper_source_sha256"] == native_plan["helper_source_sha256"]
        and native_plan["helper_source_sha256"]["flux_functional_decoder.py"] == DECODER_SHA256 and native_plan["helper_source_sha256"]["flux_functional_vae.py"] == VAE_SHA256, "executed native/helper provenance differs")
    provenance = audit["source_provenance"]
    require(provenance == source["source_provenance"] == native_plan["source_provenance"]
        and (provenance["source_job"], provenance["source_request_sha256"], provenance["native_manifest_sha256"], provenance["source_forecast_seal_sha256"])
        == (PRIOR_JOB, PRIOR_REQUEST, PRIOR_MANIFEST, PRIOR_SEAL), "accepted prior remaining lineage differs")
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        require(sha(member(root, Path(native_plan[key]).name)) == native_plan[key + "_sha256"], "retrieval metadata differs")
    frontend = dict(input_state_device="cpu", unpack_device="cpu", bn_statistics_device="cpu", bn_sqrt_device="cpu", bn_multiply_device="cpu", bn_add_device="cpu", unpatch_device="cpu", decoder_argument_device=native_plan["device"], transfer_boundary="After CPU unpatch, before post_quant_conv; applies to forecast, native and diagnostic lanes.")
    require(audit["native_contract"] == contract and contract["frontend"] == frontend and contract["replication"] is True
        and contract["dtype"] == "torch.bfloat16" and contract["inference_mode"] is True and contract["autocast"] is False, "CPU frontend/native decoder contract differs")
    require(set(audit["transactions"]) == {"input", "forecast"} and all(v["verified"] and v["phase"] == "closed" for v in audit["transactions"].values())
        and audit["transactions"]["forecast"]["seal_sha256"] == source["forecast_seal"]["seal_sha256"] == native_summary["forecast_seal"]["seal_sha256"], "closed global forecast seal differs")
    require(audit["totals"] == native_summary["totals"] and audit["counts"] == native_summary["counts"]
        and audit["totals"]["functional_decode_calls"] == audit["totals"]["native_decode_calls"] == 4
        and audit["totals"]["forecast_module_entries"] == audit["totals"]["denoiser_transitions"] == 0, "physical decoder work differs")
    macros = read(root / "macro-index.json")
    require(macros == audit["macro_records"] == {lane: read(root / ("source-" + lane + "-macros.json")) for lane in ("forecast", "native")}, "full macro index differs")
    names = macro_names(read(root / Path(native_plan["vae_config"]).name))
    require(audit["coverage"]["cases"] == 4 and audit["coverage"]["full_macros_per_case"] == len(names) and audit["coverage"]["full_micro_arrays"] is False
        and set(audit["input_records"]) == set(CASES) and source["input_records"] == audit["input_records"], "four-case retention coverage differs")
    events = [json.loads(line) for line in (root / "source-timeline.jsonl").read_text().splitlines()]
    seals = [(i, r) for i, r in enumerate(events) if r["kind"] == "cohort_globally_sealed"]
    require(len(seals) == 1 and seals[0][1]["native_decode_calls"] == 0 and seals[0][1]["seal"] == source["forecast_seal"], "global pretarget chronology differs")
    forecasts = [(i, r) for i, r in enumerate(events) if r["kind"] == "forecast_case_completed"]
    require([r["case"] for _, r in forecasts] == list(CASES) and all(i < seals[0][0] for i, _ in forecasts), "four forecasts must precede global seal")
    entries = {case: {} for case in CASES}
    for i, event in enumerate(events):
        if event["kind"] in ("native_decode_entry", "native_decoder_entry", "native_case_completed"):
            require(i > seals[0][0] and event["kind"] not in entries[event["case"]], "native consumer chronology differs")
            entries[event["case"]][event["kind"]] = (i, event)
    joins, micro_retrieval, misses = {}, {}, {}
    for case in CASES:
        require(audit["input_records"][case]["cut"] == native_plan["inputs"][case]["cut"] == INPUT_CUTS[case], "unchanged predicted finalS4 input differs")
        require(set(audit["macro_comparisons"][case]) == set(names), "complete macro metrics differ")
        for lane in ("forecast", "native"):
            require(set(macros[lane][case]) == set(names), "complete macro retention differs")
            parent = INPUT_CUTS[case]
            for name in names:
                row = macros[lane][case][name]
                require(row["parent"] == parent and row["name"] == name and row["original_source_cut"] == INPUT_CUTS[case], "macro parent/source chain differs")
                parent = row["cut"]
            require(macros[lane][case]["post_quant_conv"]["descriptor"] == macros[lane][case]["decoder_input"]["descriptor"], "postquant/decoder entry alias differs")
            png_name = lane + "-images/" + case + ".png"
            p = member(root, png_name)
            require(sha(p) == inventory[png_name]["sha256"] and p.stat().st_size == inventory[png_name]["bytes"], "original PNG custody differs")
            with Image.open(p) as image:
                require(image.format == "PNG" and image.size == (native_plan["width"], native_plan["height"]), "original PNG geometry differs")
                image.verify()
        before, after, done = [entries[case][key] for key in ("native_decode_entry", "native_decoder_entry", "native_case_completed")]
        require(before[0] < after[0] < done[0] and before[1]["actual_before_post_quant_descriptor"] == macros["native"][case]["latent_unpatched"]["descriptor"]
            and after[1]["actual_after_post_quant_descriptor"] == macros["native"][case]["decoder_input"]["descriptor"], "actual pre/postquant argument joins differ")
        joins[case] = dict(input_cut=INPUT_CUTS[case], before_postquant=before[1], after_postquant=after[1], completed=done[1], same_latent=True)
        rows = audit["primitive_witnesses"][case]
        require(all(row["ordinal"] == i for i, row in enumerate(rows)) and len({row["id"] for row in rows}) == len(rows), "micro retention order differs")
        misses[case] = [row for row in rows if any(not f["full_descriptor_exact"] or not f["layout_exact"] or not f["witness_bitwise_exact"] for f in row["fields"].values())]
        micro_retrieval[case] = {lane: dict(path=lane + "-micro/" + case + ".json", **inventory[lane + "-micro/" + case + ".json"]) for lane in ("forecast", "native")}
        diagnostic = audit["diagnostic_reconstruction"][case]
        if diagnostic["status"] == "hash-matched-reconstruction":
            sealed = diagnostic["original_sealed"]["descriptor"]
            require(all(diagnostic["replay_descriptor"][k] == diagnostic["forecast_full"]["descriptor"][k] == sealed[k] for k in ("sha256", "shape", "dtype")), "post-observation sealed hash proof differs")
        else:
            require("forecast_full" not in diagnostic, "unmatched diagnostic claims original forecast tensor")
    require(set(audit["signed_pair_responses"]) == set(CASES[1:]) and all(set(v) == set(names) for v in audit["signed_pair_responses"].values()), "complete signed macro responses differ")
    starts = [(i, r) for i, r in enumerate(events) if r["kind"] == "diagnostic_prefix_started"]
    finishes = [(i, r) for i, r in enumerate(events) if r["kind"] == "diagnostic_prefix_completed"]
    needed = [case for case in CASES if audit["diagnostic_reconstruction"][case]["prefix_calls"]]
    require([r["case"] for _, r in starts] == [r["case"] for _, r in finishes] == needed, "optional diagnostic occurrence differs")
    last_native = max(row["native_case_completed"][0] for row in entries.values())
    for (i, start), (j, finish) in zip(starts, finishes, strict=True):
        require(last_native < i < j and start["native_decode_calls"] == finish["native_decode_calls"] == 4
            and finish["status"] == audit["diagnostic_reconstruction"][start["case"]]["status"], "post-target diagnostic chronology differs")
    external = read(member(root, "external-evidence.json"))
    require(external["manifests"]["native"] == {name: row["sha256"] for name, row in inventory.items()}, "raw checksum retrieval inventory differs")
    return audit, manifest, summary, names, joins, micro_retrieval, misses


def zero_axis(ax):
    values = [float(v) for line in ax.lines for v in line.get_ydata() if math.isfinite(float(v))]
    ax.set_ylim(bottom=0)
    if values and not any(values):
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(.5, .5, "All plotted values exactly zero", transform=ax.transAxes, ha="center", color="#444")


def decorate(ax, names, ylabel):
    ax.set_xticks(range(len(names)), [name.replace("latent_", "").replace("decoder_", "") for name in names], rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Macro retention clock; aliases included, not physical transitions")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)


def figures(audit, names, root, output):
    prefix = "Saved-only recovery; original producer FAILED in optional reporting\n" if "recovery" in audit else ""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
    for case in CASES:
        for ax, key in zip(axes, ("rms", "relative_rms"), strict=True):
            values = [audit["macro_comparisons"][case][name][key] for name in names]
            ax.plot(range(len(names)), [float("nan") if v is None else v for v in values], "o-", label=LABELS[case], color=COLORS[case], markersize=3)
    for ax, ylabel in zip(axes, ("RMS functional − native", "Error RMS / native macro RMS"), strict=True):
        decorate(ax, names, ylabel)
        ax.legend(fontsize=8)
        zero_axis(ax)
    fig.suptitle(prefix + "Decoder replication: full macro errors; all numerical misses retained")
    fig.savefig(output / "macro-errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), layout="constrained")
    for case in CASES[1:]:
        response = audit["signed_pair_responses"][case]
        for col, field in enumerate(("rms", "signed_mean")):
            for lane, style in (("predicted", "--"), ("native", "-")):
                axes[col].plot(range(len(names)), [response[name][lane + ("_rms" if col == 0 else "_signed_mean")] for name in names], style, color=COLORS[case], marker="o", markersize=3, label=LABELS[case] + " " + lane)
        axes[2].plot(range(len(names)), [response[name]["rms"] for name in names], "o-", color=COLORS[case], markersize=3, label=LABELS[case])
    for i, ylabel in enumerate(("RMS(case − same-lane sham)", "Mean(case − same-lane sham)", "RMS functional response − native response")):
        decorate(axes[i], names, ylabel)
        axes[i].legend(fontsize=7)
        if i != 1:
            zero_axis(axes[i])
        else:
            axes[i].axhline(0, color="#888", linewidth=.5)
    fig.suptitle(prefix + "Three case-minus-sham decoder contrasts: solid native, dashed functional")
    fig.savefig(output / "signed-responses.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, name in zip(axes, ("decoded", "pixels"), strict=True):
        values = [audit["macro_comparisons"][case][name]["rms"] for case in CASES]
        ax.bar(range(4), values, color=[COLORS[case] for case in CASES])
        ax.set_xticks(range(4), [LABELS[case] for case in CASES], rotation=25, ha="right", fontsize=8)
        ax.set_title("Raw RGB (BF16)" if name == "decoded" else "Pixels (uint8)")
        ax.set_ylabel("RMS functional − native")
        if not any(values):
            ax.set_ylim(0, 1)
            ax.set_yticks([0])
            ax.text(.5, .5, "All RMS values zero", transform=ax.transAxes, ha="center")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(prefix + "Same-latent decoder output errors")
    fig.savefig(output / "rgb-pixel-errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(11, 20), layout="constrained")
    for i, case in enumerate(CASES):
        for j, lane in enumerate(("forecast", "native")):
            with Image.open(root / (lane + "-images") / (case + ".png")) as image:
                axes[i, j].imshow(image.convert("RGB"), interpolation="nearest")
            axes[i, j].axis("off")
            axes[i, j].set_title(LABELS[case] + (" — functional primitives" if lane == "forecast" else " — native SAME-LATENT"), fontsize=10)
    fig.suptitle(prefix + "Eight original PNGs: decoder replication, endpoints already observed\nLeft: explicit functional primitive forecast. Right: native same-latent decode.\nDeclared CPU unpack/BN/unpatch → CUDA decoder; no new prompt or independent kernel claim.", fontsize=12)
    fig.savefig(output / "image-contact-sheet.png", dpi=130)
    plt.close(fig)


def publish(root, output, *, expected_manifest, expected_request, expected_job, collection_receipt=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    audit, manifest, summary, names, joins, retrieval, misses = authenticate(root, expected_manifest, expected_request, expected_job)
    if collection_receipt is not None:
        collected = read(collection_receipt)
        require(collected["files_manifest_sha256"] == expected_manifest and collected["job_id"] == summary["job_id"] and collected["request_sha256"] == summary["request_sha256"], "authenticated collection receipt differs")
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    for name, target in (("audit.json", "audit-data.json"), ("macro-index.json", "macro-index.json"), ("external-evidence.json", "external-evidence.json"), ("retrieve_evidence.py", "retrieve_evidence.py")):
        shutil.copyfile(root / name, output / target)
    native_plan = read(root / "source-plan.json")
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        name = Path(native_plan[key]).name
        shutil.copyfile(root / name, output / name)
    publication = dict(schema="mdb-functional-decoder-publication-v1", scope=SCOPE, cases=list(CASES), case_labels=LABELS,
        orientation="predicted-minus-native; finite responses are case-minus-same-lane-sham", denominator="Native full-macro or native finite-response RMS at the same retention clock; null when zero.",
        source=dict(job=audit["job"], request_sha256=audit["native_request_sha256"], raw_manifest_sha256=audit["native_manifest_sha256"], compact_manifest_sha256=expected_manifest,
            audit_sha256=sha(root / "audit.json"), auditor_sha256=AUDITOR_SHA256, worker_sha256=WORKER_SHA256, decoder_helper_sha256=DECODER_SHA256, prior_vae_helper_sha256=VAE_SHA256, source_model_identity=audit["source_model_identity"]),
        input_records=audit["input_records"], source_provenance=audit["source_provenance"], native_contract=audit["native_contract"], counts=audit["counts"], totals=audit["totals"], transactions=audit["transactions"],
        macro_retention_clock=names, macro_records=audit["macro_records"], macro_comparisons=audit["macro_comparisons"], signed_pair_responses=audit["signed_pair_responses"],
        primitive_witnesses=audit["primitive_witnesses"], all_micro_misses=misses, raw_micro_retrieval=retrieval,
        first_macro_difference=audit["first_macro_difference"], first_micro_descriptor_difference=audit["first_micro_descriptor_difference"],
        diagnostic_reconstruction=audit["diagnostic_reconstruction"], diagnostic_instrument=audit["diagnostic_instrument"], actual_consumer_joins=joins,
        coverage=audit["coverage"], parameter_restoration=audit["parameter_restoration"], compact_summary=summary,
        lineage=dict(prior_denoiser_transitions=84, prior_seals="28+56; separately sealed", additional_denoiser_transitions=0, macro_indices_are_physical_transitions=False, endpoints_already_observed=True),
        limitations=["Shared original Torch primitives; no independent kernel arithmetic, new semantic input or family induction.",
            "Full macro values retained near compute in bounded chunks; full ordinary micro arrays were not retained.",
            "Micro full-hash/layout comparisons and fixed-witness errors remain separate; raw commitments/witness values are individually retrievable.",
            "Optional first-micro functional prefix reconstruction occurs after all targets; a matching original sealed hash does not make it pretarget full retention.",
            "Macro retention clocks include aliases and are not added to the 84 denoiser transitions."])
    if "recovery" in audit:
        publication.update(producer_status="failed", saved_only_audit_status="successful recovery audit", recovery=audit["recovery"],
            final_resident_parameter_check=audit["final_resident_parameter_check"], declared_profile_coverage=audit["declared_profile_coverage"],
            scope=SCOPE + " Original producer failed after all primary calls during optional catalog reporting. This separate saved-only audit recovers retained evidence; missing final resident parameter checks remain unestablished.")
        shutil.copytree(root / "failed-source-authentication", output / "failed-source-authentication")
        shutil.copytree(root / "recovery-metadata", output / "recovery-metadata")
    save(output / "publication.json", publication)
    for lane in ("forecast", "native"):
        directory = output / "images" / lane
        directory.mkdir(parents=True)
        for case in CASES:
            shutil.copyfile(root / (lane + "-images") / (case + ".png"), directory / (case + ".png"))
    figures(audit, names, root, output)
    (output / "README.md").write_text("# Decoder replication saved evidence\n\n" + publication["scope"] + "\n\nErrors are functional minus native. Responses are case minus same-lane sham; signed means and every miss remain in publication.json. Null relative error means a zero native denominator. Linear charts preserve exact zero without a logarithmic floor. The contact sheet uses all eight original PNGs; images/ preserves their exact bytes. Full macro chunks and raw micro commitments remain near compute, addressable by checksum through retrieve_evidence.py. Optional diagnostic prefix reconstruction is post-observation and separately counted; only a full tensor whose hash matches the original sealed forecast is admitted as reconstructed evidence.\n")
    receipt = dict(schema="mdb-functional-decoder-publication-receipt-v1", source_sha256=sha(__file__), compact_root=str(root), authenticated_manifest_sha256=expected_manifest,
        authentication_scope="Externally supplied authenticated compact manifest and native job/request, complete member hashes, executed source pins and saved sidecar/consumer joins.", inputs={"files.json": expected_manifest, **manifest},
        collection_receipt=None if collection_receipt is None else dict(path=str(Path(collection_receipt).resolve()), sha256=sha(collection_receipt)),
        outputs={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}, raw_arrays_collected=False, new_model_calls=0, new_numerical_recurrence=0)
    save(output / "publication-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-request", required=True)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--collection-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest, expected_request=args.expected_request, expected_job=args.expected_job, collection_receipt=args.collection_receipt), sort_keys=True))
