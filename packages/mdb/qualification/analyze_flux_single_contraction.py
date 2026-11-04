"""Offline, source-bound audit of a single6 attention/MLP contraction assay.

Only saved NumPy arrays, public receipts and multipart custody are read. Numeric
forecast errors, signed contraction and native futures are observations rather
than scientific collection gates. No model, native projection or suffix runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON = runpy.run_path(str(HERE / "analyze_flux_single_contexts.py"))
PANEL = COMMON["PANEL"]
BASE = PANEL["BASE"]
require, exact_bits, distance = (BASE[key] for key in ("require", "exact_bits", "distance"))
sha_file, read_jsonl = COMMON["sha_file"], COMMON["read_jsonl"]
PRIOR_JOB = "job-b8e38f8a326f"
PRIOR_ARCHIVE = "7ba8da5c2b0093310697760d397868932cad3b8b08fc3fd7cba01eac33c469d0"
PRIOR_AUDIT = "b28a9e223544c4e7646b9cd03ba2c500181848c28d9a71353fda90ef070da7c6"
PRIOR_REQUEST = "aa1facdaa69342c56707c305b83d3fb0b8b93dad7f5b487630405590ff7a21fb"
PRIOR_REPORT = "a8f2f6e72ef9e2f1f86c689c7d7c9fbdb4607871cfa3d33d3acecc06978436a6"
PRIOR_MANIFEST = "1c2fa006196aba700abfb1d19a1fa3fc12c2d93fa5575291279fb09036fe4013"
ROW_IDS = ("zero", "text", "sham", "A.image", "M.image", "AM.image", "text.text", "coherent", "AM.text")
COUNTS = {"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0,
          "adapter_steps": 143, "candidate_calls": 5, "acts": 10,
          "receipt_emissions": 153, "images": 9, "vae_calls": 9, "physical_batch": 1}


def vector_statistics(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.size and np.isfinite(value).all(), "invalid contraction vector")
    return {"shape": list(value.shape), "l2": float(np.linalg.norm(value.ravel())),
            "rms": float(np.sqrt(np.mean(value * value))), "max_abs": float(np.abs(value).max()),
            "signed_mean": float(value.mean()),
            "vector_float64_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def contraction_identity(low_input, high_input, low_output, high_output):
    """Actual rounded updates, never a decomposition into attention/MLP causes."""
    values = [np.asarray(x, dtype=np.float64) for x in (low_input, high_input, low_output, high_output)]
    require(all(x.shape == values[0].shape and x.size and np.isfinite(x).all() for x in values),
            "contraction input/output geometry differs")
    low_r, high_r, low_y, high_y = values
    dr, dy = high_r - low_r, high_y - low_y
    du = (high_y - high_r) - (low_y - low_r)
    cross = float(2 * np.dot(dr.ravel(), du.ravel()))
    update_energy = float(np.dot(du.ravel(), du.ravel()))
    observed = float(np.dot(dy.ravel(), dy.ravel()) - np.dot(dr.ravel(), dr.ravel()))
    return {"formula": "||dR+dU||^2-||dR||^2 = 2 dot(dR,dU)+||dU||^2; U=actual post-pre",
            "dR": vector_statistics(dr), "dU": vector_statistics(du), "dY": vector_statistics(dy),
            "cross_term": cross, "update_energy": update_energy,
            "observed_squared_distance_change": observed, "reconstructed_squared_distance_change": cross + update_energy,
            "identity_residual": observed - (cross + update_energy),
            "vector_closure": vector_statistics(dy - (dr + du)),
            "signed_l2_change": float(np.linalg.norm(dy.ravel()) - np.linalg.norm(dr.ravel())),
            "scope": "Observed rounded whole-block update geometry. No attention/MLP attribution follows from this sum alone; signed results are retained without a contraction gate."}


def feature_selection(original, used, donor, *, text_rows, attention_width, selection):
    """Check every feature, including unchanged recipient text and image slices."""
    original, used, donor = map(np.asarray, (original, used, donor))
    require(original.ndim == 3 and original.shape == used.shape == donor.shape
            and type(text_rows) is int and 0 < text_rows < original.shape[1]
            and type(attention_width) is int and 0 < attention_width < original.shape[2]
            and selection in {"native", "A", "M", "AM"}, "to_out feature partition ABI differs")
    a = donor if selection in {"A", "AM"} else original
    m = donor if selection in {"M", "AM"} else original
    controls = {"recipient_text_exact": exact_bits(used[:, :text_rows], original[:, :text_rows]),
        "image_attention_exact_selected": exact_bits(used[:, text_rows:, :attention_width], a[:, text_rows:, :attention_width]),
        "image_mlp_exact_selected": exact_bits(used[:, text_rows:, attention_width:], m[:, text_rows:, attention_width:])}
    require(all(controls.values()), "feature substitution changed an undeclared recipient partition")
    return controls


def sampled_projection(original, donor, weight_rows, bias, residual, gate, *, text_rows,
                       attention_width, selection, query_indices, output_indices, quantize):
    """Independent FP64 linear equation and explicit BF16 native-cast reference."""
    original, donor, weights = map(np.asarray, (original, donor, weight_rows))
    require(original.ndim == 3 and original.shape == donor.shape and original.shape[0] == 1
            and weights.ndim == 2 and weights.shape == (len(output_indices), original.shape[-1])
            and len(set(query_indices)) == len(query_indices) and len(set(output_indices)) == len(output_indices)
            and all(type(q) is int and 0 <= q < original.shape[1] - text_rows for q in query_indices)
            and all(type(o) is int and 0 <= o < residual.shape[-1] for o in output_indices)
            and residual.shape[:2] == original.shape[:2] and gate.ndim == 3
            and selection in {"A", "M", "AM"}, "sampled projection operands differ")
    rows = np.asarray(query_indices) + text_rows
    concat = np.asarray(original[:, rows], dtype=np.float64).copy()
    selected = np.asarray(donor[:, rows], dtype=np.float64)
    if selection in {"A", "AM"}:
        concat[..., :attention_width] = selected[..., :attention_width]
    if selection in {"M", "AM"}:
        concat[..., attention_width:] = selected[..., attention_width:]
    projection = concat @ np.asarray(weights, dtype=np.float64).T
    if bias is not None:
        require(np.asarray(bias).shape == (len(output_indices),), "projection bias geometry differs")
        projection += np.asarray(bias, dtype=np.float64)
    r = np.asarray(residual[:, rows][:, :, output_indices], dtype=np.float64)
    g = np.broadcast_to(gate, residual.shape)[:, rows][:, :, output_indices].astype(np.float64)
    projection_bf16 = quantize(projection.astype(np.float32))
    product = quantize((g * projection_bf16).astype(np.float32))
    return {"projection_fp64": projection, "projection_bf16": projection_bf16,
            "updated_fp64": r + g * projection,
            "product_bf16": product, "updated_bf16": quantize((r + product).astype(np.float32))}


def factorial_interaction(low, a, m, am):
    values = [np.asarray(x, dtype=np.float64) for x in (low, a, m, am)]
    require(all(x.shape == values[0].shape for x in values), "interaction cell geometry differs")
    return {"formula": "Y_AM.image - Y_A.image - Y_M.image + Y_low",
            **vector_statistics(values[3] - values[1] - values[2] + values[0]),
            "scope": "Full-vector factorial nonadditivity, not subtraction of distances or a semantic conjunction."}


def local_vector_account(low_pre, high_pre, low_post, high_post, a_post, m_post, am_post):
    """Exact counterfactual vector census, not a uniquely inferred causal split."""
    low_pre, high_pre, low, high, a, m, am = [np.asarray(x, dtype=np.float64)
        for x in (low_pre, high_pre, low_post, high_post, a_post, m_post, am_post)]
    require(all(x.shape == low.shape and x.size and np.isfinite(x).all()
                for x in (low_pre, high_pre, high, a, m, am)), "local vector account geometry differs")
    dr, da, dm = high_pre - low_pre, a - low, m - low
    joint, remainder = am - a - m + low, high - am
    epsilon = remainder - dr
    observed = high - low
    parts = {"dR": dr, "a": da, "m": dm, "j": joint, "r": remainder, "epsilon": epsilon, "dI_post": observed}
    return {"formula": "a=I_A-I_L; m=I_M-I_L; j=I_AM-I_A-I_M+I_L; r=I_H-I_AM; epsilon=r-dR; dI_post=dR+a+m+j+epsilon",
            "vectors": {key: vector_statistics(value) for key, value in parts.items()},
            "dot_with_dR": {key: float(np.dot(dr.ravel(), value.ravel())) for key, value in parts.items() if key != "dR"},
            "identity_residual": vector_statistics(observed - (dr + da + dm + joint + epsilon)),
            "scope": "Algebraic account of the five retained native/counterfactual postimages. A/M feature substitution enters an affine fused projection at fixed low R/gate: local j reflects precision/rounding, not a newly identified nonlinear A–MLP mechanism. r and epsilon include the different entry residual and addition rounding. Signed alignments are descriptive, not an independently unique causal or additive energy attribution. Endpoint interaction may include later nonlinear consumers."}


def endpoint_trends(comparisons):
    """Retain signed margins and misses without assigning scientific failure."""
    return {"text_direction": {name: {observable: comparisons[name]["text"][observable][metric]
                   - comparisons["text.text"]["text"][observable][metric]
                   for observable, metric in (("latent", "base_l2"), ("rgb", "base_mae"))}
                for name in ("A.image", "M.image", "AM.image")},
            "image_only_relative_to_low": {name: {observable: comparisons[name]["text"][observable][metric]
                   - comparisons["zero"]["text"][observable][metric]
                   for observable, metric in (("latent", "base_l2"), ("rgb", "base_mae"))}
                for name in ("A.image", "M.image", "AM.image")},
            "scope": "Positive text-direction margin favors text.text; positive image-only-relative-to-low margin is farther from high than low. One dependent history; no quality or monotonicity gate."}


def verify_source_inventory(payload, selection, manifest, h):
    """Authenticate original bytes through pinned producer inventory, not self hashes."""
    require(isinstance(selection["source_files"], dict) and selection["source_files"], "empty selected source inventory")
    joined = {}
    for target, record in selection["source_files"].items():
        original = record["prior_path"]
        require(isinstance(original, str) and not Path(original).is_absolute() and ".." not in Path(original).parts,
                "unsafe original selected path")
        path = h["file_path"](payload, target)
        require(manifest.get(original) == record["sha256"] == sha_file(path)
                and path.stat().st_size == record["bytes"], "selected source differs from original producer manifest")
        joined[target] = original
        if target in selection["files"]:
            require(target.startswith("input-store/")
                    and original == "typed-store/" + target.removeprefix("input-store/")
                    and selection["files"][target] == record["sha256"], "CAS/original path mapping differs")
    require(set(selection["files"]) <= set(joined), "imported CAS omitted from original manifest join")
    actual = {p.relative_to(payload).as_posix() for p in (payload / "input-store").rglob("*") if p.is_file()}
    require(actual == set(selection["files"]), "extra or missing imported CAS file")
    return {"files": len(actual), "bytes": sum((payload / p).stat().st_size for p in actual),
            "original_manifest_joined_files": len(joined)}


def native_implementation_closure(current, prior):
    def basenames(mapping):
        result = {}
        for path, digest in mapping.items():
            name = Path(path).name
            require(name not in result or result[name] == digest, "ambiguous native basename identity")
            result[name] = digest
        return result
    current, prior = basenames(current), basenames(prior)
    require(current == prior and current, "native implementation byte closure differs")
    return current


def prior_reference(audit, role, next_index):
    entries = [entry for entry in audit["trajectory"]["references"][role]
               if entry["point"]["logical_step"] == 3 and entry["point"]["phase"] == "single"
               and entry["point"]["index"] == next_index]
    require(len(entries) == 1, "original reference boundary is missing or ambiguous")
    return entries[0]


def declared_boundary(entry):
    return {"cut": entry["cut"], "point": entry["point"], "receipt_id": entry["receipt_id"],
            "arrays": {slot: rec["descriptor"] for slot, rec in entry["arrays"].items()}}


def prior_boundary_receipts(audit, role):
    """The first recorded exit belongs to the native prefix, then the suffix."""
    return set(audit["paired_source_checks"]["prefixes"][role]["receipts"]) | set(audit["native_paths"][role]["receipts"])


def verify_same_history(e, low, high):
    a, b = e.fields(low), e.fields(high)
    require(set(a) == set(b) and all(a[k] == b[k] for k in a if k not in {"image", "text"})
            and all(a[k] != b[k] for k in ("image", "text"))
            and e.point(low) == e.point(high), "paired states differ outside image/text or declared difference absent")
    return {"same_all_other_slots": True, "image_differs": True, "text_differs": True}


def transaction_verifications_match(actual, saved):
    """Compare transaction facts while preserving verifier authority scopes.

    Worker self-verification has local-hash-consistency authority. Supplying the
    archived external seal upgrades offline identity_scope without changing the
    transaction. No seal, event, inventory or verification fact is excluded.
    """
    fields = {"schema", "verified", "seal_sha256", "event_head", "events", "phase", "frontiers", "forecasts"}
    return (set(actual) == set(saved) == fields | {"identity_scope"}
            and actual["identity_scope"] == "externally-pinned"
            and saved["identity_scope"] in {"local-hash-consistency", "externally-pinned"}
            and all(actual[key] == saved[key] for key in fields))


def audit_imported_source(e, parent):
    h, payload, source, plan = e.h, parent["root"] / "payload", parent["source"], parent["plan"]
    selection = h["read"](h["file_path"](payload, plan["source_selection"]))
    require(selection["schema"] == "mdb-flux-single-contraction-selection-v1"
            and selection["prior_job_id"] == PRIOR_JOB and selection["archive_sha256"] == PRIOR_ARCHIVE
            and selection["request_sha256"] == PRIOR_REQUEST and selection["context_id"] == "p1-s53"
            and plan["context_id"] == source["context_id"] == "p1-s53"
            and selection["input_cuts"] == plan["input_cuts"] == source["imported_cuts"]
            and selection == source["source_selection"], "selected predecessor occurrence differs")
    loaded = {}
    for key in ("source_record", "audit_record", "prior_request", "prior_report", "prior_manifest"):
        rec = selection[key]
        path = h["file_path"](payload, rec["path"])
        require(sha_file(path) == rec["sha256"], "selected prior metadata hash differs: " + key)
        loaded[key] = h["read"](path)
    prior, audit, request, report, manifest = (loaded[key] for key in
        ("source_record", "audit_record", "prior_request", "prior_report", "prior_manifest"))
    require(selection["audit_record"]["sha256"] == PRIOR_AUDIT
            and selection["prior_report"]["sha256"] == PRIOR_REPORT
            and selection["prior_manifest"]["sha256"] == PRIOR_MANIFEST
            and report["files_manifest_sha256"] == PRIOR_MANIFEST
            and audit["mechanics_status"] == "valid" and audit["instrument_status"] == "native-replay-qualified"
            and audit["source"]["job_id"] == report["job_id"] == prior["job_id"] == PRIOR_JOB
            and audit["source"]["request_sha256"] == report["request_sha256"] == prior["request_sha256"] == PRIOR_REQUEST
            and audit["source"]["archive_sha256"] == report["archive_sha256"] == PRIOR_ARCHIVE
            and audit["source"]["source_record_sha256"] == selection["source_record"]["sha256"],
            "pinned original audit/transport/manifest/source differs")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(h["sha"](json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == request["request_sha256"] == PRIOR_REQUEST,
            "original request self seal differs")
    expected_files = set(selection["files"]) | {selection["source_record"]["path"]}
    expected_files |= {rec["path"] for rec in selection["prior_rows"].values()}
    require(set(selection["source_files"]) == expected_files, "selected original source inventory differs")
    inventory = verify_source_inventory(payload, selection, manifest, h)
    imported = e.typed["audit_store"](payload / "input-store")
    require(set(imported["cuts"]) == set(selection["input_cuts"].values()), "unexpected imported complete cut")
    for role in ("zero", "text"):
        rec = selection["prior_rows"][role]
        path = h["file_path"](payload, rec["path"])
        row = h["read"](path)
        require(sha_file(path) == rec["sha256"] == manifest["rows/" + role + ".json"]
                and row["id"] == role and row["final_cut"] == audit["native_paths"][role]["final"]
                and row["image"]["sha256"] == selection["expected_pngs"][role] == plan["expected_pngs"][role]
                and row["final_latent"]["descriptor"] == selection["expected_final_latents"][role]
                    == plan["expected_final_latents"][role], "prior reference row/endpoint differs")
        for entry in row["reference_trajectory"]:
            require(entry["receipt_id"] in prior_boundary_receipts(audit, role), "prior boundary receipt absent from audited path")
        for index, field in ((6, "expected_frontiers"), (7, "expected_outputs")):
            original = prior_reference(audit, role, index)
            row_entry = next((entry for entry in row["reference_trajectory"] if entry["cut"] == original["cut"]), None)
            require(row_entry is not None and declared_boundary(row_entry) == declared_boundary(original)
                    == selection[field][role], "selected prior boundary/descriptor/cut binding differs")
        original = prior_reference(audit, role, 3)
        cut = selection["input_cuts"][role]
        require(original["retained_full_cut"] and cut == original["cut"]
                and imported["cuts"][cut] == e.store["cuts"][cut]
                and e.point(cut) == original["point"], "imported cut is not the authenticated after-single2 state")
    native_binding = native_implementation_closure(source["implementation_sources"], prior["implementation_sources"])
    require(source["native_source_binding"] == native_binding
            and source["versions"] == prior["versions"] and source["execution"] == prior["execution"]
            and source["model_identity"] == prior["model_identity"]
            and source["checkpoint_sha256"] == prior["checkpoint_sha256"] == plan["expected_checkpoint_sha256"],
            "native source/checkpoint/execution/version closure differs")
    verify_same_history(e, selection["input_cuts"]["zero"], selection["input_cuts"]["text"])
    return selection, {"kind": "authenticated-prior-after-single2-replay", "prior_job_id": PRIOR_JOB,
        "prior_archive_sha256": PRIOR_ARCHIVE, "prior_request_sha256": PRIOR_REQUEST,
        "prior_audit_sha256": PRIOR_AUDIT, "prior_report_sha256": PRIOR_REPORT,
        "prior_producer_manifest_sha256": PRIOR_MANIFEST, "input_inventory": inventory,
        "native_source_binding": native_binding, "selection": selection,
        "scope": "Original prior report/manifest/audit authenticate selected CAS and observed replay controls. One current job, no fabricated child custody. Both I/T differ: feature transplantation selects a whole observed history, not a text-bank-only cause."}


def nested_slot_array(e, cut, slot, path):
    """Read an authenticated tensor inside a retained public slot page."""
    root = e.data / "typed-store"
    record = e.read(root / "cuts" / (cut + ".json"))
    page = e.read(root / "pages" / (record["slots"][slot] + ".json"))
    encoded, described = page["encoded"], page["value"]
    for key in path:
        require(encoded["kind"] == described["kind"] == "mapping", "nested slot is not a mapping")
        encoded = encoded["items"][key]
        described = next(item["value"] for item in described["items"] if item["key"]["value"] == key)
    require(encoded["kind"] == described["kind"] == "tensor", "nested slot is not a tensor")
    with (root / "blobs" / (page["tensor_blob"] + ".safetensors")).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        tensor = header[encoded["name"]]
        start, stop = tensor["data_offsets"]
        stream.seek(8 + size + start)
        raw = stream.read(stop - start)
    require(tensor["dtype"] == "BF16", "nested native modulation dtype differs")
    values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(tensor["shape"])
    require(e.typed["matches"](values, described, bf16=True), "nested slot tensor descriptor differs")
    return values


def audit_candidates(e, candidates, source, prefixes):
    require(set(candidates) == {"zero", "text", "A", "M", "AM"}, "five candidate inventory differs")
    ports, checks, controls = {}, {}, {}
    for name, candidate in candidates.items():
        role = "text" if name == "text" else "zero"
        native = name in {"zero", "text"}
        parent = source["input_cuts"][role]
        require(candidate["input_parent"] == parent and candidate["selection"] == ("native" if native else name)
                and candidate["source_parent"] == (None if native else source["input_cuts"]["text"]),
                "candidate source/selection differs")
        values = {key: e.array(ref) for key, ref in candidate["ports"].items()}
        require(set(values) == {"residual", "gate", "original_concat", "used_concat", "projection", "updated", "image", "text"},
                "native candidate port inventory differs")
        n, width = candidate["text_rows"], candidate["attention_width"]
        require(n == 512 and width == 3072 and values["residual"].shape == values["updated"].shape == (1, 1536, width)
                and values["original_concat"].shape == values["used_concat"].shape == (1, 1536, 4 * width)
                and values["projection"].shape == values["residual"].shape
                and values["gate"].shape == (1, 1, width)
                and exact_bits(values["residual"][:, :n], e.slot(parent, "text"))
                and exact_bits(values["residual"][:, n:], e.slot(parent, "image"))
                and exact_bits(values["updated"][:, :n], values["text"])
                and exact_bits(values["updated"][:, n:], values["image"]), "candidate native input/output ABI differs")
        modulation = nested_slot_array(e, parent, "context", ("single_mod",))
        if modulation.ndim == 2:
            modulation = modulation[:, None]
        require(modulation.shape == (1, 1, 3 * width) and exact_bits(values["gate"], modulation[..., 2 * width:]),
                "candidate gate differs from retained modulation")
        checks[name] = PANEL["parallel_residual"](values["residual"], values["gate"], values["projection"],
                                                   values["updated"], e.typed["quantize_bf16"])
        require(candidate["equation_exact"] is True, "reported whole residual equation differs")
        ports[name] = values
    for name, candidate in candidates.items():
        values = ports[name]
        selection = candidate["selection"]
        checked = feature_selection(values["original_concat"], values["used_concat"], ports["text"]["original_concat"],
            text_rows=candidate["text_rows"], attention_width=candidate["attention_width"], selection=selection)
        expected = {"text_concat_exact_recipient": checked["recipient_text_exact"],
                    "A_image_exact_selected": checked["image_attention_exact_selected"],
                    "M_image_exact_selected": checked["image_mlp_exact_selected"]}
        require(candidate["partition_controls"] == expected, "recorded feature partition controls differ")
        checks[name]["feature_selection"] = checked
        if name in {"zero", "text"}:
            for slot in ("image", "text"):
                controls[name + "_candidate_" + slot + "_exact_native"] = exact_bits(values[slot], e.slot(prefixes[name]["post_single6"], slot))
        else:
            require(exact_bits(values["original_concat"], ports["zero"]["original_concat"])
                    and exact_bits(values["residual"], ports["zero"]["residual"])
                    and exact_bits(values["gate"], ports["zero"]["gate"]), "edited candidate changed recipient operands")
            controls[name + "_text_exact_zero_candidate"] = exact_bits(values["text"], ports["zero"]["text"])
    n = candidates["zero"]["text_rows"]
    controls["AM_projection_image_exact_text_candidate"] = exact_bits(ports["AM"]["projection"][:, n:], ports["text"]["projection"][:, n:])
    contrasts = {name: {"image_from_low": distance(ports[name]["image"], ports["zero"]["image"]),
                        "image_from_high": distance(ports[name]["image"], ports["text"]["image"]),
                        "text_from_low": distance(ports[name]["text"], ports["zero"]["text"])}
                 for name in ("A", "M", "AM")}
    contrasts["H_M_over_A"] = {"signed_A_minus_M_L2_to_high": contrasts["A"]["image_from_high"]["base_l2"] - contrasts["M"]["image_from_high"]["base_l2"],
        "observed": "matched" if contrasts["M"]["image_from_high"]["base_l2"] < contrasts["A"]["image_from_high"]["base_l2"] else "miss",
        "scope": "Predeclared tentative local distance ordering; a miss retains every arm, not a mechanics failure."}
    return ports, checks, controls, contrasts


def check_boundary(e, cut, expected):
    require(cut == expected["cut"] and e.point(cut) == expected["point"], "replayed complete frontier identity differs")
    for slot, desc in expected["arrays"].items():
        require(e.typed["matches"](e.slot(cut, slot), desc, bf16=desc["dtype"] == "torch.bfloat16"),
                "replayed live array differs from pinned prior descriptor")


def audit_prefixes(e, source, selection, replay, prefixes):
    require(set(replay) == set(prefixes) == {"zero", "text"}, "source replay/prefix role inventory differs")
    retained = {cut: e.fields(cut) for cut in e.store["cuts"]}
    conditioning = e.fields(source["input_cuts"]["zero"])["conditioning"]
    replay_paths, prefix_paths, controls = {}, {}, {}
    for role in ("zero", "text"):
        start, pre6 = source["imported_cuts"][role], source["input_cuts"][role]
        path = e.walk(start, pre6, limit=5)
        record = replay[role]
        require(path["native_steps"] == 3 and path["acts"] == 0 and record["input_cut"] == start
                and record["pre6_cut"] == pre6 and record["receipts"] == [e.receipts[rid] for rid in path["receipts"]]
                and path["receipts"][-1] == selection["expected_frontiers"][role]["receipt_id"]
                and record["complete_cut_exact_prior"] is True and record["live_descriptors_exact_prior"] is True,
                "after2-to-pre6 three-step replay lineage differs")
        check_boundary(e, pre6, selection["expected_frontiers"][role])
        require(e.point(pre6)["index"] == e.point(pre6)["local_clock"] == 6, "source site is not pre-single6")
        for slot, ref in source["input_arrays"][role].items():
            require(slot in {"image", "text", "conditioning", "latent"}
                    and exact_bits(e.array(ref), e.slot(pre6, slot))
                    and ref["native_stride"] == e.fields(pre6)[slot]["stride"], "retained pre6 source array/cut differs")
        require(set(source["input_arrays"][role]) == {"image", "text", "conditioning", "latent"}, "pre6 port inventory differs")
        path["conditioning_coverage"] = BASE["conditioning_coverage"](path, e.receipts, retained, conditioning, e.point)
        replay_paths[role] = path
        prefix = prefixes[role]
        post = prefix["post_single6"]
        path = e.walk(pre6, post, limit=3)
        require(prefix["input_cut"] == pre6 and path["native_steps"] == 1 and path["acts"] == 0
                and prefix["receipt"] == e.receipts[path["receipts"][0]], "native single6 prefix differs")
        expected = selection["expected_outputs"][role]
        controls[role + "_prefix_cut_equals_prior"] = post == expected["cut"]
        # Replay mismatch is an instrument observation; retained source provenance
        # is checked separately. Never substitute a known prior output as a repair.
        controls[role + "_prefix_descriptors_equal_prior"] = all(
            e.typed["matches"](e.slot(post, slot), desc, bf16=desc["dtype"] == "torch.bfloat16")
            for slot, desc in expected["arrays"].items())
        require(e.point(post) == expected["point"] and e.point(post)["index"] == 7, "native6 prefix clock differs")
        path["conditioning_coverage"] = BASE["conditioning_coverage"](path, e.receipts, retained, conditioning, e.point)
        prefix_paths[role] = path
    paired = verify_same_history(e, source["input_cuts"]["zero"], source["input_cuts"]["text"])
    return replay_paths, prefix_paths, controls, paired


def projection_timing(timeline, *, replay_steps=6):
    require(replay_steps in {0, 6}, "unsupported source replay timing contract")
    require([event["sequence"] for event in timeline] == list(range(len(timeline))), "timeline sequence differs")
    main = [event for event in timeline if event["kind"] == "hypotheses_sealed"]
    secondary = [event for event in timeline if event["kind"] == "projection_forecast_sealed"]
    starts = {event["candidate"]: event for event in timeline if event["kind"] == "candidate_start"}
    ends = {event["candidate"]: event for event in timeline if event["kind"] == "candidate_complete"}
    require(len(starts) == sum(event["kind"] == "candidate_start" for event in timeline)
            and len(ends) == sum(event["kind"] == "candidate_complete" for event in timeline), "repeated candidate timeline event")
    checks = {"one_main_seal": len(main) == 1, "one_numeric_seal": len(secondary) == 1,
              "five_candidate_events": set(starts) == set(ends) == {"zero", "text", "A", "M", "AM"}}
    if len(main) == len(secondary) == 1:
        primary, seal = main[0], secondary[0]
        checks.update(main_after_only_six_allowed_replay_steps=primary["counts"]["adapter_steps"] == replay_steps
            and primary["counts"]["receipt_emissions"] == replay_steps and primary["blocked_native_entries"] == 0
            and all(primary["counts"][key] == 0 for key in ("candidate_calls", "acts", "images", "vae_calls", "prompt_encodings", "noise_draws")),
            numeric_after_only_two_native_sources=seal["counts"]["candidate_calls"] == 2
            and seal["counts"]["adapter_steps"] == seal["counts"]["receipt_emissions"] == replay_steps
            and all(seal["counts"][key] == 0 for key in ("acts", "images", "vae_calls")),
            sources_after_main_before_numeric=all(name in starts and name in ends
                and primary["sequence"] < starts[name]["sequence"] < ends[name]["sequence"] < seal["sequence"] for name in ("zero", "text")),
            edited_after_numeric=all(name in starts and name in ends
                and seal["sequence"] < starts[name]["sequence"] < ends[name]["sequence"] for name in ("A", "M", "AM")))
    if replay_steps == 0:
        checks["main_before_any_panel_native_step"] = checks.pop("main_after_only_six_allowed_replay_steps")
    return {"checks": checks, "qualified": all(checks.values()),
            "status": "prospective-timing-qualified" if all(checks.values()) else "prospective-timing-unresolved",
            "scope": ("Source-bound worker event/counter policy. Six preseal own-history replay steps are allowed; no current-site direct calculation before main seal. Not hermetic isolation or independent kernel-call tracing." if replay_steps else
                "Prepared fresh states precede this panel. Main seal before all panel native calls; numeric seal after exactly two native sources and no panel prefix. Source-bound policy, not hermetic isolation or independent kernel-call tracing.")}


def audit_projection_transaction(e, source, candidates, ports, summary, timeline, *, replay_steps=6):
    h, read = e.h, e.read
    root = e.data / "projection-transaction"
    record = read(e.data / "projection-forecast.json")
    require(record["seal"] == summary["projection_forecast_seal"], "numeric seal/summary differs")
    verified = h["verify_transaction"](root, expected_seal_sha256=record["seal"]["seal_sha256"])
    require(transaction_verifications_match(verified, record["verification"]) and verified["phase"] == "closed", "projection transaction custody differs")
    tree = h["NumericalTree"](root)
    binding = tree.decode(read(root / "frontiers/source_binding.json")["value"])
    prediction = tree.decode(read(root / "forecasts/sampled_projection.json")["predictions"])
    observed = tree.decode(read(root / "observations/edited_candidates.json")["value"])
    require(binding == record["binding"] and set(binding) == {"source_candidates", "linear_path", "weight_rows", "bias_rows", "weight_parameter", "bias_parameter", "row_extraction",
                "query_indices", "output_indices", "text_rows", "attention_width", "counts_before_seal", "main_seal_sha256"}
            and binding["source_candidates"] == {role: {"input_parent": source["input_cuts"][role], "ports": candidates[role]["ports"]}
                for role in ("zero", "text")}
            and binding["linear_path"] == source["linear_path"] == "single_transformer_blocks.6.attn.to_out"
            and binding["main_seal_sha256"] == summary["seal"]["seal_sha256"]
            and binding["query_indices"] == source["query_indices"] == prediction["query_indices"] == [0, 31, 32, 255, 511, 512, 767, 1023]
            and binding["output_indices"] == source["output_dims"] == prediction["output_indices"] == list(range(0, 3072, 96))
            and binding["text_rows"] == 512 and binding["attention_width"] == 3072
            and set(prediction["cases"]) == set(observed["samples"]) == set(record["samples"]) == {"A", "M", "AM"},
            "conditional projection source/indices/target inventory differs")
    full_weight = source["source_parameters"]["attn.to_out.weight"]
    require(full_weight["shape"] == [3072, 12288] and full_weight["dtype"] == "torch.bfloat16"
            and binding["weight_parameter"] == full_weight
            and binding["bias_parameter"] == source["source_parameters"].get("attn.to_out.bias")
            and isinstance(binding["row_extraction"], str) and "index_select(0, output_indices)" in binding["row_extraction"],
            "native linear source geometry differs")
    weights = e.array(binding["weight_rows"])
    bias = None if binding["bias_rows"] is None else e.array(binding["bias_rows"])
    require(weights.shape == (32, 12288) and binding["weight_rows"]["descriptor"]["dtype"] == "torch.bfloat16"
            and ((bias is None and "attn.to_out.bias" not in source["source_parameters"])
                 or (bias is not None and bias.shape == (32,) and source["source_parameters"]["attn.to_out.bias"]["shape"] == [3072])),
            "selected native projection weight/bias rows differ")
    cases = {}
    positions = np.asarray(prediction["query_indices"], dtype=np.int64) + binding["text_rows"]
    outputs = prediction["output_indices"]
    low_r = ports["zero"]["residual"][:, positions][:, :, outputs].astype(np.float64)
    low_g = np.broadcast_to(ports["zero"]["gate"], ports["zero"]["residual"].shape)[:, positions][:, :, outputs].astype(np.float64)
    require(exact_bits(prediction["recipient_residual_samples"], low_r)
            and exact_bits(prediction["recipient_gate_samples"], low_g), "forecast recipient sample source differs")
    sample_error = PANEL["sample_error"]
    for name, predicted in prediction["cases"].items():
        rebuilt = sampled_projection(ports["zero"]["original_concat"], ports["text"]["original_concat"], weights, bias,
            ports["zero"]["residual"], ports["zero"]["gate"], text_rows=512, attention_width=3072,
            selection=name, query_indices=prediction["query_indices"], output_indices=outputs, quantize=e.typed["quantize_bf16"])
        require(set(predicted) == set(rebuilt), "projection forecast quantity inventory differs")
        for key, value in predicted.items():
            expected_dtype = np.float64 if key.endswith("fp64") else np.float32
            require(np.asarray(value).shape == (1, 8, 32) and np.asarray(value).dtype == expected_dtype
                    and np.isfinite(value).all(), "numeric forecast finite/dtype/shape differs")
        p16 = e.typed["quantize_bf16"](predicted["projection_fp64"].astype(np.float32))
        g = np.broadcast_to(ports["zero"]["gate"], ports["zero"]["residual"].shape)[:, positions][:, :, outputs]
        r = ports["zero"]["residual"][:, positions][:, :, outputs]
        product16 = e.typed["quantize_bf16"](g * p16)
        require(exact_bits(p16, predicted["projection_bf16"]) and exact_bits(product16, predicted["product_bf16"])
                and exact_bits(e.typed["quantize_bf16"](r + product16), predicted["updated_bf16"])
                and exact_bits(low_r + low_g * predicted["projection_fp64"], predicted["updated_fp64"]),
                "sealed BF16 projection/product/residual reference differs")
        actual = {key: ports[name][key][:, positions][:, :, outputs] for key in ("projection", "updated")}
        saved = observed["samples"][name]
        require(record["samples"][name]["sample_shape"] == [1, 8, 32]
                and record["samples"][name]["actual_native_dtype"] == "torch.bfloat16", "native numerical sample metadata differs")
        for key in actual:
            require(exact_bits(saved[key], actual[key]) and exact_bits(e.array(record["samples"][name][key]), actual[key]),
                    "numeric observation borrowed another target sample")
        errors = {key: sample_error(value, actual[key.split("_")[0]])
                  for key, value in predicted.items() if key != "product_bf16"}
        require(saved["errors_actual_minus_prediction"] == record["samples"][name]["errors_actual_minus_prediction"],
                "numeric reported error records differ")
        cases[name] = {"errors_actual_minus_prediction": errors,
            "projection_role": "represented high-image projection consistency control" if name == "AM" else "novel conditional mixed-feature projection",
            "updated_role": "novel conditional image update using low residual (including AM)",
            "reported_error_deltas": {key: {metric: errors[key][metric] - saved["errors_actual_minus_prediction"][key][metric]
                for metric in errors[key]} for key in errors},
            "source_recomputation_error": {key: sample_error(rebuilt[key], value) for key, value in predicted.items()},
            "cast_reference_exact_native": {key: exact_bits(predicted[key + "_bf16"], actual[key]) for key in actual},
            "actual_rms": {key: float(np.sqrt(np.mean(value.astype(np.float64) ** 2))) for key, value in actual.items()},
            "samples": record["samples"][name], "shape": [1, 8, 32]}
    expected_count = {**source["preseal_counts"], "candidate_calls": 5}
    require(observed["counts"] == expected_count, "projection edited-call census differs")
    timing = projection_timing(timeline, replay_steps=replay_steps)
    seals = [event for event in timeline if event["kind"] == "projection_forecast_sealed"]
    if len(seals) == 1:
        require(seals[0]["seal"] == record["seal"] and seals[0]["counts"] == binding["counts_before_seal"]
                and seals[0]["binding_sha256"] == h["sha"](h["canonical"](binding)), "numeric timeline binding differs")
    return {"instrument_status": "applicable", "prospective_claim": "qualified" if timing["qualified"] else "not-established",
            "timing": timing, "verification": verified, "saved_worker_verification": record["verification"],
            "seal": record["seal"], "source_binding": binding,
            "query_indices": prediction["query_indices"], "output_indices": outputs, "cases": cases,
            "arithmetic": prediction["arithmetic"], "scope": prediction["scope"],
            "signed_error_convention": "actual native output minus sealed prediction",
            "limits": ["Numerical errors and recomputation deltas are observations, not a collection or quality gate.",
                "Selected native weight rows are staged-source-authorized index_select outputs; the retained full-weight descriptor does not independently prove a partial-row Merkle inclusion.",
                "Trusted forecaster dataflow receives only source concat, recipient residual/gate and selected weights/bias. Source outputs remain resident and the transaction binding retains them; not hermetic isolation or an independent A/MLP-feature, attention, full-query-field or RGB prediction."]}


def current_parent(root, *, custody_verify=None):
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request = h["read"](root / "request.json")
    if custody_verify is None:
        custody_verify = runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    custody = custody_verify(root, limits=request["config"].get("transport_limits"))
    job, envelope = (h["read"](root / name) for name in ("job.json", "report.json"))
    require(custody["mechanics_status"] == "verified" and custody["job_id"] == job["job_id"] == envelope["job_id"]
            and custody["request_sha256"] == envelope["request_sha256"] == request["request_sha256"]
            and envelope["archive_root"] == "flux-single-contraction-result", "current custody/job/archive root differs")
    data = root / "extracted" / envelope["archive_root"]
    plan, summary, outer, source, frozen = (h["read"](data / name) for name in
        ("plan.json", "summary.json", "outer-summary.json", "source.json", "plan-frozen.json"))
    require(plan == request["config"] and outer == envelope["report"]
            and outer["panel_summary_sha256"] == sha_file(data / "summary.json")
            and outer["counts"] == COUNTS and summary["counts"] == {**COUNTS, "model_loads": 0}
            and source["job_id"] == summary["job_id"] == outer["job_id"] == job["job_id"]
            and source["request_sha256"] == summary["request_sha256"] == outer["request_sha256"] == request["request_sha256"]
            and source["source_hashes"] == request["source_hashes"]
            and source["model_identity"] == summary["model_identity"] == outer["model_identity"],
            "current outer/panel/source physical census differs")
    require(plan["single_index"] == 6 and plan["selector"] == {"logical_step": 3, "phase": "single", "index": 6}
            and frozen["request_sha256"] == request["request_sha256"] and frozen["plan_sha256"] == sha_file(data / "plan.json")
            and all(frozen[key] == 0 for key in ("model_loads", "adapter_steps", "candidate_calls")), "global preload plan/site freeze differs")
    verification = h["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    require(transaction_verifications_match(verification, summary["transaction_verification"])
            and verification["phase"] == "closed", "main contraction transaction differs")
    return {"root": root, "helpers": h, "request": request, "job": job, "custody": custody,
            "data": data, "plan": plan, "summary": summary, "outer": outer, "source": source,
            "frozen": frozen, "transaction": verification, "saved_worker_verification": summary["transaction_verification"]}


def audit_rows(e, source, plan, observed_rows, prefixes, ports, timeline):
    specs = {spec["id"]: spec for spec in plan["branches"]}
    require(tuple(specs) == ROW_IDS, "fixed nine branch plan differs")
    rows = {row["id"]: row for row in observed_rows}
    starts = {event["row"]: event for event in timeline if event["kind"] == "branch_start"}
    ends = {event["row"]: event for event in timeline if event["kind"] == "branch_complete"}
    # Some source-bound workers emit only branch_complete. Receipt emissions and
    # the public primary transaction remain authoritative for branch ancestry.
    require(len(rows) == len(observed_rows) == 9 and set(rows) == set(specs) == set(ends)
            and (not starts or set(starts) == set(specs)), "nine retained row/timing inventory differs")
    retained = {cut: e.fields(cut) for cut in e.store["cuts"]}
    conditioning = e.fields(source["input_cuts"]["zero"])["conditioning"]
    paths, pixels, latents = {}, {}, {}
    for name, row in rows.items():
        require(e.read(e.data / "rows" / (name + ".json")) == row
                and all(row[key] == value for key, value in specs[name].items()), "persisted row/spec differs")
        role = "text" if name == "text" else "zero"
        parent = prefixes[role]["post_single6"]
        require(row["parent"] == parent and row["prefix_source"] == role and row["start_point"] == e.point(parent)
                and row["continuation_policy"] == "recipient-stored-C-no-further-project", "row borrowed another native prefix")
        expected = {}
        if name not in {"zero", "text"}:
            for slot, spec_key in (("image", "image_source"), ("text", "text_source")):
                selected = specs[name][spec_key]
                if selected is not None:
                    expected[slot] = e.slot(prefixes[selected]["post_single6"], slot) if selected in {"zero", "text"} else ports[selected][slot]
        require([action["slot"] for action in row["acts"]] == list(expected), "typed Act slot inventory differs")
        current = parent
        for action in row["acts"]:
            current = e.act(action, current, expected[action["slot"]])
        path = e.walk(parent, row["final_cut"], row["acts"])
        require(path["native_steps"] == 15 and e.point(row["final_cut"])["logical_step"] == 4
                and e.point(row["final_cut"])["phase"] == "project"
                and row["continuation_receipts"] == [e.receipts[rid] for rid in path["receipts"] if e.receipts[rid]["operation"] == "continue"],
                "15-transition native suffix lineage differs")
        path["conditioning_coverage"] = BASE["conditioning_coverage"](path, e.receipts, retained, conditioning, e.point)
        require(e.typed["matches"](e.slot(source["input_cuts"]["zero"], "conditioning"), row["final_conditioning"], bf16=True),
                "final conditioning witness differs")
        latents[name] = e.array(row["final_latent"])
        require(exact_bits(latents[name], e.slot(row["final_cut"], "latent")), "final latent array/complete cut differs")
        pixels[name], paths[name] = e.png(row["image"]), path
    return rows, paths, pixels, latents


def emission_census(e, replay_paths, prefix_paths, paths, summary, plan):
    emissions = read_jsonl(e.data / "receipt-emissions.jsonl")
    # Source JSON object ordering is not a scientific or custody invariant. Each
    # replay is a complete contiguous three-step block, in either role order.
    replay_order = [emissions[index]["branch"] for index in (0, 3)] if len(emissions) >= 6 else []
    require(set(replay_order) == {"replay.zero", "replay.text"}, "source replay emission role blocks differ")
    expected = [(branch, rid) for branch in replay_order for rid in replay_paths[branch.removeprefix("replay.")]["receipts"]]
    expected += [("prefix." + role, rid) for role in ("zero", "text") for rid in prefix_paths[role]["receipts"]]
    expected += [(spec["id"], rid) for spec in plan["branches"] for rid in paths[spec["id"]]["receipts"]]
    require([item["sequence"] for item in emissions] == list(range(len(emissions)))
            and [(item["branch"], item["receipt"]) for item in emissions] == expected
            and len(emissions) == summary["counts"]["receipt_emissions"] == 153
            and sum(p["native_steps"] for p in [*replay_paths.values(), *prefix_paths.values(), *paths.values()]) == 143
            and sum(p["acts"] for p in paths.values()) == 10, "physical receipt emission census differs")
    return {"source_replay_steps": 6, "source_replay_role_order": replay_order,
            "native6_prefix_steps": 2, "suffix_steps": 135,
            "adapter_steps": 143, "acts": 10, "receipt_emissions": 153,
            "scope": "Physical emissions counted before deduplication; equal sham/coherent paths may share receipt identities."}


def native_geometry(ports, quantize):
    n = ports["zero"]["text"].shape[1]
    result = contraction_identity(ports["zero"]["residual"][:, n:], ports["text"]["residual"][:, n:],
                                  ports["zero"]["image"], ports["text"]["image"])
    result["native_terms"] = {}
    for role in ("zero", "text"):
        p = ports[role]
        write = quantize(p["gate"] * p["projection"])[:, n:].astype(np.float64)
        effective = p["image"].astype(np.float64) - p["residual"][:, n:].astype(np.float64)
        result["native_terms"][role] = {"rounded_gate_projection_product": vector_statistics(write),
            "effective_post_minus_pre": vector_statistics(effective),
            "residual_add_rounding_rho": vector_statistics(effective - write)}
    result["precision_scope"] = "Whole native fused projection and BF16 gate product verified separately from residual addition. Ueff includes the measured residual-add rounding rho; independently rounded A/M contributions are not asserted to sum to native Ueff."
    return result


def analyze(root, *, custody_verify=None):
    parent = current_parent(root, custody_verify=custody_verify)
    data, h = parent["data"], parent["helpers"]
    run = {"root": parent["root"], "data": data, "plan": parent["plan"], "request": parent["request"],
        "source": {**parent["custody"], "transaction": parent["transaction"]},
        "transaction": data / "transaction", "tree": h["NumericalTree"](data / "transaction")}
    e = COMMON["SavedEvidence"](data, run=run)
    return analyze_evidence(parent, e)


def analyze_evidence(parent, e, *, source_adapter=None):
    """Shared full-port panel with explicit source admission and physical census.

    The default remains the authenticated historical after2 replay. A fresh
    prepared-source adapter must supply its own lineage, prefixes and census;
    it does not create a substitute historical report or scheduler job.
    """
    data, run = e.data, e.run
    source, summary, plan = parent["source"], parent["summary"], parent["plan"]
    selection, lineage = (audit_imported_source(e, parent) if source_adapter is None else source_adapter.verify_source(e, parent))
    expected_preseal = ({**COUNTS, "model_loads": 0, "adapter_steps": 6, "candidate_calls": 0,
                "acts": 0, "receipt_emissions": 6, "images": 0, "vae_calls": 0} if source_adapter is None else source_adapter.preseal_counts)
    expected_outer = ({"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0} if source_adapter is None else source_adapter.outer_preseal_counts)
    observed = run["tree"].decode(e.read(data / "transaction/observations/native.json")["value"])
    hypotheses = run["tree"].decode(e.read(data / "transaction/forecasts/contraction_hypotheses.json")["predictions"])
    require(hypotheses == plan["hypotheses"] and source["branches"] == plan["branches"]
            and observed["counts"] == summary["counts"]
            and observed["source_sha256"] == sha_file(data / "source.json")
            and observed["control_observations"] == summary["control_observations"]
            and source["preseal_counts"] == expected_preseal and source["outer_preseal_counts"] == expected_outer,
            "primary source/plan/observation/preseal count differs")
    # Constant complete-state closure is independent of the replay outcome.
    anchor = e.fields(source["input_cuts"]["zero"])
    for cut, manifest in e.store["cuts"].items():
        fields = e.fields(cut)
        require(manifest["model_identity"] == source["model_identity"] and manifest["execution"] == source["execution"]
                and fields["conditioning"] == anchor["conditioning"]
                and all(fields[key] == value for key, value in anchor.items()
                    if key not in {"image", "text", "context", "latent", "noise_prediction", "step", "cursor", "phase"}),
                "typed model/conditioning/schedule closure differs")
    for receipt in e.receipts.values():
        require(receipt["model_identity"] == source["model_identity"] and receipt["execution"] == source["execution"],
                "receipt native identity differs")
    points = BASE["receipt_points"](e.receipts.values())
    require(all(cut not in e.store["cuts"] or e.point(cut) == point for cut, point in points.items()), "cut/receipt point disagreement")
    prefixes = e.read(data / "prefixes.json")["prefixes"]
    if source_adapter is None:
        replay = e.read(data / "source-replay.json")
        replay_paths, prefix_paths, controls, paired = audit_prefixes(e, source, selection, replay, prefixes)
    else:
        replay_paths, prefix_paths, controls, paired = source_adapter.audit_prefixes(e, source, prefixes)
    candidates = e.read(data / "candidates.json")["candidates"]
    ports, candidate_checks, candidate_controls, contrasts = audit_candidates(e, candidates, source, prefixes)
    controls.update(candidate_controls)
    timeline = read_jsonl(data / "timeline.jsonl")
    numerical = audit_projection_transaction(e, source, candidates, ports, summary, timeline,
        replay_steps=6 if source_adapter is None else source_adapter.native_replay_steps)
    main_events = [event for event in timeline if event["kind"] == "hypotheses_sealed"]
    require(len(main_events) == 1 and main_events[0]["seal"] == summary["seal"]
            and main_events[0]["counts"] == source["preseal_counts"], "primary timeline/seal differs")
    rows, paths, pixels, latents = audit_rows(e, source, plan, observed["rows"], prefixes, ports, timeline)
    census = (emission_census(e, replay_paths, prefix_paths, paths, summary, plan) if source_adapter is None else
        source_adapter.emission_census(e, prefix_paths, paths, summary, plan))
    comparisons = {name: {base: {"latent": distance(latents[name], latents[base]), "rgb": distance(pixels[name], pixels[base])}
                         for base in ("zero", "text")} for name in ROW_IDS}
    for name, base in (("sham", "zero"), ("coherent", "text")):
        controls[name + "_png_equals_" + base] = rows[name]["image"]["sha256"] == rows[base]["image"]["sha256"]
        controls[name + "_latent_equals_" + base] = comparisons[name][base]["latent"]["exact_base"]
    if source_adapter is None:
        for role in ("zero", "text"):
            controls[role + "_png_equals_prior"] = rows[role]["image"]["sha256"] == selection["expected_pngs"][role]
            controls[role + "_latent_equals_prior"] = rows[role]["final_latent"]["descriptor"] == selection["expected_final_latents"][role]
    extra_controls = {"zero_prefix_descriptors_equal_prior", "text_prefix_descriptors_equal_prior", "sham_latent_equals_zero", "coherent_latent_equals_text"}
    declared = {key: value for key, value in controls.items() if key not in extra_controls}
    require(declared == summary["control_observations"], "worker named controls differ from saved evidence")
    replay_qualified = all(controls.values())
    data_relative = data.relative_to(parent["root"]).as_posix()
    exported_rows = []
    for name in ROW_IDS:
        row = rows[name]
        def relative_ref(ref):
            return {**ref, "path": data_relative + "/" + ref["path"]}
        exported_rows.append({"id": name, "image": relative_ref(row["image"]), "final_latent": relative_ref(row["final_latent"]),
            "row_record": {"path": data_relative + "/rows/" + name + ".json", "sha256": sha_file(data / "rows" / (name + ".json"))},
            "parent": row["parent"], "final_cut": row["final_cut"], "acts": row["acts"],
            "native_path": paths[name], "comparisons": comparisons[name]})
    contrasts["AM_text_vs_coherent"] = {"post_image": distance(ports["AM"]["image"], ports["text"]["image"]),
        "installed_text_exact_high": True, "scope": "AM.text and coherent install the same high native post-T and use represented high image feature projection. Remaining post-I difference includes low-versus-high entry residual and native addition rounding; endpoint metric differences need not be additive."}
    geometry = native_geometry(ports, e.typed["quantize_bf16"])
    n = candidates["zero"]["text_rows"]
    geometry["counterfactual_vector_account"] = local_vector_account(ports["zero"]["residual"][:, n:],
        ports["text"]["residual"][:, n:], ports["zero"]["image"], ports["text"]["image"],
        ports["A"]["image"], ports["M"]["image"], ports["AM"]["image"])
    interactions = {"Z4_latent": factorial_interaction(*(latents[name] for name in ("zero", "A.image", "M.image", "AM.image"))),
                    "final_RGB": factorial_interaction(*(pixels[name] for name in ("zero", "A.image", "M.image", "AM.image")))}
    differentials = {label: {"from": first, "to": second,
        "latent": distance(latents[second], latents[first]), "rgb": distance(pixels[second], pixels[first])}
        for label, first, second in (("native_text_return", "zero", "text.text"),
            ("mixed_image_under_high_text", "text.text", "AM.text"),
            ("remaining_image_residual", "AM.text", "coherent"), ("AM_high_text_return", "AM.image", "AM.text"))}
    return {"schema": "mdb-flux-single-contraction-audit-v1", "source": {**parent["custody"],
                "source_record_sha256": sha_file(data / "source.json"), "staged_source_hashes": parent["request"]["source_hashes"],
                "global_plan_freeze": parent["frozen"], "transaction": parent["transaction"],
                "saved_worker_transaction_verification": parent["saved_worker_verification"], "custody_verifications": 1},
        "mechanics_status": "valid", "instrument_status": "native-replay-qualified" if replay_qualified else "replay-unresolved",
        "execution_status": {"job_id": parent["job"]["job_id"], "job_state": parent["job"]["state"], "model_loads": 1},
        "trend_status": "One observed whole-history feature factorial; all signed local/endpoint countertrends and forecast errors retained.",
        "terminal_status": "unassessed", "counts": parent["outer"]["counts"], "panel_counts": summary["counts"],
        "physical_census": census, "typed_store_counts": e.store["counts"], "imported_source": lineage,
        "paired_source_checks": {**paired, "source_replay": replay_paths, "native_prefixes": prefix_paths},
        "control_observations": controls, "replay_qualification": {"status": "qualified" if replay_qualified else "unresolved", "controls": controls},
        "candidate_checks": candidate_checks, "local_feature_contrasts": contrasts, "contraction_geometry": geometry,
        "numerical_projection_forecast": numerical, "sealed_hypotheses": hypotheses,
        "architectural_seal": summary["seal"], "numeric_seal": summary["projection_forecast_seal"],
        "comparisons": comparisons, "native_paths": paths, "rows": exported_rows,
        "image_feature_interactions": interactions, "continuation_differentials": differentials,
        "endpoint_trends": endpoint_trends(comparisons), "preseal_counts": source["preseal_counts"],
        "nested_audit_tensor_reference_base": data_relative,
        "reference_scopes": {"rows": "rows image/final_latent/row_record paths relative to run directory",
            "nested_arrays": "arrays/... in source, candidate and numerical metadata relative to nested_audit_tensor_reference_base",
            "staged_inputs": "selected source paths are relative to current run/payload; prior_path refers to original prior producer root"},
        "models_executed": False, "auditor_source_sha256": sha_file(Path(__file__)),
        "audit_helpers": {name: sha_file(HERE / name) for name in ("analyze_flux_single_contexts.py", "analyze_flux_single_kv.py",
            "analyze_flux_rescue_routes.py", "summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py", "flux_family_custody.py")},
        "limits": ["One previously observed p1-s53 history, nine dependent futures. Prior native reference suffixes/PNG endpoints were executed controls, not newly forecast RGB or fresh replication.",
            "Both working image and text differ at pre6. High A/M features encode a whole prior history, not an isolated text-bank cause.",
            "Only image A/M slices at one native fused projection are changed. Text features, recipient residual and gate are preserved; all edited post-T matches low unless a separate declared postblock text Act installs high.",
            "No exact additive decomposition into independently rounded A and M writes is claimed. Ueff is actual rounded post-pre; residual-add rho is retained.",
            "Sampled full-concat projection/write forecasts use observed source features and native weight rows. No independent attention/MLP-feature, readout, full trajectory or RGB prediction.",
            "Public receipt footprint coverage is source-bound metadata, not independent low-level kernel-read tracing.",
            "No semantic repair, object identity, automatic law induction or universal pathway claim follows from distance or arithmetic closure."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "counts")}))
