"""Offline, parent-job-bound audit of three fresh single0 K/V panels.

Custody is streamed once before any panel arrays are opened. Contexts are
audited sequentially, with real parent job/context identities and no invented
child reports. Numerical and endpoint disagreements remain observations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PANEL = runpy.run_path(str(HERE / "analyze_flux_single_kv.py"))
require, exact_bits, distance = (PANEL[key] for key in ("require", "exact_bits", "distance"))
CONTEXT_IDS = ("p0-s53", "p1-s47", "p1-s53")
ROW_IDS = ("zero", "text", "sham", "K.image", "V.image", "KV.image", "KV.both", "text.text", "coherent")
PANEL_COUNTS = {"adapter_steps": 191, "acts": 10, "candidate_calls": 5, "images": 9,
                "vae_calls": 9, "receipt_emissions": 201, "prompt_encodings": 0,
                "noise_draws": 0, "model_loads": 0, "physical_batch": 1}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path, *, max_line_bytes=4 << 20):
    """Small event records only; never load a producer's transport log text."""
    result = []
    with Path(path).open("rb") as stream:
        while line := stream.readline(max_line_bytes + 1):
            require(len(line) <= max_line_bytes, "event line exceeds declared bound")
            if line.strip():
                result.append(json.loads(line))
    return result


def recompute_attention(zero, text, *, text_rows, query_indices):
    """Independent source-QKV-only FP64 reference; one head at a time."""
    require(set(zero) == set(text) == {"q", "k", "v"}, "QKV source inventory differs")
    shape = np.asarray(zero["q"]).shape
    require(len(shape) == 4 and shape[0] == 1 and type(text_rows) is int
            and 0 < text_rows < shape[1] and query_indices
            and len(set(query_indices)) == len(query_indices)
            and all(type(index) is int and 0 <= index < shape[1] - text_rows for index in query_indices)
            and all(value.shape == shape and np.isfinite(value).all()
                    for source in (zero, text) for value in source.values()),
            "source QKV/query geometry differs")
    result = {}
    for selection in ("K", "V", "KV"):
        output = np.empty((1, len(query_indices), shape[2], shape[3]), dtype=np.float64)
        mass = np.empty((1, shape[2], len(query_indices)), dtype=np.float64)
        for head in range(shape[2]):
            query = np.array(zero["q"][0, np.array(query_indices) + text_rows, head], dtype=np.float64)
            key = np.array(zero["k"][0, :, head], dtype=np.float64)
            value = np.array(zero["v"][0, :, head], dtype=np.float64)
            if selection in {"K", "KV"}:
                key[:text_rows] = text["k"][0, :text_rows, head]
            if selection in {"V", "KV"}:
                value[:text_rows] = text["v"][0, :text_rows, head]
            logits = np.matmul(query, key.T) / math.sqrt(shape[-1])
            weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            output[0, :, head] = np.matmul(weights, value)
            mass[0, head] = weights[:, :text_rows].sum(axis=-1)
        result[selection] = {"attention": output, "text_mass": mass}
    return result


def forecast_math(prediction, source_qkv, *, text_rows, quantize):
    if prediction["instrument_status"] == "not-applicable":
        return {"instrument_status": "not-applicable", "cases": {},
                "scope": "Native rows retained; no numerical formula claim for this dispatch profile."}
    require(prediction["instrument_status"] == "applicable", "unknown forecast instrument status")
    rebuilt = recompute_attention(source_qkv["zero"], source_qkv["text"], text_rows=text_rows,
                                  query_indices=prediction["query_indices"])
    require(set(prediction["cases"]) == set(rebuilt), "forecast case inventory differs")
    cases = {}
    for name, expected in rebuilt.items():
        saved = prediction["cases"][name]
        cast = quantize(expected["attention"].astype(np.float32))
        cases[name] = {
            "saved_fp64_minus_offline_reference": PANEL["sample_error"](expected["attention"], saved["attention"]),
            "saved_mass_minus_offline_reference": PANEL["sample_error"](expected["text_mass"], saved["text_mass"]),
            "saved_bf16_reference_bit_exact_offline": exact_bits(cast, saved["attention_bf16"]),
            "offline_text_mass": {"min": float(expected["text_mass"].min()),
                                  "mean": float(expected["text_mass"].mean()),
                                  "max": float(expected["text_mass"].max())}}
    return {"instrument_status": "applicable", "cases": cases,
            "arithmetic": "FP64 recipient image Q times selected K / sqrt(head width), stable row softmax, selected V; text-bank sum; one head at a time",
            "allowed_inputs": "Only the two observed zero/text source QKV arrays and fixed image-query indices",
            "numerical_policy": "Cross-platform FP64/BLAS differences and BF16 rounding disagreements are retained, with no arbitrary error threshold or semantic gate.",
            "scope": "Recomputation of the sealed sampled-attention formula, not a native module/kernel or full RGB forecast."}


def text_direction(comparisons):
    """Prospective H-text-direction; signed positive margin supports prediction."""
    result = {}
    for observable, space, key in (("latent_l2", "latent", "base_l2"), ("rgb_mae", "rgb", "base_mae")):
        own = float(comparisons["text.text"]["text"][space][key])
        require(math.isfinite(own) and own >= 0, "nonfinite/negative endpoint distance")
        result[observable] = {}
        for name in ("K.image", "V.image", "KV.image"):
            other = float(comparisons[name]["text"][space][key])
            require(math.isfinite(other) and other >= 0, "nonfinite/negative endpoint distance")
            margin = other - own
            result[observable][name] = {"text_text_distance_to_text": own,
                "image_edit_distance_to_text": other, "signed_margin": margin,
                "direction": "supports" if margin > 0 else "miss" if margin < 0 else "tie"}
    result["disagreements"] = [name for name in ("K.image", "V.image", "KV.image")
        if result["latent_l2"][name]["direction"] != result["rgb_mae"][name]["direction"]]
    result["formula"] = "distance(image edit, text reference) - distance(text.text, text reference); positive supports H-text-direction; strict closer means ties are not support"
    result["terminal_status"] = "unassessed"
    result["scope"] = "Paired endpoint-distance trend, not semantic alignment, mediation fraction or independent futures. Every sign and observable disagreement is retained."
    return result


def cross_context_summary(contexts):
    """Portable rows retain per-context signed results instead of pooled winners."""
    rows = []
    for context_id, audit in contexts.items():
        if audit.get("mechanics_status") != "valid":
            rows.append({"context_id": context_id, "mechanics_status": audit.get("mechanics_status"),
                         "error": audit.get("error"), "trend_status": "evidence-incomplete"})
            continue
        direction = text_direction(audit["comparisons"])
        rows.append({"context_id": context_id, "mechanics_status": "valid",
            "instrument_status": audit["instrument_status"], "text_direction": direction,
            "image_KV_interactions": audit["image_KV_interactions"],
            "numerical_attention_forecast": audit["numerical_attention_forecast"],
            "control_observations": audit["control_observations"],
            "trend_status": "observed; all signed margins and disagreements retained"})
    return {"contexts": rows, "contrasts": {
        "within_prompt_seed": ["p1-s47", "p1-s53"], "within_seed_prompt": ["p0-s53", "p1-s53"]},
        "historical_context_role": "Earlier p0 seed47 evidence remains a separate code/history cohort; not a missing current 2x2 cell.",
        "effective_sample_scope": "Three own-context source histories, two prompts, two seeds with one shared seed; nine dependent futures per context, not 27 independent specimens.",
        "semantic_status": "unassessed; numerical distance and attention forecasts do not classify scene content",
        "terminal_status": "unassessed"}


class SavedEvidence(PANEL["BASE"]["Evidence"]):
    """One verified store view; does not collect or claim a separate job."""
    def __init__(self, data, *, run=None):
        self.h = runpy.run_path(str(HERE / "summarize_context_window.py"))
        self.typed = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))
        self.routes = runpy.run_path(str(HERE / "analyze_flux_concept_routes.py"))
        self.data, self.run = Path(data), run
        self.store = self.typed["audit_store"](self.data / "typed-store")
        self.receipts, self.edges, self._arrays_verified = {}, {}, set()
        for path in sorted((self.data / "typed-store/receipts").glob("*.json")):
            receipt = self.read(path)
            self.h["public_receipt"](receipt)
            require(receipt["fingerprint"] == path.stem, "receipt filename differs")
            self.receipts[path.stem] = receipt
            if receipt["operation"] == "continue":
                require(receipt["parent"] not in self.edges or self.edges[receipt["parent"]] == receipt,
                        "ambiguous native continuation edge")
                self.edges[receipt["parent"]] = receipt

    def array(self, record):
        path = self.h["file_path"](self.data, record["path"])
        identity = (path, record["sha256"])
        if identity not in self._arrays_verified:
            require(sha_file(path) == record["sha256"], "saved array checksum differs")
            self._arrays_verified.add(identity)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        require(value.dtype == np.float32 and record["stored_dtype"] == "float32"
                and list(value.shape) == record["stored_shape"] and value.size and np.isfinite(value).all()
                and self.typed["matches"](value, record["descriptor"],
                    bf16=record["descriptor"]["dtype"] == "torch.bfloat16"),
                "array dtype/shape/native descriptor differs")
        return value


def builder_act(e, action, parent, expected, *, slot):
    require(action["slot"] == slot and action["undeclared_slots_exact"] is True,
            "builder Act slot/closure declaration differs")
    receipt = action["receipt"]
    require(receipt == e.receipts[action["receipt_id"]] and receipt["fingerprint"] == action["receipt_id"]
            and receipt["operation"] == "apply" and receipt["act"] == action["act"]
            and receipt["parent"] == action["parent"] == parent
            and receipt["result"] == action["post_act"]
            and action["act"]["writes"] == [slot]
            and action["act"]["operation"]["id"] == "saturn.builtin.replace",
            "builder Act receipt/source differs")
    parameter = e.typed["unpack"](action["act"]["parameters"])["value"]
    after = action["post_act"]
    require(e.fields(after) == {**e.fields(parent), slot: parameter}
            and e.typed["unpack"](receipt["writes"])[slot] == parameter
            and e.point(parent) == e.point(after) == action["point"]
            and exact_bits(e.slot(after, slot), expected)
            and e.fields(after)[slot]["stride"] == e.fields(parent)[slot]["stride"],
            "builder Act payload/stride/undeclared state differs")
    return after


def builder_path(e, declared, start, end, *, expected_steps):
    require(isinstance(declared, list) and len(declared) == expected_steps,
            "builder path physical step count differs")
    current, ids = start, []
    for receipt in declared:
        require(receipt == e.receipts[receipt["fingerprint"]] and receipt["parent"] == current
                and receipt["operation"] == "continue" and receipt["steps"] == 1
                and len(receipt["transitions"]) == 1, "builder native path is discontinuous")
        transition = receipt["transitions"][0]
        require(transition["before"] == e.point(current)
                and transition["after"] == e.point(receipt["result"])
                and transition["native_operation"]["footprint"] == "declared"
                and transition["native_operation"]["operation"] == transition["before"]["next_operation"],
                "builder native receipt point/footprint differs")
        current = receipt["result"]
        ids.append(receipt["fingerprint"])
    require(current == end, "builder native path ends at a different cut")
    return ids


def audit_builder(parent, context, reference, panel):
    h, data = parent["helpers"], parent["data"]
    path = h["file_path"](data, reference["path"])
    require(sha_file(path) == reference["sha256"] and path == data / "builders" / context["id"] / "builder.json",
            "builder record path/hash differs")
    record = h["read"](path)
    require(record["schema"] == "mdb-flux-single-context-builder-v1" and record["context"] == context
            and record["job_id"] == parent["job"]["job_id"], "builder context/parent-job identity differs")
    e = SavedEvidence(path.parent)
    own = e.array(record["arrays"]["own_conditioning"])
    zero = np.zeros_like(own)
    encoded = parent["source"]["encodings"][context["prompt_id"]]
    require(encoded["prompt"] == context["prompt"], "builder used another prompt encoding")
    encoded_path = h["file_path"](data, encoded["conditioning"]["path"])
    require(sha_file(encoded_path) == encoded["conditioning"]["sha256"], "native encoding file differs")
    require(exact_bits(own, np.load(encoded_path, mmap_mode="r", allow_pickle=False)),
            "builder own-C does not equal its native prompt encoding")
    ids_path = h["file_path"](data, encoded["txt_ids"]["path"])
    require(sha_file(ids_path) == encoded["txt_ids"]["sha256"]
            and np.array_equal(np.load(ids_path, mmap_mode="r", allow_pickle=False),
                               e.slot(record["initial"], "txt_ids").astype(np.float32)),
            "builder text-position IDs differ from its own encoding")
    initial, z2, z3 = (record[key] for key in ("initial", "Z2", "Z3"))
    points = {"initial": (0, "project", None), "Z2": (2, "project", None), "zeroZ2": (2, "project", None),
              "Z3": (3, "project", None), "zero_single0": (3, "single", 0),
              "own_single0": (3, "single", 0), "matched_text": (3, "single", 0)}
    for name, expected in points.items():
        point = e.point(record[name])
        require(tuple(point[key] for key in ("logical_step", "phase", "index")) == expected,
                "builder cut is at the wrong native frontier")
    for name, cut, slot in (("initial_latent", initial, "latent"), ("Z2_latent", z2, "latent"),
                            ("Z3_latent", z3, "latent"), ("own_conditioning", z2, "conditioning")):
        require(exact_bits(e.array(record["arrays"][name]), e.slot(cut, slot))
                and record["arrays"][name]["native_stride"] == e.fields(cut)[slot]["stride"],
                "builder source array/typed cut/stride differs")
    require(exact_bits(e.slot(initial, "conditioning"), own), "initial own-C differs")
    paths, acts = record["paths"], record["acts"]
    receipts = builder_path(e, paths["to_Z2"], initial, z2, expected_steps=56)
    zero_z2 = builder_act(e, acts["zero_conditioning"], z2, zero, slot="conditioning")
    require(zero_z2 == record["zeroZ2"], "zero-C Act result differs")
    receipts.append(acts["zero_conditioning"]["receipt_id"])
    receipts += builder_path(e, paths["to_Z3"], zero_z2, z3, expected_steps=28)
    restored = builder_act(e, acts["restore_conditioning"], z3, own, slot="conditioning")
    receipts.append(acts["restore_conditioning"]["receipt_id"])
    receipts += builder_path(e, paths["zero_to_single0"], z3, record["zero_single0"], expected_steps=6)
    receipts += builder_path(e, paths["own_to_single0"], restored, record["own_single0"], expected_steps=6)
    matched = builder_act(e, acts["transplant_text"], record["zero_single0"],
                          e.slot(record["own_single0"], "text"), slot="text")
    receipts.append(acts["transplant_text"]["receipt_id"])
    require(matched == record["matched_text"], "text-only matched cut differs")
    for name in ("Z3", "zero_single0", "matched_text"):
        require(exact_bits(e.slot(record[name], "conditioning"), zero), "zero-history stored C is not zero")
    require(exact_bits(e.slot(record["own_single0"], "conditioning"), own), "own-history stored C differs")
    for name, cut in (("zero", record["zero_single0"]), ("text", matched)):
        require(panel["input_cuts"][name] == cut
                and panel["evidence"].store["cuts"][cut] == e.store["cuts"][cut],
                "panel cut is not the actual freshly built typed cut")
    for manifest in e.store["cuts"].values():
        require(manifest["model_identity"] == parent["source"]["model_identity"]
                and manifest["execution"] == parent["source"]["execution"], "builder cut model/execution differs")
    for receipt in e.receipts.values():
        require(receipt["model_identity"] == parent["source"]["model_identity"]
                and receipt["execution"] == parent["source"]["execution"], "builder receipt identity differs")
    emissions = h["read"](path.parent / "receipt-emissions.json")
    require([event["sequence"] for event in emissions] == list(range(99))
            and [event["receipt"] for event in emissions] == receipts
            and set(receipts) == set(e.receipts)
            and record["counts"] == {"adapter_steps": 96, "acts": 3, "receipt_emissions": 99},
            "builder actual receipt census differs")
    return {"kind": "fresh-native-builder", "context_id": context["id"],
            "parent_job_id": record["job_id"], "record": reference, "counts": record["counts"],
            "path_native_steps": {name: len(value) for name, value in paths.items()},
            "typed_store_counts": e.store["counts"], "input_cuts": panel["input_cuts"],
            "own_prompt_encoding_exact": True, "text_only_transplant_exact": True,
            "scope": "Public source-bound native path/Act receipts plus retained typed cut/array closure; no imported history or invented child job. Initial seeded noise provenance follows the frozen worker source and declared seed, not an offline random-generator rerun."}


def cumulative_census(contexts, initial, final, panel_sources):
    expected = dict(initial)
    require(expected == {"model_loads": 1, "prompt_encodings": 2, "noise_draws": 0,
        "adapter_steps": 0, "acts": 0, "candidate_calls": 0, "images": 0,
        "vae_calls": 0, "receipt_emissions": 0, "physical_batch": 1},
        "resident model/two-encoding initial census differs")
    for context_id in CONTEXT_IDS:
        row, source = contexts[context_id], panel_sources[context_id]
        expected["noise_draws"] += 1
        require(source["parent_counts_before_builder"] == expected, "context noise/builder starting offset differs")
        require(row["builder_counts"] == {"adapter_steps": 96, "acts": 3, "receipt_emissions": 99},
                "source-builder counts differ")
        for key, value in row["builder_counts"].items():
            expected[key] += value
        require(source["parent_counts_after_builder"] == expected, "builder ending offset differs")
        require(row["panel_counts"] == PANEL_COUNTS, "physical per-panel census differs")
        for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "receipt_emissions"):
            expected[key] += row["panel_counts"][key]
        require(row["counts_after"] == expected, "context cumulative physical work differs")
    require(final == expected and final["adapter_steps"] == 861 and final["acts"] == 39
            and final["receipt_emissions"] == 900 and final["candidate_calls"] == 15
            and final["images"] == final["vae_calls"] == 27
            and final["model_loads"] == 1 and final["prompt_encodings"] == 2 and final["noise_draws"] == 3,
            "parent actual physical census differs")
    return {"global": final, "per_context": {"builder_native_steps": 96, "panel_native_steps": 191,
        "total_native_steps": 287, "builder_acts": 3, "panel_acts": 10, "total_acts": 13,
        "direct_single_calls": 5, "native_final_images": 9, "receipt_emissions": 300},
        "scope": "Receipt/ledger-verified adapter transitions, typed Acts, direct single-block captures and native VAE images; model load and encoding/noise belong to the real parent job, never virtual per-panel jobs."}


def parent_run(root, *, custody_verify=None):
    """The sole outer custody check, streamed before opening numerical arrays."""
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request = h["read"](root / "request.json")
    if custody_verify is None:
        custody_verify = runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    verification = custody_verify(root, limits=request["config"].get("transport_limits"))
    job, envelope = (h["read"](root / name) for name in ("job.json", "report.json"))
    require(verification["mechanics_status"] == "verified"
            and verification["job_id"] == job["job_id"] == envelope["job_id"]
            and verification["request_sha256"] == request["request_sha256"] == envelope["request_sha256"],
            "real parent custody/job/request binding differs")
    require(envelope["archive_root"] == "flux-single-contexts-result", "unexpected campaign archive root")
    data = root / "extracted" / envelope["archive_root"]
    plan, source, summary, frozen = (h["read"](data / name)
        for name in ("plan.json", "source.json", "summary.json", "plan-frozen.json"))
    require(plan == request["config"] and summary == envelope["report"]
            and summary["source_sha256"] == sha_file(data / "source.json")
            and source["source_hashes"] == request["source_hashes"]
            and source["job_id"] == summary["job_id"] == job["job_id"]
            and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
            and source["model_identity"] == summary["model_identity"]
            and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"]
            and h["sha"](h["canonical"](source["checkpoint_files"])) == source["checkpoint_sha256"],
            "parent plan/source/checkpoint/report binding differs")
    contexts = plan["contexts"]
    require([context["id"] for context in contexts] == list(CONTEXT_IDS)
            and source["contexts"] == contexts and set(summary["contexts"]) == set(CONTEXT_IDS)
            and len({context["prompt_id"] for context in contexts}) == 2
            and [(context["prompt_id"], context["seed"]) for context in contexts]
                == [("p0", 53), ("p1", 47), ("p1", 53)], "fresh connected context design differs")
    require(frozen["plan_sha256"] == sha_file(data / "plan.json")
            and frozen["request_sha256"] == request["request_sha256"] and frozen["job_id"] == job["job_id"]
            and frozen["source_identity"] == h["sha"](h["canonical"](request["source_hashes"]))
            and set(frozen["work"]) == {"model_loads", "prompt_encodings", "noise_draws", "adapter_steps", "candidate_calls"}
            and all(value == 0 for value in frozen["work"].values()), "global pre-work plan freeze differs")
    required_sources = {"flux_single_contexts_campaign.py", "flux_single_contexts.py", "flux_single_kv.py",
                        "flux_family_custody.py", "plan.json", "preflight.v1.json"}
    require(required_sources <= request["source_hashes"].keys(), "missing frozen instrument sources")
    payload_plan = h["read"](h["file_path"](root / "payload", "plan.json"))
    require(payload_plan == plan, "staged plan differs from parent observed plan")
    timeline = read_jsonl(data / "timeline.jsonl")
    require([event["sequence"] for event in timeline] == list(range(len(timeline))), "parent event sequence differs")
    starts = [event["context_id"] for event in timeline if event["kind"] == "context_start"]
    completes = [event["context_id"] for event in timeline if event["kind"] == "context_complete"]
    require(starts == completes == list(CONTEXT_IDS), "parent context scheduling inventory differs")
    require([event["prompt_id"] for event in timeline if event["kind"] == "prompt_encoded"] == ["p0", "p1"],
            "native prompt encoding event inventory differs")
    return {"root": root, "data": data, "helpers": h, "request": request, "plan": plan,
            "source": source, "summary": summary, "frozen": frozen, "job": job,
            "custody": verification, "timeline": timeline}


class FreshSource:
    panel_model_loads = 0
    limits = ["One freshly built own-context source history; nine dependent futures within this panel.",
              "Architectural seal follows observed builder work. Numerical seal follows exactly two allowed native source captures; this is conditional QKV attention forecasting, not raw-prompt or RGB prediction."]

    def __init__(self, parent, context):
        self.parent, self.context, self.lineage = parent, context, None

    def plan_matches(self, e, plan, request):
        base = request["config"]
        expected = {**base, "context": self.context, "max_adapter_steps": 191, "max_acts": 10,
                    "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9}
        variable = plan.get("max_wall_seconds")
        require(type(variable) is int and 1 <= variable <= base["max_wall_seconds"],
                "derived panel wall bound differs")
        expected["max_wall_seconds"] = variable
        return plan == expected

    def verify(self, e, source, summary):
        parent, context = self.parent, self.context
        record = parent["summary"]["contexts"][context["id"]]
        summary_path = parent["helpers"]["file_path"](parent["data"], record["panel_summary"]["path"])
        require(summary_path == e.data / "summary.json"
                and sha_file(summary_path) == record["panel_summary"]["sha256"]
                and record["context"] == context
                and source["context"] == context and source["context_id"] == summary["context_id"] == context["id"]
                and source["job_id"] == summary["job_id"] == parent["job"]["job_id"]
                and source["source_selection"] == {"kind": "fresh-native-builder", "builder": record["builder"]}
                and source["builder"] == record["builder"], "fresh panel/builder/real parent identity differs")
        for key in ("checkpoint_sha256", "execution", "implementation_sources", "versions"):
            require(source[key] == parent["source"][key], "panel native source identity differs: " + key)
        require(summary["counts"] == record["panel_counts"] == PANEL_COUNTS,
                "panel summary physical counts differ")
        self.lineage = audit_builder(parent, context, record["builder"],
                                     {"input_cuts": source["input_cuts"], "evidence": e})
        return self.lineage


def analyze(root):
    parent = parent_run(root)
    h, data = parent["helpers"], parent["data"]
    sources = {context_id: h["read"](data / "contexts" / context_id / "source.json") for context_id in CONTEXT_IDS}
    census = cumulative_census(parent["summary"]["contexts"], parent["source"]["counts_after_encoding"],
                               parent["summary"]["counts"], sources)
    audits = {}
    for context in parent["plan"]["contexts"]:
        context_id = context["id"]
        e = None
        try:
            panel_data = data / "contexts" / context_id
            summary = h["read"](panel_data / "summary.json")
            verification = h["verify_transaction"](panel_data / "transaction",
                expected_seal_sha256=summary["seal"]["seal_sha256"])
            require(verification == summary["transaction_verification"] and verification["phase"] == "closed",
                    "main context transaction seal/closure differs")
            source_authority = {**parent["custody"], "parent_job_id": parent["job"]["job_id"],
                "context_id": context_id, "source_record_sha256": sha_file(panel_data / "source.json"),
                "transaction": verification, "scope": "Context within one real parent archive/job; no child scheduler report or archive."}
            run = {"root": parent["root"], "data": panel_data, "plan": h["read"](panel_data / "plan.json"),
                "request": parent["request"], "source": source_authority, "transaction": panel_data / "transaction",
                "tree": h["NumericalTree"](panel_data / "transaction"),
                "execution_status": {"parent_job_state": parent["job"]["state"], "parent_job_id": parent["job"]["job_id"],
                    "context_id": context_id, "child_job": False}}
            e, adapter = SavedEvidence(panel_data, run=run), FreshSource(parent, context)
            audit = PANEL["analyze_evidence"](e, source_adapter=adapter)
            prediction = h["NumericalTree"](panel_data / "attention-transaction").decode(
                h["read"](panel_data / "attention-transaction/forecasts/sampled_attention.json")["predictions"])
            candidates = h["read"](panel_data / "candidates.json")["candidates"]
            source_qkv = {name: {key: e.array(ref) for key, ref in candidates[name]["observed_qkv"].items()}
                          for name in ("zero", "text")}
            audit.update(context_id=context_id, context=context, fresh_builder=adapter.lineage,
                offline_attention_math=forecast_math(prediction, source_qkv,
                    text_rows=candidates["zero"]["text_rows"], quantize=e.typed["quantize_bf16"]),
                text_direction=text_direction(audit["comparisons"]))
            audits[context_id] = audit
            del source_qkv, prediction, candidates, run
        except Exception as error:
            audits[context_id] = {"context_id": context_id, "parent_job_id": parent["job"]["job_id"],
                "mechanics_status": "evidence-incomplete", "instrument_status": "unresolved",
                "trend_status": "not assessed for this context; other completed contexts retained",
                "terminal_status": "unassessed", "error": f"{type(error).__name__}: {error}"}
        finally:
            del e
            gc.collect()
    complete = [key for key, value in audits.items() if value["mechanics_status"] == "valid"]
    return {"schema": "mdb-flux-single-contexts-audit-v1", "source": {
        **parent["custody"], "parent_job_id": parent["job"]["job_id"],
        "source_record_sha256": sha_file(data / "source.json"), "plan_record_sha256": sha_file(data / "plan.json"),
        "global_plan_freeze": parent["frozen"], "staged_source_hashes": parent["request"]["source_hashes"],
        "custody_verifications": 1, "scope": "One streamed multipart parent custody verification before panel arrays; context views are not separate jobs."},
        "execution_status": {"job_state": parent["job"]["state"], "job_id": parent["job"]["job_id"], "model_loads": 1},
        "mechanics_status": "valid" if len(complete) == 3 else "evidence-incomplete",
        "instrument_status": "per-context controls and forecast applicability retained",
        "trend_status": "exploratory paired contrasts; all signed effects/countertrends retained",
        "terminal_status": "unassessed", "counts": parent["summary"]["counts"], "physical_census": census,
        "contexts": audits, "completed_contexts": complete, "cross_context_summary": cross_context_summary(audits),
        "models_executed": False, "auditor_source_sha256": sha_file(Path(__file__)),
        "audit_helpers": {name: sha_file(HERE / name) for name in ("analyze_flux_single_kv.py", "analyze_flux_rescue_routes.py",
            "summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py", "flux_family_custody.py")},
        "limits": ["No model/kernel replay, new pixels, scheduler submissions or historical-source import by this auditor.",
            "Architecture and dataflow are source-bound trusted worker claims, not hermetic process isolation or independent low-level read traces.",
            "Native continuation arrays and RGB are observations; sampled attention forecasts do not independently predict MLP or final images.",
            "Numerical/endpoint metrics do not certify colors, shapes, object count or prompt semantics.",
            "Three contexts do not support universal generalization or independent replication of each of 27 dependent futures."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("mechanics_status", "counts", "completed_contexts")}))
