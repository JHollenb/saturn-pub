"""Join saved native attention into a copied canonical MDB Project; zero forwards."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from saturn_pub import StateAddress

from mdb.project import Project
from mdb.symbol_attention import attach_attention, definition_at, retained_binding
from mdb.symbol_lifecycle import SymbolUse


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def tensor_blob(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.stem:
        raise ValueError("native tensor blob digest mismatch")
    length = int.from_bytes(raw[:8], "little")
    return raw, json.loads(raw[8:8 + length]), length + 8


def native_array(project, cut_id, slot, layer=None):
    store = project.store.root
    cut = read(store / "cuts" / (cut_id + ".json"))
    page_path = store / "pages" / (cut["slots"][slot] + ".json")
    if sha(page_path) != page_path.stem:
        raise ValueError("native page digest mismatch")
    page = read(page_path)
    encoded = page["encoded"] if layer is None else page["encoded"]["items"][layer]
    raw, header, offset = tensor_blob(store / "blobs" / (page["tensor_blob"] + ".safetensors"))
    item = header[encoded["name"]]
    dtype = {"BF16": "<u2", "I64": "<i8"}[item["dtype"]]
    lo, hi = item["data_offsets"]
    return np.frombuffer(raw, dtype=dtype, count=(hi - lo) // np.dtype(dtype).itemsize,
        offset=offset + lo).reshape(item["shape"])


def captured_bits(root, descriptor):
    path = root / descriptor["path"]
    if path.is_absolute() and not path.is_relative_to(root):
        raise ValueError("attention artifact escapes declared root")
    if sha(path) != descriptor["sha256"]:
        raise ValueError("attention capture digest mismatch")
    raw = np.load(path, allow_pickle=False)
    if descriptor["dtype"] != "torch.bfloat16":
        raise ValueError("this saved link requires the pinned BF16 capture contract")
    return np.frombuffer(raw.tobytes(), dtype="<u2").reshape(descriptor["shape"])


def source_locations(layer, row):
    return (StateAddress("tokens", (0, row)), *tuple(StateAddress(slot, (layer, 0, kv, row))
        for slot in ("keys", "values") for kv in range(2)))


def native_use(lifecycle, sid, receipt, *, observations=(), metadata=None):
    return SymbolUse("native-use-" + receipt["fingerprint"] + "-" + lifecycle.binding_fingerprint,
        lifecycle.binding_fingerprint, sid, "native-qwen2-suffix",
        source_cut=lifecycle.definitions[0].source_cut, parent_cut=receipt["parent"],
        result_cut=receipt["result"], branch="main", receipt=receipt["fingerprint"],
        execution=lifecycle.binding.execution, locations=lifecycle.binding.locations,
        evidence_refs=("cas://sha256/" + receipt["fingerprint"],), status="observed",
        signed_observations=tuple(observations), metadata=metadata or {})


def link(results, attention_analysis, output):
    output.mkdir(exist_ok=False)
    inventory = read(results / "output-manifest.json")["files"]
    for row in inventory:
        if (results / row["path"]).stat().st_size != row["bytes"] or sha(results / row["path"]) != row["sha256"]:
            raise ValueError("original native inventory mismatch")
    # Native bytes are copied intact. Only the derivative index receives new symbols.
    shutil.copytree(results / "project", output / "project")
    project = Project(output / "project")
    data = read(results / "public-result.json")
    observations = read(attention_analysis / "head-observations.json")
    observed = {(row["arm"], row["step"], row["layer"], row["head"]): row
        for row in observations if row["lane"] == "public"}
    if len(observed) != 5040 or sum(row["lane"] == "public" for row in observations) != 5040:
        raise ValueError("attention observation coverage is duplicated or incomplete")
    source = data["source_row"]
    receipts = {sid: {r.fingerprint: r.to_dict() for r in project.receipts(sid)}
        for sid in ("public-recipient", "public-donor")}
    definitions, writer_inputs, export_rows, output_rows = {}, {}, [], []
    for side in ("recipient", "donor"):
        sid = "public-" + side
        writer = next(r for r in receipts[sid].values() if r["operation"] == "continue" and r.get("steps") == 33
            and project.cut_descriptor(sid, r["parent"])["execution_point"]["logical_step"] == source)
        binding = retained_binding(project, sid, writer["parent"], "temperature.description.input",
            (StateAddress("tokens", (0, source)),), context="source-formation/" + side,
            claim="Supplied description token at its native prompt-computation boundary.")
        lifecycle = definition_at(project, sid, writer["parent"], binding,
            metadata={"role": "description-input", "side": side, "source_row": source})
        lifecycle = replace(lifecycle, uses=(native_use(lifecycle, sid, writer),))
        project.save_symbol_lifecycle(sid, lifecycle, include_slices=False)
        writer_inputs[side] = lifecycle
        for layer in range(28):
            binding = retained_binding(project, sid, writer["result"], f"temperature.description.formed.L{layer}",
                source_locations(layer, source), context="source-formation/" + side,
                context_slots=("tokens", "keys", "values"),
                claim="Native source K/V rows formed during supplied description-token computation; semantic name supplied.")
            lifecycle = definition_at(project, sid, writer["result"], binding, receipt=writer["fingerprint"],
                definition_refs=(writer_inputs[side].definitions[0].occurrence_id,),
                metadata={"role": "formed-description", "side": side, "layer": layer,
                    "writer_receipt": writer["fingerprint"], "source_row": source})
            project.save_symbol_lifecycle(sid, lifecycle, include_slices=False)
            definitions[(side, layer)] = lifecycle
    origin_checks, reader_checks, logit_checks, allocation_checks = 0, 0, 0, 0
    for arm in data["results"]:
        name = arm["arm"].removeprefix("public-")
        if name == "unobserved-native":
            continue
        side = "donor" if name == "donor-native" else "recipient"
        sid = "public-" + side
        for step, row in enumerate(arm["trajectory"]):
            readout = receipts[sid][row["receipt"]]
            if readout["parent"] != row["parent_cut"]:
                raise ValueError("attention trace differs from retained native parent")
            commit = next(r for r in receipts[sid].values() if r["operation"] == "continue"
                and r["parent"] == readout["result"] and r["result"] == row["result_cut"])
            if not np.array_equal(native_array(project, readout["result"], "logits"), captured_bits(results, row["logits"])):
                raise ValueError("saved readout differs from canonical native logits")
            logits = (captured_bits(results, row["logits"]).astype("<u4") << 16).view("<f4")
            if int(logits[0].argmax()) != row["next_token_id"]:
                raise ValueError("linked token differs from full-vocabulary native argmax")
            tokens = native_array(project, commit["result"], "tokens").tolist()[0]
            if tokens != row["input_ids"] + [row["next_token_id"]]:
                raise ValueError("canonical token commit differs from native output")
            logit_checks += 1
            context = f"temperature/{name}/step{step}"
            formed, pending = [], []
            for layer in range(28):
                value_side = ("donor" if name == "donor-all" or name == "donor-native"
                    or (name == "donor-late" and layer >= 19) else "recipient")
                key_origin, value_origin = definitions[(side, layer)], definitions[(value_side, layer)]
                trace = arm["traces"][step]["layers"][layer]
                probabilities = (captured_bits(results, trace["attention"]).astype("<u4") << 16).view("<f4")
                for head in range(12):
                    observed_head = observed[(name, step, layer, head)]
                    actual = probabilities[0, head, 0]
                    if (observed_head["source_mass"] != float(actual[source])
                            or not np.array_equal(np.asarray(observed_head["allocation"]), actual)):
                        raise ValueError("symbol attention summary differs from the native attention array")
                    allocation_checks += 1
                for slot, origin in (("keys", key_origin), ("values", value_origin)):
                    original = native_array(project, origin.definitions[0].source_cut, slot, layer)[:, :, source]
                    current = native_array(project, row["parent_cut"], slot, layer)[:, :, source]
                    captured = captured_bits(results, trace["key" if slot == "keys" else "value"])[:, :, source]
                    if not np.array_equal(original, current):
                        raise ValueError("native K/V origin differs from declared writer lineage")
                    if not np.array_equal(current, captured):
                        raise ValueError("attention reader did not consume the bound native K/V rows")
                    origin_checks += 1
                    reader_checks += 1
                binding = retained_binding(project, sid, row["parent_cut"], f"temperature.description.L{layer}",
                    source_locations(layer, source), context=context, context_slots=("tokens", "keys", "values"),
                    claim="Observed description K/V source at this receiver; query-head allocation attached as native use evidence.")
                lifecycle = definition_at(project, sid, row["parent_cut"], binding,
                    definition_refs=tuple(dict.fromkeys((key_origin.definitions[0].occurrence_id,
                        value_origin.definitions[0].occurrence_id))),
                    metadata={"role": "description-source", "arm": name, "step": step, "layer": layer,
                        "source_row": source, "key_origin_binding": key_origin.binding_fingerprint,
                        "value_origin_binding": value_origin.binding_fingerprint,
                        "key_origin_side": side, "value_origin_side": value_side,
                        "lineage_support": "exact-native-KV-row-bytes-and-observed-reader-bytes"})
                formed.append(lifecycle)
                pending.append(trace)
            output_binding = retained_binding(project, sid, readout["result"], "temperature.next-token.logits",
                (StateAddress("logits"),), context=context, context_slots=("tokens", "logits"),
                evidence=("cas://sha256/" + row["logits"]["sha256"],), claim="Actual native logits consumed by greedy sample and token commit.")
            output_life = definition_at(project, sid, readout["result"], output_binding,
                definition_refs=tuple(life.definitions[0].occurrence_id for life in formed),
                metadata={"role": "next-token-readout", "arm": name, "step": step,
                    "emitted_token_id": row["next_token_id"], "emitted_token_text": row["next_token_text"],
                    "native_commit_receipt": commit["fingerprint"]})
            output_life = replace(output_life, uses=(native_use(output_life, sid, commit,
                observations=({"observable": "emitted_token_id", "value": row["next_token_id"], "n": 1},)),))
            project.save_symbol_lifecycle(sid, output_life, include_slices=False)
            out = {"arm": name, "step": step, "binding": output_binding.to_dict(),
                "lifecycle_id": output_life.fingerprint, "source_cut": readout["result"],
                "readout_receipt": readout["fingerprint"], "commit_receipt": commit["fingerprint"],
                "committed_cut": commit["result"], "token_id": row["next_token_id"], "token_text": row["next_token_text"]}
            output_rows.append(out)
            for layer, lifecycle in enumerate(formed):
                heads = [observed[(name, step, layer, head)] for head in range(12)]
                joined = attach_attention(project, sid, lifecycle, receipt_id=readout["fingerprint"],
                    trace=pending[layer], heads=heads, source_row=source, output_binding=output_binding,
                    metadata={"arm": name, "step": step, "layer": layer})
                saved = project.save_symbol_lifecycle(sid, joined, include_slices=False)
                if any(u["receipt_closure"] != "native-parent-result-matched" for u in saved["joins"]["uses"]):
                    raise ValueError("attention symbol use lacks native receipt closure")
                export_rows.append({"session_id": sid, "arm": name, "step": step, "layer": layer,
                    "binding": joined.binding.to_dict(), "lifecycle_id": joined.fingerprint,
                    "definition": joined.definitions[0].to_dict(), "uses": [u.to_dict() for u in joined.uses],
                    "heads": heads, "output": out})
        print(json.dumps({"arm": name, "linked_layer_symbols": len(export_rows)}), flush=True)
    # Read-only reopen must resolve canonical records and both navigation directions.
    reopened = Project(output / "project", read_only=True)
    reopened.store.load = lambda *a, **kw: (_ for _ in ()).throw(ValueError("metadata lookup hydrated state"))
    selected = export_rows[20]
    found = reopened.symbols_at(selected["session_id"], "values", cut_id=selected["definition"]["source_cut"], selector=(20, 0, 0, source))
    if selected["binding"]["fingerprint"] not in {r["binding_fingerprint"] for r in found["matches"]}:
        raise ValueError("canonical reverse source address lost linked attention binding")
    if reopened.symbol_lifecycle(selected["session_id"], selected["lifecycle_id"])["lifecycle"]["uses"] != selected["uses"]:
        raise ValueError("reopened canonical attention uses differ")
    all_ids = {d.definitions[0].occurrence_id for d in definitions.values()} | {
        x.definitions[0].occurrence_id for x in writer_inputs.values()} | {r["definition"]["occurrence_id"] for r in export_rows}
    for row in export_rows:
        if not set(row["definition"]["definition_refs"]) <= all_ids:
            raise ValueError("source writer reference is unresolved")
    for item in inventory:
        if sha(results / item["path"]) != item["sha256"]:
            raise ValueError("original raw evidence changed during linking")
    (output / "linked-symbols.json").write_text(json.dumps(export_rows) + "\n")
    (output / "outputs.json").write_text(json.dumps(output_rows, indent=2) + "\n")
    (output / "formation-symbols.json").write_text(json.dumps([life.to_dict() for life in
        (*writer_inputs.values(), *definitions.values())], indent=2) + "\n")
    summary = {"native_forwards": 0, "source_layer_bindings": len(export_rows), "output_bindings": len(output_rows),
        "formation_bindings": len(definitions), "source_input_bindings": len(writer_inputs),
        "canonical_bindings": sum(len(project.symbols(sid)) for sid in receipts),
        "attention_uses": sum(len(r["uses"]) for r in export_rows), "output_commit_uses": len(output_rows),
        "writer_input_uses": len(writer_inputs), "native_KV_origin_checks": origin_checks,
        "native_KV_reader_checks": reader_checks, "exact_native_logit_commit_checks": logit_checks,
        "exact_native_attention_checks": allocation_checks,
        "raw_files_reverified": len(inventory), "frameworks_imported": [n for n in ("torch", "transformers", "saturn") if n in sys.modules],
        "mechanics_status": "valid", "coverage": "All 5040 observed public head uses; Bench remains a separate comparison contract",
        "scope": "Canonical saved-native bindings and observed uses; supplied semantic names; head necessity untested"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("attention_analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    link(args.results.resolve(), args.attention_analysis.resolve(), args.output.resolve())
