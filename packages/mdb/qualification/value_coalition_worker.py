"""Same-parent late layer/KV value-coalition decomposition, one resident model."""
from __future__ import annotations

import base64
from dataclasses import replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import torch
from saturn_pub import StateAddress
from saturn_pub.core import Act

from attention_support import Panel as AttentionPanel
from mdb.attention_observer import AttentionObserver
from mdb.symbol_attention import attach_attention, definition_at, retained_binding
from mdb.symbol_lifecycle import SymbolUse
from mdb.semantic_scaffold import resolve_span
from mdb.value_coalition import LATE_CELLS, plan, replacement
from scaffold_io import digest_file, save


class Panel(AttentionPanel):
    def run_arm(self, session, ids, arm, source_row, origin, *,
                symbol_layers=tuple(range(19, 28)), capture_layers=None, operation_limit=6500,
                source_evidence=()):
        name = arm["arm"]
        trajectory, own, symbols = [], [], []
        for step in range(3):
            parent = session.checkpoint()["cut_id"]
            context = name + f"/step{step}"
            sources = []
            if step == 0:
                for layer in symbol_layers:
                    binding = retained_binding(self.project, session.session_id, parent,
                        f"temperature.description.L{layer}",
                        (StateAddress("tokens", (0, source_row)), *tuple(StateAddress(slot, (layer, 0, kv, source_row))
                            for slot in ("keys", "values") for kv in range(2))),
                        context=context, context_slots=("tokens", "keys", "values"), evidence=source_evidence,
                        claim="Observed source carrier in a declared layer/KV coalition branch; role supplied.")
                    sources.append(definition_at(self.project, session.session_id, parent, binding,
                        metadata={"arm": name, "layer": layer, "source_row": source_row,
                            "cell_origins": origin, "intervention_kind": arm["kind"]}))
            observer = AttentionObserver(self.model) if step == 0 else None
            if observer:
                with observer:
                    receipt = session.session.continue_(31)
            else:
                receipt = session.session.continue_(31)
            self.operations += 31
            readout = session.checkpoint()["cut_id"]
            logits = session.session.read("logits")
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite native output")
            output = retained_binding(self.project, session.session_id, readout, "temperature.next-token.logits",
                (StateAddress("logits"),), context=context, context_slots=("tokens", "logits"),
                claim="Actual native full vocabulary readout consumed by sample/commit.")
            lifecycle = definition_at(self.project, session.session_id, readout, output,
                definition_refs=tuple(s.definitions[0].occurrence_id for s in sources),
                metadata={"arm": name, "step": step})
            commit = session.session.continue_(2)
            self.operations += 2
            result = session.checkpoint()["cut_id"]
            token = int(logits[0].argmax())
            expected = ids + own
            if session.session.read("tokens").tolist()[0] != expected + [token]:
                raise ValueError("native own-token commit lost authority")
            own.append(token)
            use = SymbolUse("commit-" + commit.fingerprint + "-" + output.to_dict()["fingerprint"],
                output.to_dict()["fingerprint"], session.session_id, "native-qwen2-suffix",
                source_cut=readout, parent_cut=readout, result_cut=result, branch="main",
                execution=output.execution, receipt=commit.fingerprint, locations=output.locations,
                status="observed", metadata={"emitted_token_id": token})
            lifecycle = replace(lifecycle, uses=(use,))
            self.project.save_symbol_lifecycle(session.session_id, lifecycle, include_slices=False)
            traces = []
            if observer:
                if [r["layer"] for r in observer.rows] != list(range(28)):
                    raise ValueError("incomplete native observation")
                for captured in observer.rows:
                    layer = captured["layer"]
                    if capture_layers is not None and layer not in capture_layers:
                        continue
                    trace = {"layer": layer}
                    for key in ("query", "key", "value", "attention", "head_return"):
                        if not torch.isfinite(captured[key]).all():
                            raise ValueError("nonfinite attention capture")
                        trace[key] = self.storage.tensor(f"{name}-L{layer}-{key}", captured[key])
                    traces.append(trace)
                    if layer not in symbol_layers:
                        continue
                    probabilities = captured["attention"][0, :, 0].float()
                    heads = []
                    for head in range(12):
                        group = head // 6
                        allocation = probabilities[head]
                        contribution = allocation[source_row] * captured["value"][0, group, source_row].float()
                        heads.append({"head": head, "kv_head": group,
                            "source_mass": float(allocation[source_row]),
                            "entropy_nats": float(-(allocation * allocation.clamp_min(1e-30).log()).sum()),
                            "source_value_return_norm": float(contribution.norm())})
                    joined = attach_attention(self.project, session.session_id, sources[symbol_layers.index(layer)],
                        receipt_id=receipt.fingerprint, trace=trace, heads=heads, source_row=source_row,
                        output_binding=output, metadata={"arm": name, "layer": layer})
                    self.project.save_symbol_lifecycle(session.session_id, joined, include_slices=False)
                    symbols.append(joined.to_dict())
            trajectory.append({"step": step, "parent_cut": parent, "readout_cut": readout,
                "result_cut": result, "readout_receipt": receipt.fingerprint, "commit_receipt": commit.fingerprint,
                "input_ids": expected, "next_token_id": token, "next_token_text": self.tokenizer.decode([token]),
                "logits": self.storage.tensor(f"{name}-s{step}-logits", logits),
                "output_lifecycle": lifecycle.to_dict(), "traces": traces})
        if self.operations > operation_limit:
            raise RuntimeError("native operation budget exceeded")
        return {"arm": arm, "trajectory": trajectory, "own_ids": own,
            "own_text": self.tokenizer.decode(own), "source_symbols": symbols}

    def experiment(self, case):
        encoded = {side: self.tokenizer(case[side]["text"], add_special_tokens=False,
            return_offsets_mapping=True) for side in ("recipient", "donor")}
        ids = {side: encoded[side]["input_ids"] for side in encoded}
        positions = {side: resolve_span(encoded[side]["offset_mapping"], case[side]["source_span"]) for side in encoded}
        if positions["recipient"] != positions["donor"] or len(positions["recipient"]) != 1:
            raise ValueError("requires matching one-token source")
        source = positions["recipient"][0]
        if ids["recipient"][:source] != ids["donor"][:source] or ids["recipient"][source+1:] != ids["donor"][source+1:]:
            raise ValueError("contrast changed beyond declared source")
        wrong = source - 1  # The supplied colon source-control from the previous assay.
        sessions, parents, values = {}, {}, {}
        for side in ("recipient", "donor"):
            sessions[side], parents[side] = self.opened("public", side, ids[side], source)
            values[side] = sessions[side].session.read("values")
        outcomes = []
        for arm in plan():
            name, kind = arm["arm"], arm["kind"]
            side = "donor" if kind in {"donor-native", "reciprocal"} else "recipient"
            session = sessions[side]
            session.restore(parents[side])
            before = session.checkpoint()["cut_id"]
            act = None
            target = wrong if kind == "wrong-target" else source
            payload_row = wrong if kind == "wrong-payload" else source
            payload_side = "recipient" if kind in {"sham", "reciprocal"} else "donor"
            origin = [{"layer": l, "kv_head": kv, "side": side} for l, kv in LATE_CELLS]
            if kind not in {"native", "donor-native"}:
                current = session.session.read("values")
                installed = replacement(current, values[payload_side], arm["cells"], target, payload_row, arm["dose"])
                act = session.session.apply(Act.replace("values", installed)).to_dict()
                session.checkpoint()
                if target == source:
                    for cell in origin:
                        if (cell["layer"], cell["kv_head"]) in [tuple(c) for c in arm["cells"]]:
                            cell.update(side=payload_side, payload_row=payload_row, dose=arm["dose"])
            outcome = self.run_arm(session, ids[side], arm, source, origin)
            outcome.update({"session_id": session.session_id, "before_act_cut": before,
                "act": act, "source_row": source, "target_row": target, "payload_row": payload_row,
                "payload_side": payload_side, "cell_origins": origin})
            save(self.root / "arms" / (name + ".json"), outcome)
            outcomes.append({"arm": name, "own_text": outcome["own_text"], "cells": len(arm["cells"])})
            print("SCAFFOLD_PROGRESS " + json.dumps({"arm": name, "output": outcome["own_text"],
                "operations": self.operations}), flush=True)
        save(self.root / "experiment.json", {"case": case, "source_row": source, "wrong_row": wrong,
            "token_ids": ids, "token_labels": {side: [self.tokenizer.decode([t]) for t in ids[side]] for side in ids},
            "native_parents": parents, "plan": plan(), "outcomes": outcomes})
        return outcomes


def main():
    import parameter_source
    config = json.loads(Path("saturn_run_config.json").read_text())
    request = json.loads(Path("saturn_debug_request.json").read_text())
    job = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "unknown"))
    root = Path(config["output_root"]) / job
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = parameter_source._write_parameter_manifest(Path(config["model_path"]), root / "parameter-artifacts.json")
    spec, model, tokenizer, weights, _, _ = parameter_source._load_model(config, manifest["provenance"])
    save(root / "request.json", request)
    panel = Panel(model, tokenizer, spec, weights, root)
    outcomes = panel.experiment(config["cases"][0])
    summary = {"schema": "mdb-value-coalition-v1", "job_id": job,
        "request_sha256": request["request_sha256"], "model_identity": weights,
        "worker_wall_s": time.monotonic() - started, "native_operations": panel.operations,
        "outcomes": outcomes, "mechanics_status": "collected; saved audit pending",
        "instrument_status": "typed cell-level value replacements and observed head uses; semantic role supplied",
        "terminal_status": "not-requested; exploratory"}
    save(root / "report.json", summary)
    inventory = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
        "sha256": digest_file(p)} for p in sorted(root.rglob("*")) if p.is_file()]
    save(root / "output-manifest.json", {"files": inventory, "expanded_bytes": sum(p["bytes"] for p in inventory)})
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix())
    raw = stream.getvalue()
    if len(raw) > request["workload_geometry"]["output_archive_max_bytes"]:
        raise RuntimeError("archive exceeds declared transport bound; raw evidence retained")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i+(512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__":
    main()
