"""Attention nominates native source addresses before typed intervention execution."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import torch
from saturn_pub.core import Act

from coalition_support import Panel as CoalitionPanel
from mdb.attention_observer import AttentionObserver
from mdb.attention_source_resolver import CAPTURE_LAYERS, READER_HEAD, READER_LAYER, arms, nominate
from mdb.saturn_pub_adapter import SaturnPubDebuggerAdapter
from mdb.value_coalition import replacement
from scaffold_io import digest_file, save


class Panel(CoalitionPanel):
    def open_prompt(self, name, ids):
        # This boundary depends only on length, with no role/address annotation.
        prefix_length = max(1, len(ids) - 3)
        adapter = SaturnPubDebuggerAdapter(self.native.session(ids[:prefix_length]),
            project=self.project, session_id=name)
        adapter.session.continue_(33)
        self.operations += 33
        adapter.checkpoint()
        for index, token in enumerate(ids[prefix_length:], start=prefix_length):
            self.native.stage_prompt_token(adapter.session, token)
            adapter.checkpoint()
            if index < len(ids) - 1:
                adapter.session.continue_(33)
                self.operations += 33
                adapter.checkpoint()
        return adapter, adapter.checkpoint()["cut_id"]

    def resolve(self, adapter, ids, label):
        parent = adapter.checkpoint()["cut_id"]
        cached_rows = adapter.session.read("values")[READER_LAYER].shape[2]
        started = time.monotonic()
        with AttentionObserver(self.model) as observer:
            receipt = adapter.session.continue_(READER_LAYER + 2)
        self.operations += READER_LAYER + 2
        discovered = adapter.checkpoint()["cut_id"]
        captured = next(row for row in observer.rows if row["layer"] == READER_LAYER)
        weights = captured["attention"][0, READER_HEAD, 0].float().cpu().tolist()
        decision = nominate(weights, cached_rows)
        trace = {"layer": READER_LAYER, **{key: self.storage.tensor(label + "-discovery-" + key, captured[key])
            for key in ("query", "key", "value", "attention", "head_return")}}
        adapter.restore(parent)
        restored = adapter.checkpoint()["cut_id"]
        if restored != parent:
            raise ValueError("discovery did not restore its exact native parent")
        return {"parent": parent, "discovery_cut": discovered, "receipt": receipt.fingerprint,
            "restored_parent": restored, "cached_rows": cached_rows, "input_ids": ids,
            "nomination": decision, "trace": trace, "wall_s": time.monotonic() - started}

    def case(self, case):
        # No offset mapping, word search, supplied span, or cross-prompt difference is used.
        ids = {side: self.tokenizer.encode(case[side]["text"], add_special_tokens=False)
            for side in ("recipient", "donor")}
        if any(not seq or len(seq) > 128 for seq in ids.values()):
            raise ValueError("prompt exceeds the declared geometry")
        sessions, parents, values, resolutions = {}, {}, {}, {}
        for side in ("recipient", "donor"):
            sessions[side], parents[side] = self.open_prompt("public-" + case["id"] + "-" + side, ids[side])
            resolutions[side] = self.resolve(sessions[side], ids[side], case["id"] + "-" + side)
            values[side] = sessions[side].session.read("values")
        outcomes, skipped = [], []
        for config in arms():
            short, kind = config["arm"], config["kind"]
            arm = {**config, "arm": case["id"] + "--" + short}
            side = "donor" if kind in {"donor-native", "reciprocal"} else "recipient"
            payload_side = "recipient" if kind in {"sham", "reciprocal"} else "donor"
            observed = resolutions[side]["nomination"]["source_row"]
            target = observed
            payload = resolutions[payload_side]["nomination"]["source_row"]
            if kind == "wrong-target" and target is not None:
                target -= 1
            if kind == "wrong-payload" and payload is not None:
                payload -= 1
            if kind == "fixed-distance":
                target, payload = len(ids[side])-4, len(ids[payload_side])-4
            native = kind in {"native", "donor-native"}
            if not native and (observed is None or target is None or payload is None or target < 0 or payload < 0
                    or target >= values[side][20].shape[2] or payload >= values[payload_side][20].shape[2]):
                skipped.append({"arm": arm, "reason": "nominated-or-control-row-outside-retained-prefix"})
                continue
            adapter = sessions[side]
            adapter.restore(parents[side])
            before = adapter.checkpoint()["cut_id"]
            act = None
            if not native:
                installed = replacement(adapter.session.read("values"), values[payload_side],
                    config["cells"], target, payload)
                act = adapter.session.apply(Act.replace("values", installed)).to_dict()
                adapter.checkpoint()
            evidence = tuple("cas://sha256/" + item for item in (
                resolutions[side]["receipt"], resolutions[side]["trace"]["attention"]["sha256"]))
            origin = {"resolver": resolutions[side], "payload_resolver": resolutions[payload_side],
                "target_row": target, "payload_row": payload, "payload_side": payload_side,
                "arm_kind": kind, "cells": config["cells"], "semantic_role": "supplied task family; row nominated by native attention"}
            started = time.monotonic()
            outcome = self.run_arm(adapter, ids[side], arm, observed if observed is not None else 0, origin,
                symbol_layers=CAPTURE_LAYERS if observed is not None else (), capture_layers=CAPTURE_LAYERS,
                operation_limit=8500, source_evidence=evidence)
            outcome.update({"case_id": case["id"], "session_id": adapter.session_id,
                "before_act_cut": before, "act": act, "source_row": observed,
                "target_row": target, "payload_row": payload, "payload_side": payload_side,
                "branch_wall_s": time.monotonic()-started})
            save(self.root / "arms" / (arm["arm"] + ".json"), outcome)
            outcomes.append({"case": case["id"], "arm": short, "own_text": outcome["own_text"]})
            print("SCAFFOLD_PROGRESS " + json.dumps({**outcomes[-1], "nominated_row": observed,
                "operations": self.operations}), flush=True)
        save(self.root / "cases" / (case["id"] + ".json"), {"case": case, "native_parents": parents,
            "token_ids": ids, "token_labels": {side: [self.tokenizer.decode([t]) for t in seq] for side, seq in ids.items()},
            "resolutions": resolutions, "outcomes": outcomes, "skipped": skipped,
            "prompt_ingestion": "constant-final-four-token-tail; length-only boundary"})
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
    outcomes = [row for case in config["cases"] for row in panel.case(case)]
    summary = {"schema": "mdb-live-source-resolver-v1", "job_id": job, "request_sha256": request["request_sha256"],
        "model_identity": weights, "worker_wall_s": time.monotonic()-started, "native_operations": panel.operations,
        "outcomes": outcomes, "mechanics_status": "collected; saved audit pending",
        "instrument_status": "native partial-pass nomination, exact restore, nomination-addressed Acts; evaluator withheld",
        "terminal_status": "not-requested; exploratory"}
    save(root / "report.json", summary)
    entries = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": digest_file(p)}
        for p in sorted(root.rglob("*")) if p.is_file()]
    save(root / "output-manifest.json", {"files": entries, "expanded_bytes": sum(p["bytes"] for p in entries)})
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file(): archive.add(path, arcname=path.relative_to(root).as_posix())
    raw = stream.getvalue()
    if len(raw) > request["workload_geometry"]["output_archive_max_bytes"]:
        raise RuntimeError("archive exceeds declared transport bound; native evidence retained")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i+(512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__": main()
