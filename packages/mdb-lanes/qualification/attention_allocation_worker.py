"""Finite attention-allocation microscope on the existing semantic organism."""
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

from mdb_lanes.attention_observer import AttentionObserver
from mdb_lanes.bench import BenchSession
from mdb_lanes.bench_hidden_memory import logical_address
from mdb_lanes.project import Project
from mdb_lanes.prompt_qwen import PromptQwenAdapter
from mdb_lanes.saturn_pub_adapter import SaturnPubDebuggerAdapter
from mdb_lanes.semantic_scaffold import resolve_span
from scaffold_io import Assay, digest_file, save


class Panel:
    def __init__(self, model, tokenizer, spec, weights, root):
        self.model, self.tokenizer, self.spec, self.weights, self.root = model, tokenizer, spec, weights, root
        self.storage = Assay(model, tokenizer, root)
        self.native = PromptQwenAdapter(model)
        self.project = Project(root / "project")
        self.operations = 0

    def opened(self, lane, side, ids, source_row):
        if lane == "bench":
            session = BenchSession.from_resident(self.model, self.tokenizer,
                root=self.root / "bench" / side, model_id=self.spec.model_id,
                revision=self.spec.revision, model_sha256=self.weights, lane="virtual",
                attention_backend="native-eager", resident_kv_bytes=128 << 20,
                continuation_capacity=8, publication_mode="journal")
            session.prefill(ids[:-1])
            parent = session.branch.cut
        else:
            session = SaturnPubDebuggerAdapter(self.native.session(ids[:source_row + 1]),
                project=self.project, session_id=lane + "-" + side)
            session.session.continue_(33)
            self.operations += 33
            session.checkpoint()
            for index, token in enumerate(ids[source_row + 1:], start=source_row + 1):
                self.native.stage_prompt_token(session.session, token)
                session.checkpoint()
                if index < len(ids) - 1:
                    session.session.continue_(33)
                    self.operations += 33
                    session.checkpoint()
            parent = session.checkpoint()["cut_id"]
        return session, parent

    def values(self, lane, session):
        if lane == "bench":
            return [session.resolve_value(f"kv_cache.layer_{l:03d}.value").detach().clone()
                for l in range(28)]
        return session.session.read("values")

    def install(self, lane, session, layers, source_row, donor):
        if lane == "bench":
            return session.memory_write({"mode": "replace", "selector": {"layers": layers,
                "row_start": source_row, "row_stop": source_row + 1, "components": ["value"]},
                "values": {logical_address(l, "value", source_row): donor[l][:, :, source_row:source_row + 1].float().tolist()
                    for l in layers}})
        values = session.session.read("values")
        for layer in layers:
            values[layer][:, :, source_row] = donor[layer][:, :, source_row]
        result = session.session.apply(Act.replace("values", values)).to_dict()
        session.checkpoint()
        return result

    def run(self, lane, session, ids, label, *, observe):
        trajectory, own, traces = [], [], []
        for step in range(3):
            if lane == "bench":
                if step == 0:
                    session.stage_input(ids[-1])
                parent = session.branch.cut
                history = list(session.branch.token_ids)
                context = AttentionObserver(self.model) if observe else None
                if context:
                    with context:
                        receipt = session.step("token")
                else:
                    receipt = session.step("token")
                logits = session.resolve_value("logits").detach().clone()
                result_cut = session.branch.cut
                committed = receipt["token_id"]
                receipt_ref = receipt["receipt_ref"]
            else:
                parent = session.checkpoint()["cut_id"]
                history = session.session.read("tokens").tolist()[0]
                context = AttentionObserver(self.model) if observe else None
                if context:
                    with context:
                        receipt = session.session.continue_(31)
                else:
                    receipt = session.session.continue_(31)
                session.checkpoint()
                logits = session.session.read("logits")
                session.session.continue_(2)
                result_cut = session.checkpoint()["cut_id"]
                committed = history[-1]
                receipt_ref = receipt.fingerprint
            self.operations += 33
            if self.operations > 4000:
                raise RuntimeError("declared native operation budget exceeded")
            token = int(logits[0].argmax())
            expected_input = ids + own
            if committed != expected_input[-1]:
                raise RuntimeError("native receiver did not consume its declared prompt/own token")
            if lane != "bench" and session.session.read("tokens").tolist()[0] != expected_input + [token]:
                raise RuntimeError("public token commit/own feedback mismatch")
            own.append(token)
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite native logits")
            row = {"step": step, "parent_cut": parent, "result_cut": result_cut,
                "receipt": receipt_ref, "history_ids": history, "consumed_token_id": committed,
                "input_ids": expected_input,
                "next_token_id": token, "next_token_text": self.tokenizer.decode([token]),
                "logits": self.storage.tensor(label + f"-s{step}-logits", logits)}
            trajectory.append(row)
            if context:
                if [r["layer"] for r in context.rows] != list(range(28)):
                    raise RuntimeError("attention observer missed a native decoder layer")
                items = []
                for captured in context.rows:
                    layer = captured["layer"]
                    item = {"layer": layer}
                    for key in ("query", "key", "value", "attention", "head_return"):
                        if not torch.isfinite(captured[key]).all():
                            raise RuntimeError("nonfinite native attention evidence")
                        item[key] = self.storage.tensor(label + f"-s{step}-L{layer}-{key}", captured[key])
                    items.append(item)
                traces.append({"step": step, "layers": items})
        return {"arm": label, "lane": lane, "own_ids": own, "own_text": self.tokenizer.decode(own),
            "trajectory": trajectory, "traces": traces}

    def lane(self, lane, case):
        encoded = {side: self.tokenizer(case[side]["text"], add_special_tokens=False,
            return_offsets_mapping=True) for side in ("recipient", "donor")}
        source = {side: resolve_span(encoded[side]["offset_mapping"], case[side]["source_span"])
            for side in encoded}
        if source["recipient"] != source["donor"] or len(source["recipient"]) != 1:
            raise ValueError("matched single-token semantic source required")
        row = source["recipient"][0]
        ids = {side: encoded[side]["input_ids"] for side in encoded}
        recipient, parent = self.opened(lane, "recipient", ids["recipient"], row)
        donor, _ = self.opened(lane, "donor", ids["donor"], row)
        rv, dv = self.values(lane, recipient), self.values(lane, donor)
        results = []
        results.append(self.run(lane, recipient, ids["recipient"], lane + "-unobserved-native", observe=False))
        recipient.restore(parent)
        native = self.run(lane, recipient, ids["recipient"], lane + "-native", observe=True)
        results.append(native)
        results.append(self.run(lane, donor, ids["donor"], lane + "-donor-native", observe=True))
        for name, layers, values in (("sham", list(range(28)), rv),
                ("donor-late", list(range(19, 28)), dv), ("donor-all", list(range(28)), dv)):
            recipient.restore(parent)
            act = self.install(lane, recipient, layers, row, values)
            result = self.run(lane, recipient, ids["recipient"], lane + "-" + name, observe=True)
            result["act"] = act
            results.append(result)
        recipient.restore(parent)
        if lane == "bench":
            recipient.close()
            donor.close()
        body = {"lane": lane, "source_row": row, "case": case,
            "token_ids": ids, "token_text": {side: [self.tokenizer.decode([t]) for t in seq] for side, seq in ids.items()},
            "results": results, "parent": parent}
        save(self.root / (lane + "-result.json"), body)
        print("SCAFFOLD_PROGRESS " + json.dumps({"lane": lane,
            "arms": [{"arm": r["arm"], "output": r["own_text"]} for r in results]}), flush=True)
        return body


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
    spec, model, tokenizer, weights_sha, _, _ = parameter_source._load_model(config, manifest["provenance"])
    save(root / "request.json", request)
    panel = Panel(model, tokenizer, spec, weights_sha, root)
    results = [panel.lane(lane, config["cases"][0]) for lane in ("bench", "public")]
    summary = {"schema": "mdb-attention-allocation-v1", "job_id": job,
        "request_sha256": request["request_sha256"], "model_identity": weights_sha,
        "worker_wall_s": time.monotonic() - started, "native_operations": panel.operations,
        "cases": [{"lane": r["lane"], "arms": [{"arm": a["arm"], "output": a["own_text"]} for a in r["results"]]} for r in results],
        "mechanics_status": "collected; passive parity and attention arithmetic require saved audit",
        "terminal_status": "not-requested; exploratory"}
    save(root / "report.json", summary)
    entries = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
        "sha256": digest_file(p)} for p in sorted(root.rglob("*")) if p.is_file()]
    save(root / "output-manifest.json", {"files": entries, "expanded_bytes": sum(p["bytes"] for p in entries)})
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix())
    raw = stream.getvalue()
    if len(raw) > request["workload_geometry"]["output_archive_max_bytes"]:
        raise RuntimeError("archive exceeds declared bound; remote evidence retained")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i + (512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__":
    main()
