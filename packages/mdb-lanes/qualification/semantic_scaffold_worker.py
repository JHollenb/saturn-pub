"""Finite source-aware semantic/copy/recall/code assay through native MDB cuts."""
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

from mdb_lanes.bench import BenchSession
from mdb_lanes.bench_hidden_memory import logical_address
from mdb_lanes.bench_qwen import tensor_digest
from mdb_lanes.live_scaffold import fingerprint
from mdb_lanes.semantic_scaffold import evaluate, resolve_span
from scaffold_io import Assay, digest_file, save


def token_aliases(tokenizer, word):
    ids = []
    for text in (word, " " + word):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        first = tokenizer.decode([encoded[0]]).strip() if encoded else ""
        if first and word.startswith(first):
            ids.append(encoded[0])
    if not ids:
        ids = tokenizer.encode(word, add_special_tokens=False)[:1]
    return sorted(set(ids))


class Ports:
    def __init__(self, model, layer):
        self.model, self.layer, self.handles = model, layer, []
        self.values = {}

    def __enter__(self):
        def reader(_module, args):
            self.values["reader"] = args[0][:, -1].detach().clone()
        def writer(_module, _args, value):
            self.values["writer"] = value[:, -1].detach().clone()
        self.handles.append(self.model.model.layers[self.layer].self_attn.o_proj.register_forward_pre_hook(reader))
        self.handles.append(self.model.model.layers[min(self.layer + 1, len(self.model.model.layers) - 1)].mlp.register_forward_hook(writer))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


class Panel:
    def __init__(self, model, tokenizer, spec, weight_sha, root, *, depth_bands=False):
        self.model, self.tokenizer, self.spec, self.weight_sha, self.root = model, tokenizer, spec, weight_sha, root
        self.storage = Assay(model, tokenizer, root)
        self.steps, self.benches = 0, []
        self.depth_bands = depth_bands

    def bench(self, case, side):
        text = case[side]["text"]
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if len(ids) > 128:
            raise ValueError("declared input length exceeded")
        source = resolve_span(encoded["offset_mapping"], case[side]["source_span"])
        wrong = resolve_span(encoded["offset_mapping"], case[side]["wrong_span"])
        if max(source + wrong) >= len(ids) - 1:
            raise ValueError("candidate sources must be committed before receiver")
        bench = BenchSession.from_resident(self.model, self.tokenizer,
            root=self.root / "native-sessions" / (case["id"] + "-" + side),
            model_id=self.spec.model_id, revision=self.spec.revision, model_sha256=self.weight_sha,
            lane="virtual", attention_backend="native-eager", resident_kv_bytes=128 << 20,
            continuation_capacity=8, publication_mode="journal")
        self.benches.append(bench)
        bench.prefill(ids[:-1])
        return bench, {"token_ids": ids, "source_rows": source, "wrong_rows": wrong,
                       "token_text": [self.tokenizer.decode([i]) for i in ids],
                       "offsets": encoded["offset_mapping"], "parent_cut": bench.branch.cut,
                       "input_sha256": fingerprint(ids), "span_authority": "supplied experiment grammar"}

    def kv_hashes(self, bench):
        return {str(layer): {kind: tensor_digest(bench.resolve_value(f"kv_cache.layer_{layer:03d}.{kind}"))
                            for kind in ("key", "value")}
                for layer in range(self.model.config.num_hidden_layers)}

    def run(self, bench, info, label, *, layer=None):
        rows, tokens, first_logits, first_ports = [], [], None, None
        for step in range(3):
            self.steps += 1
            if self.steps > 256:
                raise RuntimeError("declared explicit native-transition budget exceeded")
            if step == 0:
                bench.stage_input(info["token_ids"][-1])
            parent = bench.branch.cut
            if layer is not None and step == 0:
                with Ports(self.model, layer) as ports:
                    receipt = bench.step("token")
                first_ports = {key: value for key, value in ports.values.items()}
            else:
                receipt = bench.step("token")
            logits = bench.resolve_value("logits").detach().clone()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite native logits")
            if step == 0:
                first_logits = logits
            token = int(logits[0].argmax())
            tokens.append(token)
            row = {"label": label + f"-s{step}", "offset": step, "parent_cut": parent,
                   "cut_id": bench.branch.cut, "receipt_ref": receipt["receipt_ref"],
                   "consumed_token_id": receipt["token_id"],
                   "input_authority": "caller-last-input" if step == 0 else "own-native-previous-output",
                   "history_ids": list(bench.branch.token_ids), "history_sha256": fingerprint(bench.branch.token_ids),
                   "next_token_id": token, "next_token_text": self.tokenizer.decode([token]),
                   "logits": self.storage.tensor(label + f"-s{step}-logits", logits)}
            if step == 0:
                row["kv_hashes"] = self.kv_hashes(bench)
                if first_ports:
                    row["ports"] = {key: self.storage.tensor(label + "-" + key, value)
                                    for key, value in first_ports.items()}
            rows.append(row)
        return {"own_token_ids": tokens, "own_text": self.tokenizer.decode(tokens), "trajectory": rows}, first_logits, first_ports

    def value(self, bench, layer, row):
        return bench.resolve_value(f"kv_cache.layer_{layer:03d}.value")[:, :, row:row + 1].detach().cpu().clone()

    def write(self, bench, changes, label):
        # Each typed transaction preserves every unselected head in the row.
        payloads, prepared = [], []
        for layer, row, head, kind, donor_value, dose in changes:
            before = self.value(bench, layer, row)
            after = before.clone()
            if kind == "scale":
                if head is None:
                    after *= dose
                else:
                    after[:, head:head + 1] *= dose
            elif head is None:
                after.copy_(donor_value)
            else:
                after[:, head:head + 1] = donor_value[:, head:head + 1]
            prepared.append((layer, row, head, before, after))
        rows = {item[1] for item in prepared}
        layers = sorted({item[0] for item in prepared})
        if len(prepared) != len(layers) * (max(rows) - min(rows) + 1):
            raise ValueError("source Act must cover its declared layer/row rectangle exactly")
        selector = {"layers": layers, "row_start": min(rows), "row_stop": max(rows) + 1, "components": ["value"]}
        act = bench.memory_write({"mode": "replace", "selector": selector,
            "values": {logical_address(layer, "value", row): after.float().tolist()
                       for layer, row, head, before, after in prepared}})
        for layer, row, head, before, after in prepared:
            actual = self.value(bench, layer, row)
            if not torch.equal(actual, after):
                raise RuntimeError("typed value installation mismatch")
            if head is not None:
                mask = torch.arange(before.shape[1]) != head
                if not torch.equal(actual[:, mask], before[:, mask]):
                    raise RuntimeError("typed value edit escaped selected head")
            payloads.append({"layer": layer, "row": row, "kv_head": head,
                "before": self.storage.tensor(label + f"-L{layer}-R{row}-before", before),
                "after": self.storage.tensor(label + f"-L{layer}-R{row}-after", actual)})
        return {"acts": [act], "payloads": payloads}

    def case(self, case, frozen):
        rec, ri = self.bench(case, "recipient")
        donor, di = self.bench(case, "donor")
        baseline, logits, _ = self.run(rec, ri, case["id"] + "-native")
        donor_native, donor_logits, _ = self.run(donor, di, case["id"] + "-donor-native")
        rec.restore(ri["parent_cut"])
        donor.restore(di["parent_cut"])
        a = max(token_aliases(self.tokenizer, case["expected"]), key=lambda i: float(logits[0, i]))
        b = max(token_aliases(self.tokenizer, case["donor_expected"]), key=lambda i: float(donor_logits[0, i]))
        if a == b:
            raise ValueError("semantic contrast has no distinct first-token aliases")
        discovery = None
        if case["split"] == "discovery":
            discovery = rec.discover_circuit({"name": case["id"] + "-source-restricted",
                "plan": [{"kind": "input", "token_ids": ri["token_ids"][-1:]}],
                "receiver_offsets": [0], "layers": list(range(self.model.config.num_hidden_layers)),
                "row_start": min(ri["source_rows"]), "row_stop": max(ri["source_rows"]) + 1,
                "candidates_per_receiver": 2, "scales": [0.0], "continuation_tokens": 1,
                "observable": {"kind": "token_margin", "token_id": b, "against_token_id": a},
                "top_k": 1, "max_native_operations": 2000, "max_capture_bytes": 256 << 20})["report"]
            save(self.root / (case["id"] + "-discovery.json"), discovery)
            selected = max(discovery["tests"], key=lambda t: t["effect"]["steps"][0]["signed_margin_effect"])
            candidate = next(c for c in discovery["receivers"][0]["candidates"] if c["id"] == selected["candidate_id"])
            frozen = {"layer": candidate["layer"], "kv_head": candidate["kv_head"],
                      "source_index": ri["source_rows"].index(candidate["source_row"]),
                      "discovered_on": case["id"], "selection": "largest donor-margin shift after zeroing among two source-restricted read-score nominations",
                      "discovery_signed_effect": selected["effect"]["steps"][0]["signed_margin_effect"]}
            save(self.root / (case["lane"] + "-frozen-route.json"), frozen)
        layer, head = frozen["layer"], frozen["kv_head"]
        index = frozen["source_index"]
        if index >= min(len(ri["source_rows"]), len(di["source_rows"])):
            raise ValueError("frozen relative source token unavailable; new tokenization needs separate resolver")
        source, other = ri["source_rows"][index], di["source_rows"][index]
        wrong = ri["wrong_rows"][-1]
        rec.restore(ri["parent_cut"])
        donor.restore(di["parent_cut"])
        reference, reference_logits, reference_ports = self.run(rec, ri, case["id"] + "-observed", layer=layer)
        observer_exact = torch.equal(reference_logits, logits)
        base_margin = float(logits[0, b].float() - logits[0, a].float())
        rec.restore(ri["parent_cut"])
        rec_value, donor_value = self.value(rec, layer, source), self.value(donor, layer, other)
        arms = [
            ("sham", [(layer, source, head, "replace", rec_value, None)]),
            ("zero-role", [(layer, source, head, "scale", None, 0.0)]),
            ("half-role", [(layer, source, head, "scale", None, 0.5)]),
            ("donor-role", [(layer, source, head, "replace", donor_value, None)]),
            ("wrong-source-zero", [(layer, wrong, head, "scale", None, 0.0)]),
            ("donor-other-head", [(layer, source, 1 - head, "replace", donor_value, None)]),
            ("donor-all-depth", [(l, source, None, "replace", self.value(donor, l, other), None)
                                  for l in range(self.model.config.num_hidden_layers)]),
        ]
        if self.depth_bands and case["lane"] == "coding":
            all_changes = arms[-1][1]
            for name, low, high in (("early", 0, 9), ("middle", 9, 19), ("late", 19, 28)):
                arms.append(("donor-depth-" + name, [item for item in all_changes if low <= item[0] < high]))
                arms.append(("donor-without-" + name, [item for item in all_changes if not low <= item[0] < high]))
        if case.get("span_lowering") and len(ri["source_rows"]) > 1 and len(di["source_rows"]) == 1:
            tail = [(l, row, None, "scale", None, 0.0)
                    for l in range(self.model.config.num_hidden_layers) for row in ri["source_rows"][1:]]
            first = [(l, source, None, "replace", self.value(donor, l, other), None)
                     for l in range(self.model.config.num_hidden_layers)]
            arms.append(("erase-source-tail", tail))
            arms.append(("donor-first-row-erase-tail", first + tail))
        outcomes = []
        for name, changes in arms:
            rec.restore(ri["parent_cut"])
            installed = self.write(rec, changes, case["id"] + "-" + name)
            future, actual, ports = self.run(rec, ri, case["id"] + "-" + name, layer=layer)
            delta = actual.float() - logits.float()
            changed = [l for l, value in future["trajectory"][0]["kv_hashes"].items()
                       if value != baseline["trajectory"][0]["kv_hashes"][l]]
            outcome = {"arm": name, "installed": installed, "future": future,
                       "recipient_score": evaluate(case, future["own_text"]),
                       "donor_score": evaluate(case, future["own_text"], donor=True),
                       "donor_margin_effect": float(actual[0, b].float() - actual[0, a].float()) - base_margin,
                       "first_logits_exact": torch.equal(actual, logits), "logit_l2": float(delta.norm()),
                       "writer_l2": float((ports["writer"].float() - reference_ports["writer"].float()).norm()),
                       "reader_l2": float((ports["reader"].float() - reference_ports["reader"].float()).norm()),
                       "changed_kv_layers": changed}
            outcomes.append(outcome)
        rec.restore(ri["parent_cut"])
        restored = tensor_digest(rec.resolve_value("logits"))
        result = {"case": case, "recipient_binding": ri, "donor_binding": di,
                  "frozen_route": frozen, "discovery_report": case["id"] + "-discovery.json" if discovery else None,
                  "recipient_native": baseline, "donor_native": donor_native, "observed_native": reference,
                  "recipient_score": evaluate(case, baseline["own_text"]),
                  "donor_score": evaluate(case, donor_native["own_text"], donor=True),
                  "observable": {"donor_token_id": b, "recipient_token_id": a, "baseline_margin": base_margin,
                                 "arithmetic": "FP32 subtraction of native logits; first receiver only"},
                  "observer_first_logits_exact": observer_exact, "arms": outcomes,
                  "restored_parent_cut": rec.branch.cut, "restored_parent_logits_sha256": restored,
                  "mechanics_status": "valid" if observer_exact else "observer-parity-mismatch",
                  "semantic_status": "supplied labels and source spans; discovery-selection and frozen transfer separate"}
        save(self.root / (case["id"] + "-result.json"), result)
        rec.close()
        donor.close()
        self.benches.clear()
        print("SCAFFOLD_PROGRESS " + json.dumps({"case": case["id"], "native": baseline["own_text"],
              "donor": donor_native["own_text"], "frozen_route": frozen, "steps": self.steps}), flush=True)
        return result, frozen


def main():
    import parameter_source
    started = time.monotonic()
    config = json.loads(Path("saturn_run_config.json").read_text())
    request = json.loads(Path("saturn_debug_request.json").read_text())
    job_id = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "unknown"))
    root = Path(config["output_root"]) / job_id
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = parameter_source._write_parameter_manifest(Path(config["model_path"]), root / "parameter-artifacts.json")
    spec, model, tokenizer, weights_sha, files, content_sha = parameter_source._load_model(config, manifest["provenance"])
    if model.config.num_key_value_heads != 2:
        raise ValueError("the paired-head control requires this specimen's two KV groups")
    save(root / "request.json", request)
    panel = Panel(model, tokenizer, spec, weights_sha, root, depth_bands=config.get("depth_bands", False))
    frozen, summaries = dict(config.get("frozen_routes", {})), []
    for case in config["cases"]:
        result, route = panel.case(case, frozen.get(case["lane"]))
        frozen[case["lane"]] = route
        summaries.append({"id": case["id"], "lane": case["lane"], "split": case["split"],
            "recipient_score": result["recipient_score"], "donor_score": result["donor_score"],
            "frozen_route": route, "observer_first_logits_exact": result["observer_first_logits_exact"],
            "arms": [{k: a[k] for k in ("arm", "recipient_score", "donor_score", "donor_margin_effect",
                                       "logit_l2", "writer_l2", "changed_kv_layers")} for a in result["arms"]]})
    summary = {"schema": "mdb-semantic-scaffold-assay-v1", "job_id": job_id,
               "request_sha256": request["request_sha256"], "model_identity": weights_sha,
               "weight_files": files, "weight_content_sha256": content_sha, "cases": summaries,
               "explicit_native_steps": panel.steps, "worker_wall_s": time.monotonic() - started,
               "mechanics_status": "valid" if all(c["observer_first_logits_exact"] for c in summaries) else "observer-parity-mismatch",
               "instrument_status": "source-restricted discovery nominations and frozen layer/KV-head/relative-token transfer",
               "trend_status": "per-case native behavior, signed effects, controls and state changes retained",
               "terminal_status": "not-requested; exploratory"}
    save(root / "report.json", summary)
    inventory = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": digest_file(p)}
                 for p in sorted(root.rglob("*")) if p.is_file()]
    save(root / "output-manifest.json", {"files": inventory, "expanded_bytes": sum(x["bytes"] for x in inventory)})
    archive_io = io.BytesIO()
    with tarfile.open(fileobj=archive_io, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix())
    raw = archive_io.getvalue()
    if len(raw) > request["workload_geometry"]["output_archive_max_bytes"]:
        raise RuntimeError("archive exceeds declared transport bound; raw remote evidence retained")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i + (512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__":
    main()
