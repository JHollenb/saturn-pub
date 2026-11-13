"""Finite native Qwen role-binding, argument intervention and output assay."""
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

from mdb_lanes.bench_qwen import tensor_digest
from mdb_lanes.live_scaffold import (ADDITION_ROLES, fingerprint, invocation_graph,
                               render_html, resolve_addition, scaffold)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Capture:
    """Observe actual native ports; replacements act only on declared heads."""
    def __init__(self, model, edit=None):
        self.model, self.edit, self.handles = model, edit, []
        self.heads, self.writes = {}, {}

    def __enter__(self):
        for layer in (20, 22):
            attention = self.model.model.layers[layer].self_attn

            def before(_module, args, layer=layer):
                incoming = args[0]
                changed = incoming
                if self.edit and self.edit["layer"] == layer:
                    changed = incoming.clone()
                    width = self.model.config.hidden_size // self.model.config.num_attention_heads
                    for head in self.edit["heads"]:
                        lo, hi = head * width, (head + 1) * width
                        if self.edit["kind"] == "replace":
                            changed[:, -1, lo:hi] = self.edit["value"][:, head, :]
                        else:
                            changed[:, -1, lo:hi] -= self.edit["value"][:, head, :] * self.edit["dose"]
                self.heads[layer] = changed[:, -1].detach().clone().reshape(1, self.model.config.num_attention_heads, -1)
                return (changed, *args[1:]) if changed is not incoming else None

            self.handles.append(attention.o_proj.register_forward_pre_hook(before))

        def writer(_module, _args, output):
            self.writes[23] = output[:, -1].detach().clone()

        self.handles.append(self.model.model.layers[23].mlp.register_forward_hook(writer))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


class Assay:
    def __init__(self, model, tokenizer, root):
        self.model, self.tokenizer, self.root = model, tokenizer, root
        self.calls, self.rows = 0, []

    def tensor(self, label, value):
        import numpy as np
        path = self.root / "arrays" / f"{label}.npy"
        path.parent.mkdir(exist_ok=True)
        cpu = value.detach().contiguous().cpu()
        with path.open("xb") as stream:
            np.save(stream, cpu.view(torch.uint8).numpy(), allow_pickle=False)
        return {"path": path.relative_to(self.root).as_posix(), "sha256": digest_file(path),
                "dtype": str(cpu.dtype), "shape": list(cpu.shape), "bytes": cpu.numel() * cpu.element_size()}

    def forward(self, ids, *, label, past=None, edit=None, observe=True):
        self.calls += 1
        if self.calls > 260:
            raise RuntimeError("declared forward budget exceeded")
        inputs = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            if observe:
                with Capture(self.model, edit) as capture:
                    out = self.model(input_ids=inputs, past_key_values=past, use_cache=True,
                                     output_attentions=True, logits_to_keep=1)
            else:
                capture = None
                out = self.model(input_ids=inputs, past_key_values=past, use_cache=True,
                                 output_attentions=False, logits_to_keep=1)
        logits = out.logits[:, -1]
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite native logits")
        raw = self.tensor(label + "-logits", logits)
        top = logits.float()[0].topk(5)
        row = {"label": label, "next_token_id": int(top.indices[0]),
               "next_token_text": self.tokenizer.decode([int(top.indices[0])]),
               "logits_sha256": tensor_digest(logits), "logits": raw,
               "top": [{"id": int(i), "text": self.tokenizer.decode([int(i)]), "logit": float(v)}
                       for i, v in zip(top.indices, top.values)],
               "cache_rows": int(out.past_key_values.get_seq_length())}
        if capture:
            row["writer"] = self.tensor(label + "-writer23", capture.writes[23])
            row["head_returns"] = {str(layer): self.tensor(label + f"-heads{layer}", value)
                                   for layer, value in capture.heads.items()}
        self.rows.append(row)
        return out, capture, row

    def observe_case(self, case):
        encoded = self.tokenizer(case["prompt"], add_special_tokens=False, return_offsets_mapping=True)
        base = encoded["input_ids"]
        ids, text, occurrences = list(base), "", []
        first_logits = None
        for step in range(case["digits"]):
            resolved = resolve_addition(case["prompt"], base, encoded["offset_mapping"], step)
            out, capture, row = self.forward(ids, label=case["id"] + f"-s{step}")
            if step == 0:
                first_logits = out.logits[:, -1].detach().clone()
            for binding in resolved["bindings"]:
                layer = binding["layer"]
                binding["head_output_sha256"] = tensor_digest(capture.heads[layer][:, binding["heads"]])
                weights = out.attentions[layer][0, :, -1]
                binding["native_argmax_rows"] = {str(h): int(weights[h].argmax()) for h in binding["heads"]}
                for argument in binding["arguments"]:
                    source = argument["source_row"]
                    argument["attention_by_head"] = {str(h): float(weights[h, source]) for h in binding["heads"]}
                    distance = 7 if argument["operand"] == "a" else 3
                    fixed = len(ids) - 1 - distance + next(r.column_below for r in ADDITION_ROLES if r.name == binding["role"])
                    argument["fixed_3digit_distance_row"] = fixed
                    argument["fixed_distance_attention"] = {str(h): float(weights[h, fixed]) if 0 <= fixed < len(ids) else None
                                                            for h in binding["heads"]}
                    wrong_column = (argument["column_from_right"] + 1) % resolved["operand_width"]
                    wrong = resolved["columns"][f"{argument['operand']}{wrong_column}"]["source_row"]
                    argument["shuffled_column_row"] = wrong
                    argument["shuffled_attention"] = {str(h): float(weights[h, wrong]) for h in binding["heads"]}
                    argument["control_heads_attention"] = {str(h): float(weights[h, source]) for h in (0, 3)}
            occurrence = {"case": case["id"], "answer_step": step,
                          "history_sha256": fingerprint(ids), "history_ids": list(ids),
                          "generated_prefix": text, "input_authority": "caller prompt plus own native outputs",
                          "semantic_step_alignment": all(c.isdigit() for c in text) and len(text) == step,
                          "bindings": resolved["bindings"], **row}
            occurrences.append(occurrence)
            token = row["next_token_id"]
            ids.append(token)
            text += row["next_token_text"]
            del out, capture
        plain, _, plain_row = self.forward(base, label=case["id"] + "-observer-control", observe=False)
        parity = {"case": case["id"], "logits_exact": bool(torch.equal(first_logits, plain.logits[:, -1])),
                  "max_abs": float((first_logits.float() - plain.logits[:, -1].float()).abs().max()),
                  "plain_logits": plain_row["logits"]}
        expected = str(case["a"] + case["b"])
        return occurrences, {"case": case["id"], "generated": text, "expected": expected,
                             "correct": text == expected, "observer_control": parity}

    def intervene(self, pair):
        rec_ids = self.tokenizer.encode(pair["recipient"], add_special_tokens=False)
        donor_ids = self.tokenizer.encode(pair["donor"], add_special_tokens=False)
        rec, rc, rr = self.forward(rec_ids, label=pair["id"] + "-recipient")
        donor, dc, dr = self.forward(donor_ids, label=pair["id"] + "-donor")
        encoded = self.tokenizer(pair["recipient"], add_special_tokens=False, return_offsets_mapping=True)
        resolved = resolve_addition(pair["recipient"], rec_ids, encoded["offset_mapping"], 0)
        argument = resolved["bindings"][0]["arguments"][0]
        source = argument["source_row"]
        wrong = resolved["columns"][f"a{max(0, pair['digits'] - 2)}"]["source_row"]
        weights = rec.attentions[22][0, :, -1]
        value_cache = rec.past_key_values.layers[22].values[0]
        repeats = self.model.config.num_attention_heads // self.model.config.num_key_value_heads
        def contribution(row):
            return torch.stack([weights[h, row] * value_cache[h // repeats, row]
                                for h in range(self.model.config.num_attention_heads)])[None]
        arms = [
            ("sham", {"kind": "replace", "heads": [6, 8], "value": rc.heads[22]}),
            ("donor-role", {"kind": "replace", "heads": [6, 8], "value": dc.heads[22]}),
            ("donor-control-heads", {"kind": "replace", "heads": [0, 3], "value": dc.heads[22]}),
            ("donor-whole-attention", {"kind": "replace", "heads": list(range(12)), "value": dc.heads[22]}),
            ("remove-correct-source", {"kind": "subtract", "heads": [6, 8], "value": contribution(source), "dose": 1.0}),
            ("half-correct-source", {"kind": "subtract", "heads": [6, 8], "value": contribution(source), "dose": 0.5}),
            ("remove-wrong-source", {"kind": "subtract", "heads": [6, 8], "value": contribution(wrong), "dose": 1.0}),
        ]
        target, alternative = dr["next_token_id"], rr["next_token_id"]
        baseline_margin = float(rec.logits[0, -1, target] - rec.logits[0, -1, alternative])
        results = []
        for name, edit in arms:
            edit["layer"] = 22
            out, capture, row = self.forward(rec_ids, label=pair["id"] + "-" + name, edit=edit)
            difference = out.logits[:, -1].float() - rec.logits[:, -1].float()
            margin = float(out.logits[0, -1, target] - out.logits[0, -1, alternative])
            first = {"arm": name, "next_token": row["next_token_text"],
                     "next_token_id": row["next_token_id"], "margin_effect": margin - baseline_margin,
                     "logit_l2": float(difference.norm()), "logit_max_abs": float(difference.abs().max()),
                     "writer23_l2": float((capture.writes[23].float() - rc.writes[23].float()).norm()),
                     "changed_kv_layers": [l for l in range(len(out.past_key_values.layers))
                         if not torch.equal(out.past_key_values.layers[l].values, rec.past_key_values.layers[l].values)
                         or not torch.equal(out.past_key_values.layers[l].keys, rec.past_key_values.layers[l].keys)],
                     "source_row": source, "wrong_source_row": wrong, "observation": row,
                     "meaning_status": "intervention on hypothesized role; semantics evaluated by outcomes"}
            history, tokens, continuation = list(rec_ids), [], []
            for step in range(3):
                token = int(out.logits[0, -1].argmax())
                history.append(token)
                tokens.append(token)
                if step == 2:
                    break
                past = out.past_key_values
                out, _, later = self.forward([token], past=past, label=pair["id"] + "-" + name + f"-future{step}")
                continuation.append({"consumed_token_id": token, "origin": "own-previous-output",
                                     "history_sha256": fingerprint(history), **later})
            first.update({"own_token_ids": tokens, "own_text": self.tokenizer.decode(tokens),
                          "continuation": continuation})
            results.append(first)
        return {"pair": pair, "recipient_native": rr, "donor_native": dr,
                "target_token_id": target, "alternative_token_id": alternative,
                "baseline_margin": baseline_margin, "arms": results,
                "interpretation": "head-output/source-contribution edits; source queries and K/V are not swapped"}


def object_followup(model, tokenizer, model_spec, weights_sha, root):
    from mdb_lanes.bench import BenchSession
    prompt = "Lena placed the green key in the box. Omar placed the silver coin in the bag. The object Omar placed is the"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    follow = tokenizer.encode(" The object Lena placed is the", add_special_tokens=False)
    bench = BenchSession.from_resident(model, tokenizer, root=root / "native-session",
        model_id=model_spec.model_id, revision=model_spec.revision, model_sha256=weights_sha,
        lane="virtual", attention_backend="native-eager", resident_kv_bytes=128 << 20,
        continuation_capacity=32, publication_mode="journal")
    try:
        bench.prefill(prompt_ids[:-1])
        plan = [{"kind": "input", "token_ids": prompt_ids[-1:]}, {"kind": "generate", "count": 2},
                {"kind": "input", "token_ids": follow}, {"kind": "generate", "count": 2}]
        total = 1 + 2 + len(follow) + 2
        result = bench.discover_circuit({"name": "two-receiver-object-binding", "plan": plan,
            "receiver_offsets": [0, 2 + len(follow)], "layers": [19, 22],
            "candidates_per_receiver": 1, "scales": [0.0, 0.5], "continuation_tokens": total,
            "top_k": 3, "max_native_operations": 10000, "max_capture_bytes": 256 << 20})
        save(root / "object-discovery.json", result["report"])
        return {"report": "object-discovery.json", "receiver_count": len(result["report"]["receivers"]),
                "test_count": len(result["report"]["tests"]),
                "semantic_scope": "supplied binding task, automatic bounded carrier nomination; no semantic label inferred"}
    finally:
        bench.close()


def main():
    import parameter_source
    started = time.monotonic()
    config = json.loads(Path("saturn_run_config.json").read_text())
    request = json.loads(Path("saturn_debug_request.json").read_text())
    job_id = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "unknown"))
    root = Path(config["output_root"]) / str(job_id)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = parameter_source._write_parameter_manifest(Path(config["model_path"]), root / "parameter-artifacts.json")
    spec, model, tokenizer, weights_sha, files, content_sha = parameter_source._load_model(config, manifest["provenance"])
    save(root / "request.json", request)
    assay = Assay(model, tokenizer, root)
    roles = scaffold(weights_sha)
    save(root / "frozen-schema.json", roles)
    occurrences, outcomes = [], []
    for case in config["cases"]:
        current, outcome = assay.observe_case(case)
        occurrences.extend(current)
        outcomes.append(outcome)
        print("SCAFFOLD_PROGRESS " + json.dumps({"case": case["id"], "generated": outcome["generated"], "calls": assay.calls}), flush=True)
    interventions = [assay.intervene(pair) for pair in config["pairs"]]
    graph = invocation_graph(roles, occurrences)
    save(root / "invocations.json", graph)
    render_html(graph, root / "scaffold.html")
    save(root / "interventions.json", interventions)
    save(root / "native-rows.json", assay.rows)
    object_result = object_followup(model, tokenizer, spec, weights_sha, root) if config["object_followup"] else None
    summary = {"schema": "mdb-live-scaffold-assay-v1", "job_id": job_id,
               "request_sha256": request["request_sha256"], "model_identity": weights_sha,
               "weight_files": files, "weight_content_sha256": content_sha,
               "scaffold_fingerprint": roles["fingerprint"], "cases": outcomes,
               "occurrences": len(occurrences), "graph_nodes": len(graph["nodes"]),
               "graph_edges": len(graph["edges"]), "physical_hf_forwards": assay.calls,
               "intervention_pairs": len(interventions), "object_followup": object_result,
               "worker_wall_s": time.monotonic() - started,
               "mechanics_status": "valid" if all(c["observer_control"]["logits_exact"] for c in outcomes) else "observer-parity-mismatch",
               "instrument_status": "bounded supplied role schema; task and clock coverage explicit",
               "trend_status": "inspect per-role read mass and signed per-arm effects",
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
    if len(raw) > 128 << 20:
        raise RuntimeError("archive exceeds declared transport bound")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i + (512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__":
    main()
