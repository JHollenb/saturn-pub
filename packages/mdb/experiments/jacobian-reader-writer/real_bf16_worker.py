"""Finite, frozen-model J-space qualification through the installed MDB wheel.

The fit is an analysis derivative, never a parameter update. All futures use
native Bench Acts and cuts; scientific effect sizes remain observations.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdb.bench import BenchSession
from mdb.bench_circuit_measurement import _memory, _memory_delta, _memory_summary
from mdb.bench_jacobian import _header_hash, _model_identity, application_profile
from mdb.bench_qwen import tensor_digest
from mdb.bench_verify import verify
from mdb.jacobian_io import FrozenJacobianLens
from mdb.jacobian_workspace import derive_dictionary_rows, metadata_sha256, tensor_payload_identity


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def normalize_assay_tokenizer(tokenizer):
    """Clear fit-call options before sealing the exact serving tokenizer state.

    Fast tokenizers persist transient truncation/padding in backend JSON. The
    manifest binds the assay state; fit token IDs and mask remain explicit.
    """
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()


def fit_prompt(model, ids, *, source_layer, final_layer, chunk, deadline):
    if ids.shape[1] < 18:
        raise ValueError("fit mask [16:-1] needs at least 18 tokens")
    capture = {}

    def source_hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden.detach().clone().requires_grad_(True)
        capture["source"] = leaf
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    def target_hook(_module, _args, output):
        capture["target"] = output[0] if isinstance(output, tuple) else output

    hooks = [model.model.layers[source_layer].register_forward_hook(source_hook),
             model.model.layers[final_layer].register_forward_hook(target_hook)]
    calls, rows = 0, []
    try:
        with torch.enable_grad():
            model.model(input_ids=ids.repeat(chunk, 1), use_cache=False, return_dict=True)
            source, target = capture["source"], capture["target"]
            width = target.shape[-1]
            for start in range(0, width, chunk):
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded fit deadline reached")
                stop = min(start + chunk, width)
                cotangent = torch.zeros_like(target)
                for lane, coordinate in enumerate(range(start, stop)):
                    cotangent[lane, 16:-1, coordinate] = 1
                grad = torch.autograd.grad(target, source, grad_outputs=cotangent,
                                           retain_graph=stop < width)[0]
                # Native BF16 derivative and position mean, then CPU FP32 pool.
                rows.append(grad[:stop-start, 16:-1].mean(1).detach().float().cpu())
                calls += 1
                if calls % 32 == 0:
                    print(f"FIT_CHUNKS {calls}/{(width+chunk-1)//chunk}", flush=True)
        matrix = torch.cat(rows)
        diagnostics = {"vjp_calls": calls, "vjp_lanes": width,
                       "position_mask": "[16:sequence_length-1)",
                       "source_activation": tensor_payload_identity(source),
                       "target_activation": tensor_payload_identity(target)}
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("nonfinite fitted transport")
        return matrix, diagnostics
    finally:
        for hook in hooks:
            hook.remove()


def lens_manifest(bench, matrix, protocol, fit_rows):
    profile = application_profile(bench)
    width = bench.model.config.hidden_size
    body = {
        "schema": "saturn-jacobian-lens-v1",
        "model_identity": {"tokenizer_identity_policy": "backend-json-v1"},
        "execution_compatibility": {"fitting_lane": profile["application_lane"],
                                    "permitted_application_lanes": [profile["application_lane"]]},
        "coordinate_map": {"residual_width": width, "source_layer": protocol["source_layer"],
                           "completed_layer_boundary": protocol["source_layer"] + 1,
                           "hook_convention": "decoder-block-output",
                           "source_position_semantics": "pooled fit positions [16:-1]; application uses absolute provisional token"},
        "transport": {"frozen": True, "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                      "sha256": tensor_payload_identity(matrix)["sha256"]},
        "decoder": {"residual_width": width, "vocab_size": bench.model.config.vocab_size,
                    "final_norm": "model.norm", "output_head": "lm_head",
                    "final_norm_weight_sha256": _header_hash(bench.model.model.norm.weight),
                    "unembedding_weight_sha256": _header_hash(bench.model.lm_head.weight)},
        "estimator": {"name": "target-sum/source-mean", "position_mask": "[16:-1]",
                      "target_layer": protocol["final_layer"],
                      "target_completed_layer_boundary": protocol["final_layer"]+1,
                      "target_hook_convention": "decoder-block-output-before-final-norm",
                      "fit_rows": fit_rows, "graph_dtype": "torch.bfloat16",
                      "pool_dtype": "torch.float32", "payload_cast": "torch.bfloat16",
                      "parameter_updates": 0},
        "dictionary": {"atom_derivation": "decoder-weight-times-transport",
                       "norm_scale_convention": "unit-l2-after-transport",
                       "sparse_interpretation": "unperformed"},
    }
    body["model_identity"] = _model_identity(bench, body)
    return {**body, "fingerprint": metadata_sha256(body)}


def native_readout(bench, recipient, donor):
    logits = bench.resolve_value("logits")
    flat = logits.reshape(-1, logits.shape[-1])[-1].float()
    ids = flat.topk(5).indices.tolist()
    return {"logits_sha256": tensor_digest(logits),
            "margin": float(flat[donor] - flat[recipient]),
            "pending_token": bench.branch.pending,
            "top_tokens": [{"id": token, "text": bench.tokenizer.decode([token]),
                            "logit": float(flat[token])} for token in ids]}


def consume(bench, query_ids, recipient, donor, deadline):
    trace = []
    for token in query_ids:
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded assay deadline reached")
        bench.stage_input(token)
        bench.step("token")
        trace.append(native_readout(bench, recipient, donor))
    first = trace[-1]
    answers = []
    for _ in range(2):
        answers.append(bench.branch.pending)
        bench.step("token")
        trace.append(native_readout(bench, recipient, donor))
    return {"first_answer": first, "trace": trace, "answer_token_ids": answers,
            "answer_text": bench.tokenizer.decode(answers),
            "endpoint_cut": bench.branch.cut,
            "endpoint_memory": _memory_summary(_memory(bench))["sha256"]}


def norm_candidate(source, direction, target_norm, sign):
    # Iteratively target the ACTUAL native BF16 delta norm; retain residual error.
    vector = direction.float()
    if float(vector.norm()) == 0 or target_norm == 0:
        return source.clone()
    vector = vector / vector.norm()
    scale = target_norm
    for _ in range(8):
        candidate = (source.float() + sign * scale * vector).to(source.dtype)
        actual = float((candidate.float() - source.float()).norm())
        if actual == 0:
            scale *= 2
        else:
            scale *= target_norm / actual
    return (source.float() + sign * scale * vector).to(source.dtype)


def run_row(bench, pin, matrix, protocol, row, deadline):
    tokenizer = bench.tokenizer
    token = lambda text: tokenizer.encode(text, add_special_tokens=False)
    rid, did = token(row["recipient_token"]), token(row["donor_token"])
    if len(rid) != 1 or len(did) != 1:
        raise ValueError("contrast must use single vocabulary tokens")
    recipient, donor = rid[0], did[0]
    prefixes = {role: token(row[role + "_prefix"]) for role in ("recipient", "donor")}
    if prefixes["recipient"][-1] != recipient or prefixes["donor"][-1] != donor:
        raise ValueError("source token does not match prefix-final animal")
    if prefixes["recipient"][:-1] != prefixes["donor"][:-1]:
        raise ValueError("paired animal prefixes must share exact parent token history")
    query_ids = token(row["query"])
    boundary = protocol["source_layer"] + 1
    for role in ("donor", "recipient"):
        ids = prefixes[role]
        bench.prefill(ids[:-1], branch=row["id"] + "-" + role)
        bench.stage_input(ids[-1])
        bench.step("layer", count=boundary)
        if role == "donor":
            donor_source = bench.resolve_value("residual")
    source = bench.resolve_value("residual")
    address = {"symbol": "memory.animal", "instance": row["id"], "role": "remembered",
               "spectral_join": "host-supplied opaque address; semantic binding unqualified"}
    read = bench.read_jacobian(pin["lens_ref"], token_subset=[recipient, donor], top_k=2,
                               logical_address=address)
    receipt = bench.artifacts.get(read["observation_ref"])
    transported = bench._tensor_load(receipt["readout_refs"]["transported"])
    selected = bench._tensor_load(receipt["readout_refs"]["selected_normalized_scores"])
    selected_ids = bench._tensor_load(receipt["readout_refs"]["selected_token_ids"])
    with torch.inference_mode():
        scores = bench.model.lm_head(bench.model.model.norm(transported))
    reader_exact = torch.equal(selected, scores.gather(-1, selected_ids))
    if not reader_exact:
        raise ValueError("native transported readout parity failed")
    atoms = derive_dictionary_rows(matrix.to(source.device), bench.model.lm_head.weight.detach(),
                                    [recipient, donor], normalize=True).rows.float()
    target = torch.linalg.lstsq(atoms.T, donor_source.reshape(-1).float().unsqueeze(1)).solution[:, 0]
    recipe = {"operation": "span", "token_ids": [recipient, donor],
              "target_coordinates": target.tolist(), "dose": 1,
              "consumer": {"name": "native-animal-recall", "horizon": len(query_ids)+2},
              "logical_address": address}
    # Preview once to calibrate native-cast control amplitudes, then abort exactly.
    preview = bench.preview_jacobian_edit(read["observation_ref"], recipe)
    candidate = bench._tensor_load(bench.artifacts.get(preview["jacobian_candidate_ref"])["candidate_ref"])
    delta = candidate.float() - source.float()
    j_norm = float(delta.norm())
    bench.abort_edit(preview["act_id"])
    abort_exact = torch.equal(source, bench.resolve_value("residual"))
    if not abort_exact:
        raise ValueError("preview abort changed parent residual")
    raw = bench.model.lm_head.weight.detach()[[recipient, donor]].float()
    raw = raw / raw.norm(dim=1, keepdim=True)
    difference = (donor_source.float()-source.float()).reshape(-1)
    raw_delta = (raw.T @ torch.linalg.lstsq(raw.T, difference.unsqueeze(1)).solution[:, 0]).reshape_as(source)
    generator = torch.Generator(device="cpu").manual_seed(protocol["seed"])
    random_delta = torch.randn(source.shape, generator=generator).to(source.device)
    arms = [("native", None), ("j-zero", 0), ("j-plus", 1), ("j-minus", -1), ("j-half", .5),
            ("raw-plus", norm_candidate(source, raw_delta, j_norm, 1)),
            ("raw-minus", norm_candidate(source, raw_delta, j_norm, -1)),
            ("random-plus", norm_candidate(source, random_delta, j_norm, 1)),
            ("random-minus", norm_candidate(source, random_delta, j_norm, -1)),
            ("full-donor", donor_source)]
    results, formed_memory = {}, {}
    for name, payload in arms:
        bench.restore(read["source_cut"])
        if not torch.equal(bench.resolve_value("residual"), source):
            raise ValueError("same-parent restore changed residual")
        act = None
        actual_candidate = source
        if isinstance(payload, (int, float)):
            act = bench.preview_jacobian_edit(read["observation_ref"], {**recipe, "dose": payload})
            actual_candidate = bench._tensor_load(bench.artifacts.get(act["jacobian_candidate_ref"])["candidate_ref"])
        elif isinstance(payload, torch.Tensor):
            actual_candidate = payload
            act = bench.preview_edit("residual", payload, mode="replace")
        if act:
            bench.commit_edit(act["act_id"])
            if not torch.equal(bench.resolve_value("residual"), actual_candidate):
                raise ValueError("committed native residual differs from prepared bytes")
        bench.step("token")
        carrier = bench.branch.cut
        immediate = native_readout(bench, recipient, donor)
        memory = _memory(bench)
        formed_memory[name] = memory
        future = consume(bench, query_ids, recipient, donor, deadline)
        results[name] = {"act_id": act["act_id"] if act else None,
                         "candidate_ref": act.get("jacobian_candidate_ref") if act else None,
                         "actual_delta_l2": float((actual_candidate.float()-source.float()).norm()),
                         "carrier_cut": carrier, "carrier_memory": _memory_summary(memory)["sha256"],
                         "immediate": immediate, **future}
        print(f"ARM {row['id']} {name} margin={future['first_answer']['margin']:.6f} answer={future['answer_text']!r}", flush=True)
    native = results["native"]
    for name, result in results.items():
        result["immediate_effect"] = result["immediate"]["margin"]-native["immediate"]["margin"]
        result["delayed_effect"] = result["first_answer"]["margin"]-native["first_answer"]["margin"]
        result["carrier_delta"] = _memory_delta(formed_memory[name], formed_memory["native"])
        result["control_norm_relative_error"] = (result["actual_delta_l2"]/j_norm-1) if j_norm else None
    zero = results["j-zero"]
    zero_exact = (zero["immediate"]["logits_sha256"] == native["immediate"]["logits_sha256"]
                  and zero["carrier_memory"] == native["carrier_memory"]
                  and [v["logits_sha256"] for v in zero["trace"]] == [v["logits_sha256"] for v in native["trace"]]
                  and zero["endpoint_memory"] == native["endpoint_memory"]
                  and zero["answer_token_ids"] == native["answer_token_ids"])
    if not zero_exact:
        raise ValueError("zero-dose native future differs from unedited parent")
    probes = {}
    position = len(prefixes["recipient"])-1
    selector = {"layers": list(range(max(0, protocol["final_layer"]-5), protocol["final_layer"]+1)),
                "row_start": position, "row_stop": position+1, "components": ["key", "value"]}
    for name in ("j-plus", "full-donor"):
        bench.restore(native["carrier_cut"])
        probe = bench.memory_probe({"name": row["id"] + "/" + name,
                                   "intervention": {"mode": "copy", "donor_cut": results[name]["carrier_cut"], "selector": selector},
                                   "query_token_ids": query_ids, "continuation_tokens": 2,
                                   "observable": {"kind": "token_margin", "token_id": donor, "against_token_id": recipient},
                                   "max_native_operations": 4096, "max_capture_bytes": 256 << 20})
        probes[name] = probe
        print(f"MEMORY_PROBE {row['id']} {name}", flush=True)
    return {"id": row["id"], "split": "opened-donor-assisted-exploration", "source_cut": read["source_cut"],
            "observation_ref": read["observation_ref"], "token_ids": {"recipient": recipient, "donor": donor},
            "query_token_ids": query_ids, "reader_native_exact": reader_exact,
            "transported_differs_from_source": not torch.equal(transported, source),
            "preview_abort_exact": abort_exact, "zero_native_future_exact": zero_exact,
            "j_projection_norm": j_norm, "full_donor_norm": float(difference.norm()),
            "selected_two_atom_squared_coverage": j_norm**2/float(difference.norm())**2,
            "arms": results, "memory_probes": probes}


def emit_bundle(output):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(output.parent)), recursive=False)
    data = stream.getvalue()
    print("MDB_JSPACE_BF16_BUNDLE_SHA256 " + hashlib.sha256(data).hexdigest(), flush=True)
    encoded = base64.b64encode(data).decode()
    for index, start in enumerate(range(0, len(encoded), 90000)):
        print(f"MDB_JSPACE_BF16_BUNDLE_CHUNK {index} {encoded[start:start+90000]}", flush=True)
    print("MDB_JSPACE_BF16_BUNDLE_CHUNKS " + str((len(encoded)+89999)//90000), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + protocol["execution"]["max_wall_s"]
    expected_vjps = len(protocol["fit_texts"]) * ((1536+protocol["vjp_chunk"]-1)//protocol["vjp_chunk"])
    if expected_vjps > protocol["execution"]["max_backward_invocations"]:
        raise ValueError("planned derivative operation budget exceeded")
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = Path(protocol["model_path"]) / "model.safetensors"
    if file_sha(weights) != protocol["model_safetensors_sha256"]:
        raise ValueError("model file identity mismatch")
    assets = {path.name: file_sha(path) for path in Path(protocol["model_path"]).iterdir()
              if path.is_file() and path.suffix in {".json", ".txt"}}
    tokenizer = AutoTokenizer.from_pretrained(protocol["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(protocol["model_path"], local_files_only=True,
                                                torch_dtype=torch.bfloat16, attn_implementation="eager").eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.config.hidden_size != 1536 or model.config.num_hidden_layers != 28:
        raise ValueError("declared specimen geometry mismatch")
    matrices, fit_rows = [], []
    for index, text in enumerate(protocol["fit_texts"]):
        ids = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=protocol["max_length"], return_tensors="pt").input_ids.to("cuda")
        tick = time.monotonic()
        matrix, diagnostics = fit_prompt(model, ids, source_layer=protocol["source_layer"],
                                          final_layer=protocol["final_layer"], chunk=protocol["vjp_chunk"], deadline=deadline)
        matrices.append(matrix)
        fit_rows.append({"index": index, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "token_ids": ids[0].tolist(), "elapsed_s": time.monotonic()-tick, **diagnostics})
        print(f"FIT_ROW {index} complete", flush=True)
    matrix = torch.stack(matrices).mean(0).to(torch.bfloat16).contiguous()
    if sum(row["vjp_calls"] for row in fit_rows) > protocol["execution"]["max_backward_invocations"]:
        raise ValueError("declared derivative operation budget exceeded")
    if torch.equal(matrix, torch.eye(matrix.shape[0], dtype=matrix.dtype)):
        raise ValueError("fitted transport is an identity stand-in")
    normalize_assay_tokenizer(tokenizer)
    bench = BenchSession.from_resident(model, tokenizer, root=output/"bench", model_id=protocol["model_id"],
                                       revision=protocol["revision"], model_sha256=protocol["model_safetensors_sha256"],
                                       session_id="jspace-real-bf16", lane="native", attention_backend="native-eager",
                                       continuation_capacity=64, resident_kv_bytes=64 << 20)
    bench.prefill([tokenizer.eos_token_id], branch="lens-pin")
    # Count native calls even when they originate inside memory_probe.
    original_step = bench.step
    counts = {"token_steps": 0, "explicit_layer_steps": 0}

    def bounded_step(unit, **kwargs):
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded native deadline reached")
        if unit == "token":
            counts["token_steps"] += kwargs.get("count", 1)
            if counts["token_steps"] > protocol["execution"]["max_assay_token_steps"]:
                raise ValueError("native token operation budget exceeded")
        elif unit == "layer":
            counts["explicit_layer_steps"] += kwargs.get("count", 1)
        return original_step(unit, **kwargs)

    bench.step = bounded_step
    manifest = lens_manifest(bench, matrix, protocol, fit_rows)
    FrozenJacobianLens(manifest, matrix).save(output/"lens")
    pin = bench.pin_jacobian_lens(manifest, matrix)
    rows = [run_row(bench, pin, matrix, protocol, row, deadline) for row in protocol["assay_rows"]]
    if time.monotonic() >= deadline-30:
        raise TimeoutError("insufficient declared time for export finalization")
    bench.export(output/"bench-export")
    profile = application_profile(bench)
    bench.close()
    reopened = BenchSession.resume(model, tokenizer, root=output/"bench-export",
                                    model_sha256=protocol["model_safetensors_sha256"])
    original_step = reopened.step
    reopened.step = bounded_step
    last = rows[-1]
    reopened.restore(last["source_cut"])
    noop = reopened.preview_jacobian_edit(last["observation_ref"], {
        "operation": "span", "token_ids": list(last["token_ids"].values()),
        "target_coordinates": [0, 0], "dose": 0,
        "consumer": {"name": "export-reopen-native-recall", "horizon": len(last["query_token_ids"])+2}})
    reopened.commit_edit(noop["act_id"])
    reopened.step("token")
    reread = native_readout(reopened, last["token_ids"]["recipient"], last["token_ids"]["donor"])
    replay = consume(reopened, last["query_token_ids"], last["token_ids"]["recipient"], last["token_ids"]["donor"], deadline)
    original = last["arms"]["native"]
    reopen_exact = (reread["logits_sha256"] == original["immediate"]["logits_sha256"]
                    and [v["logits_sha256"] for v in replay["trace"]] == [v["logits_sha256"] for v in original["trace"]]
                    and replay["endpoint_memory"] == original["endpoint_memory"])
    if not reopen_exact:
        raise ValueError("real exported source-bound zero-write replay differs from native")
    reopened.close()
    audit = verify(output/"bench-export")
    # The exported CAS is authoritative; do not bundle the duplicate live store.
    import shutil
    shutil.rmtree(output/"bench")
    report = {"schema": "mdb-jspace-real-bf16-proof-v1", "protocol_sha256": file_sha(args.protocol),
              "worker_sha256": file_sha(__file__), "wheel_sha256": protocol["wheel_sha256"],
              "mechanics_status": "valid", "instrument_status": "valid",
              "trend_status": "signed-observations-retained", "terminal_status": "bounded-native-reader-writer-mechanics-qualified",
              "terminal_claim": {"scope": "software mechanics for this recorded model/lens/lane/cuts; no effect-size or semantic utility claim",
                                 "criteria": ["non-identity genuinely fitted finite transport", "native transported-readout bit parity",
                                              "source-bound native candidate commits", "exact abort and zero-dose native future",
                                              "native K/V reinstall closure", "export/new-session zero-write exact replay", "final closed-export byte audit"]},
              "semantic_writer_qualified": False, "parameter_updates": 0, "model_loads": 1,
              "execution": {"device": "cuda", "dtype": str(next(model.parameters()).dtype),
                            "attention": "native-eager", "torch": torch.__version__, "transformers": transformers.__version__,
                            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "elapsed_s": time.monotonic()-started,
                            "backward_invocations": sum(v["vjp_calls"] for v in fit_rows), **counts},
              "profile": profile, "lens_fingerprint": manifest["fingerprint"],
              "export_reopen": {"same_process_new_session": True, "native_future_exact": reopen_exact,
                                "source_cut": last["source_cut"], "observation_ref": last["observation_ref"],
                                "zero_write_candidate_ref": noop["jacobian_candidate_ref"]},
              "fit_rows": fit_rows, "rows": rows, "export_audit": audit,
              "resident_assets_sha256": assets,
              "limitations": ["pooled full-sequence BF16 fit applied to cached one-token native execution",
                              "opened two-row donor-assisted assay; no held-out semantic-generalization claim",
                              "two selected vocabulary atoms; no sparse full-dictionary reconstruction",
                              "logical/spectral address metadata is supplied by the host",
                              "config/tokenizer file identities are recorded at execution; only weight-file digest was declared before this run"]}
    save_json(output/"protocol.json", protocol)
    save_json(output/"report.json", report)
    print("MDB_JSPACE_BF16_REPORT_BASE64 " + base64.b64encode(json.dumps(report, sort_keys=True, allow_nan=False).encode()).decode(), flush=True)
    if time.monotonic() >= deadline:
        raise TimeoutError("declared worker deadline reached before bundle finalization")
    emit_bundle(output)


if __name__ == "__main__":
    main()
