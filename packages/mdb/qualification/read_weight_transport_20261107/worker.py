"""Baseline-compiled read weights, sealed before edited downstream receivers."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import composition_support as previous
import torch
import torch.nn.functional as functional

from mdb.bench import BenchSession
from mdb.bench_circuit_discovery import _head_edit
from mdb.bench_circuit_measurement import _closure, _memory
from mdb.bench_hidden_memory import logical_address
from mdb.read_program import FrozenReadProgram
from mdb.read_weight_transport import (
    SmoothQwenReadTransport,
    compile_score_program,
    evaluate_score_program,
)

DOSES = (0.0, 0.25, 0.5, 1.0)
METHODS = ("transport", "linear-logweight", "stale", "wrong-sign", "query-only")
CONSUMERS = ("transport", "linear-logweight", "stale", "oracle")
save, error = previous.save, previous.error


class Capture(previous.Capture):
    @contextlib.contextmanager
    def installed(self):
        handles = []
        runtime = self.bench.vm.runtime
        native_advance = runtime.advance_token

        def boundary(execution):
            index = execution.next_layer
            observed = self.active and index in (19, 20, 21, 22)
            if observed:
                self.keep(f"input{index}", execution.hidden)
            result = native_advance(execution)
            if observed:
                self.keep(f"exit{index}", execution.hidden)
            return result

        runtime.advance_token = boundary
        try:
            for index in (19, 20, 21, 22):
                layer = self.bench.model.model.layers[index]

                def query(_module, _args, output, index=index):
                    if self.active:
                        self.live[f"rawq{index}"] = output.detach().clone()

                def attention(_module, args, output, index=index):
                    if self.active:
                        self.keep(f"projection_input{index}", args[0])
                        self.observe(index, output)

                def mlp(_module, _args, output, index=index):
                    if self.active:
                        self.keep(f"mlp{index}", output)

                handles.extend([layer.self_attn.q_proj.register_forward_hook(query),
                                layer.self_attn.o_proj.register_forward_hook(attention),
                                layer.mlp.register_forward_hook(mlp)])
            yield self
        finally:
            runtime.advance_token = native_advance
            for handle in handles:
                handle.remove()

    def observe(self, index, output):
        from transformers.models.qwen2 import modeling_qwen2 as modeling

        bench = self.bench
        execution = bench._backend().execution
        state = execution.state
        module = bench.model.model.layers[index].self_attn
        heads = bench.model.config.num_attention_heads
        groups, width = bench.vm.runtime.codec.groups, module.head_dim
        q = self.live.pop(f"rawq{index}").view(1, 1, heads, width).transpose(1, 2).contiguous()
        position = torch.tensor([[state.length]], device=q.device)
        rope = bench.model.model.rotary_emb(execution.hidden, position)
        q, _ = modeling.apply_rotary_pos_emb(q, q, *rope)
        q = q.contiguous()
        k = torch.cat((state.ancestor[index][0], state.delta[index][0][:, :, :state.delta_rows + 1]), dim=2)
        v = torch.cat((state.ancestor[index][1], state.delta[index][1][:, :, :state.delta_rows + 1]), dim=2)
        scores = q @ modeling.repeat_kv(k, heads // groups).transpose(2, 3) * module.scaling
        probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        for name, value in (("q", q), ("k", k), ("v", v), ("scores", scores),
                            ("weights", probability), ("attention", output),
                            ("current_v", v[:, :, -1:]), ("cos", rope[0]), ("sin", rope[1])):
            self.keep(f"{name}{index}", value)


class Runner(previous.Runner):
    def arm(self, name, parent, sites, last_token, target_ids, *, dose=1.0,
            edits=(), injection=None, capture=True, count=3):
        bench = self.bench
        bench.restore(parent)
        acts = []
        for port in edits:
            act, _, locality = _head_edit(bench, sites[port], 1.0 - dose if port == "A" else 0.0)
            acts.append({"port": port, "act": act, "locality": locality})
        if name == "reinstall":
            for item in acts:
                site = sites[item["port"]]
                row = bench._tensor_load(item["locality"]["before_ref"])
                bench.memory_write({"mode": "replace", "selector": {
                    "layers": [site["layer"]], "components": ["value"],
                    "row_start": site["source_row"], "row_stop": site["source_row"] + 1},
                    "values": {logical_address(site["layer"], "value", site["source_row"]): row.float().tolist()}})
        observer = Capture(bench, sites)
        trajectory = []
        with contextlib.ExitStack() as stack:
            if injection is not None:
                def inject(_module, _args, output):
                    return injection.to(output) if observer.active else output
                handle = bench.model.model.layers[22].self_attn.o_proj.register_forward_hook(inject)
                stack.callback(handle.remove)
            if capture:
                stack.enter_context(observer.installed())
            for offset in range(count):
                result = self.step(last_token if offset == 0 else None)
                logits = bench.resolve_value("logits").float()[0]
                trajectory.append({"offset": offset, "authority": "caller-supplied" if offset == 0 else "native-greedy",
                    "consumed_token_id": result["token_id"], "next_token_id": result["next_token_id"],
                    "next_text": self.tokenizer.decode([result["next_token_id"]]),
                    "cut": result["cut"], "receipt_ref": result["receipt_ref"],
                    "logits_sha256": result["logits_sha256"],
                    "target_margin": float(logits[target_ids[0]] - logits[target_ids[1]]),
                    "direct_object_token_match": result["next_token_id"] == target_ids[0] if offset == 0 else None})
                observer.active = False
        return {"name": name, "parent_cut": parent, "dose": dose, "acts": acts,
                "trajectory": trajectory, "capture": observer.refs, "end_cut": bench.branch.cut,
                "closure_sha256": _closure(bench, _memory(bench))["sha256"]}, observer.live


@contextlib.contextmanager
def forbid_native(model):
    """The forecast may read parameters, but no original module may execute."""
    entries = {"count": 0}
    handles = []

    def refuse(module, _args):
        entries["count"] += 1
        raise RuntimeError(f"native module entered during forecast: {type(module).__name__}")

    try:
        for module in model.modules():
            handles.append(module.register_forward_pre_hook(refuse))
        yield entries
    finally:
        for handle in handles:
            handle.remove()


def refs(bench, value):
    if isinstance(value, torch.Tensor):
        return bench._tensor_ref(value)
    if isinstance(value, dict):
        return {str(key): refs(bench, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [refs(bench, item) for item in value]
    return value


def local_source_attention(model, tape, site, dose):
    """Form only the preceding L19 write; never advance its receiving MLP."""
    heads, kv = model.config.num_attention_heads, model.config.num_key_value_heads
    values = tape["v19"].clone()
    values[:, site["kv_head"], site["source_row"]] *= 1.0 - dose
    expanded = values.repeat_interleave(heads // kv, dim=1)
    attended = (tape["weights19"] @ expanded).transpose(1, 2).reshape(1, 1, -1).contiguous()
    projection = model.model.layers[19].self_attn.o_proj
    return functional.linear(attended, projection.weight, projection.bias)


def weight_error(prediction, true_logp, baseline_logp):
    delta = true_logp - baseline_logp
    residual = prediction["delta_logpB"].float() - delta
    return {"log_change_rmse": float(residual.square().mean().sqrt()),
            "log_change_max_abs": float(residual.abs().max()),
            "true_log_change_rms": float(delta.square().mean().sqrt()),
            "probability_error": error(prediction["pB"].reshape(-1), true_logp.exp().reshape(-1)),
            "predicted_probability_in_range": bool(((prediction["pB"] >= 0) & (prediction["pB"] <= 1)).all()),
            "head_rows": [{"head_offset": index, "true_delta_logp": float(actual),
                           "predicted_delta_logp": float(estimated), "error": float(estimated - actual)}
                          for index, (actual, estimated) in enumerate(zip(delta.flatten(), prediction["delta_logpB"].flatten(), strict=True))]}


@torch.inference_mode()
def bundle(runner, parent, sites, last_token, target_ids):
    bench, model = runner.bench, runner.bench.model
    arms, live = {}, {}
    for name, edits, capture in (("00", (), True), ("sham", (), False), ("reinstall", ("A", "B"), False)):
        arms[name], live[name] = runner.arm(name, parent, sites, last_token, target_ids,
                                          edits=edits, capture=capture)
    for control in ("sham", "reinstall"):
        if (previous.signature(arms[control]) != previous.signature(arms["00"])
                or arms[control]["closure_sha256"] != arms["00"]["closure_sha256"]):
            raise RuntimeError(f"{control} failed native closure")
    zero = live["00"]
    if zero["q22"].is_cuda:
        torch.cuda.synchronize()
    producer_start = time.perf_counter()
    with forbid_native(model) as guards:
        sources = {dose: local_source_attention(model, zero, sites["A"], dose) for dose in DOSES}
        source_replay = error(sources[0.0], zero["attention19"])
        if not source_replay["exact"]:
            raise RuntimeError("local preceding-write builder failed baseline native parity")
        write = sources[1.0].float() - zero["attention19"].float()
        with torch.inference_mode(False), torch.enable_grad():
            compiler = SmoothQwenReadTransport(model, incoming=zero["input19"], baseline_attention=zero["attention19"],
                history={i: (zero[f"k{i}"][:, :, :-1], zero[f"v{i}"][:, :, :-1]) for i in (20, 21)},
                cos=zero["cos22"], sin=zero["sin22"])
            compiled = compiler.compile_direction(write)
        score_program = compile_score_program(zero["q22"], zero["k22"], zero["scores22"],
            compiled["q_dot"], compiled["k_dot"], kv_head=sites["B"]["kv_head"],
            source_row=sites["B"]["source_row"], scaling=model.model.layers[22].self_attn.scaling)
        if zero["q22"].is_cuda:
            torch.cuda.synchronize()
        evaluation_start = time.perf_counter()
        forecasts = {str(dose): {method: evaluate_score_program(score_program, dose, method=method)
                                for method in METHODS} for dose in DOSES}
        if zero["q22"].is_cuda:
            torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - evaluation_start
    producer_seconds = time.perf_counter() - producer_start
    weight_record = {"schema": "mdb-read-weight-forecast-v1", "parent_cut": parent, "sites": sites,
        "visibility": "native00 tape, model parameters, local L19 write only; no edited receiver or answer labels",
        "baseline_capture": arms["00"]["capture"], "source_replay": source_replay,
        "source_attention": refs(bench, sources), "write_ref": bench._tensor_ref(write),
        "compiled": refs(bench, compiled), "score_program": refs(bench, score_program),
        "forecasts": refs(bench, forecasts),
        "native_module_entries": guards["count"], "physical_steps_at_seal": runner.token_steps,
        "producer_wall_seconds": producer_seconds, "twenty_scalar_evaluations_wall_seconds": evaluation_seconds,
        "serving_tensor_elements": sum(v.numel() for v in score_program.values() if isinstance(v, torch.Tensor)),
        "serving_tensor_bytes": sum(v.numel() * v.element_size() for v in score_program.values() if isinstance(v, torch.Tensor)),
        "cost_scope": "per-receiver dense FP32 compilation plus 4 local source O projections; dose evaluator uses scalar coefficients"}
    weight_ref = bench.artifacts.put(weight_record)
    runner.events.append({"kind": "weights-sealed", "ref": weight_ref, "physical_steps": runner.token_steps})
    # The complete edited FP32 lift is a paid diagnostic, evaluated only after
    # the small-program predictions are immutable. It is not their input.
    with forbid_native(model) as dense_guards, torch.inference_mode(False), torch.enable_grad():
        dense_control = compiler.shadow((zero["attention19"].float() + write).clone())
    dense_control_refs = refs(bench, dense_control)
    for dose in DOSES[1:]:
        name = f"A-{dose}"
        runner.events.append({"kind": "edited-receiver-started", "dose": dose, "weight_ref": weight_ref,
                              "physical_steps": runner.token_steps})
        arms[name], live[name] = runner.arm(name, parent, sites, last_token, target_ids,
            dose=dose, edits=("A",), count=3 if dose == 1.0 else 1)
    a = live["A-1.0"]
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    first_head = sites["B"]["kv_head"] * groups
    selected = slice(first_head, first_head + groups)
    row = sites["B"]["source_row"]
    value = zero["v22"][0, sites["B"]["kv_head"], row]
    projection = model.model.layers[22].self_attn.o_proj
    width = model.model.layers[22].self_attn.head_dim
    vectors = torch.stack([projection.weight[:, h * width:(h + 1) * width].float() @ value.float()
                           for h in range(first_head, first_head + groups)])
    consumer_tensors = {}
    consumer_records = {}
    for method in CONSUMERS:
        weights = (a["weights22"][0, selected, 0, row].float() if method == "oracle"
                   else forecasts["1.0"][method]["pB"].reshape(-1).float().to(a["weights22"].dtype).float())
        attention = a["attention22"].float() - (weights[:, None] * vectors).sum(0)
        consumed = attention.to(a["attention22"].dtype)
        mlp, exit_state = previous.predict_exit(model.model.layers[22], a["input22"], consumed)
        consumer_tensors[method] = {"attention": attention, "consumed_attention": consumed, "mlp": mlp, "exit": exit_state}
        consumer_records[method] = {"weight_forecast_ref": weight_ref, "native_A_output_assisted": True,
            "baseline_attention_ref": arms["A-1.0"]["capture"]["attention22"], "weights_ref": bench._tensor_ref(weights),
            **refs(bench, consumer_tensors[method]), "explicit_dense_mlp_linears": 3}
    consumer_record = {"schema": "mdb-read-weight-consumer-forecast-v1", "parent_cut": parent,
                       "weight_forecast_ref": weight_ref, "methods": consumer_records,
                       "physical_steps_at_seal": runner.token_steps}
    consumer_ref = bench.artifacts.put(consumer_record)
    runner.events.append({"kind": "consumers-sealed", "ref": consumer_ref, "physical_steps": runner.token_steps})
    for method in CONSUMERS:
        arms[method], live[method] = runner.arm(method, parent, sites, last_token, target_ids,
            edits=("A", "B"), injection=consumer_tensors[method]["consumed_attention"])
    runner.events.append({"kind": "native-AB-started", "consumer_ref": consumer_ref, "physical_steps": runner.token_steps})
    arms["AB"], live["AB"] = runner.arm("AB", parent, sites, last_token, target_ids, edits=("A", "B"))
    ab = live["AB"]
    if not all(torch.equal(a[key], ab[key]) for key in ("input22", "q22", "k22", "current_v22")):
        raise RuntimeError("B edit violated fixed-receiver value-only contract")
    baseline_logp = torch.log_softmax(zero["scores22"].float(), -1)[0, selected, 0, row]
    dose_rows = []
    for dose in DOSES:
        native = zero if dose == 0 else live[f"A-{dose}"]
        if not torch.equal(native["q19"], zero["q19"]) or not torch.equal(native["k19"], zero["k19"]):
            raise RuntimeError("A edit changed its initial query or keys")
        true_logp = torch.log_softmax(native["scores22"].float(), -1)[0, selected, 0, row]
        dose_rows.append({"dose": dose, "native_capture": arms["00" if dose == 0 else f"A-{dose}"]["capture"],
            "earlier_write_error": error(sources[dose], native["attention19"]),
            "scaled_source_write_error": error(zero["attention19"].float() + dose * write, native["attention19"]),
            "methods": {method: weight_error(forecasts[str(dose)][method], true_logp, baseline_logp) for method in METHODS}})
    local = {}
    for method in CONSUMERS:
        local[method] = {key: error(consumer_tensors[method][key], ab[native_key]) for key, native_key in (
            ("attention", "attention22"), ("mlp", "mlp22"), ("exit", "exit22"))}
        local[method]["injected_mlp_exact"] = torch.equal(consumer_tensors[method]["mlp"], live[method]["mlp22"])
        local[method]["injected_exit_exact"] = torch.equal(consumer_tensors[method]["exit"], live[method]["exit22"])
    result = {"parent_cut": parent, "sites": sites, "semantic_target_ids": target_ids, "arms": arms,
        "weight_forecast_ref": weight_ref, "weight_forecast": weight_record,
        "consumer_forecast_ref": consumer_ref, "consumer_forecast": consumer_record,
        "dose_rows": dose_rows, "local_errors": local,
        "future_comparisons": {method: previous.compare_arms(bench, arms[method], arms["AB"]) for method in CONSUMERS},
        "q_transport_error": error(zero["q22"].float() + compiled["q_dot"], a["q22"]),
        "k_transport_error": error(zero["k22"][:, :, -1:].float() + compiled["k_dot"], a["k22"][:, :, -1:]),
        "dense_fp32_control_qk": dense_control_refs,
        "dense_fp32_control_errors": {"q": error(dense_control[0], a["q22"]), "k": error(dense_control[1], a["k22"][:, :, -1:])},
        "dense_control_native_module_entries": dense_guards["count"],
        "old_keys22_unchanged": torch.equal(zero["k22"][:, :, :-1], a["k22"][:, :, :-1]),
        "B_source_value_unchanged": torch.equal(zero["v22"][0, sites["B"]["kv_head"], row], a["v22"][0, sites["B"]["kv_head"], row]),
        "native_module_entries_during_forecast": guards["count"], "mechanics_status": "valid",
        "terminal_status": "not-assessed", "economical_full_query_qualified": False}
    bench.restore(parent)
    return result


def fresh_cases():
    rows = []
    for a, b, x, y in (("Mira", "Daniel", "key", "coin"), ("Zoe", "Liam", "book", "cup")):
        for previous_owner in (a, b):
            rows.append({"id": f"{a.lower()}-{previous_owner.lower()}", "owners": [a, b], "objects": [x, y],
                "previous_owner": previous_owner,
                "prompt": f"{a} placed the {x} in the box. {b} placed the {y} in the bag. The object {previous_owner} placed is the",
                "followups": [f" The object {a} placed is the", f" The object {b} placed is the"],
                "previous_current_owner_relation": "same" if previous_owner == a else "other"})
    return rows


@torch.inference_mode()
def run_case(model, tokenizer, spec, weights_sha, case, root, budget):
    ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
    copula = tokenizer.encode(" is", add_special_tokens=False)
    object_ids = [tokenizer.encode(" " + word, add_special_tokens=False) for word in case["objects"]]
    if len(copula) != 1 or any(len(tokens) != 1 for tokens in object_ids):
        raise RuntimeError("declared single-token locator or object contrast inapplicable")
    targets = [tokens[0] for tokens in object_ids]
    program = FrozenReadProgram.standard()
    bench = BenchSession.from_resident(model, tokenizer, root=root / case["id"], model_id=spec.model_id,
        revision=spec.revision, model_sha256=weights_sha, session_id=case["id"],
        attention_backend="native-eager", continuation_capacity=128)
    runner = Runner(bench, tokenizer, budget)
    try:
        bench.prefill(ids[:-1])
        runner.step(ids[-1])
        answer_span = (len(bench.branch.token_ids), len(bench.branch.token_ids) + 2)
        initial = [runner.step(), runner.step()]
        followup = runner.append(case["followups"][0])
        sites = previous.resolve_sites(program, bench, answer_span, followup, copula[0])
        first = bundle(runner, followup["parent_cut"], sites, followup["last_token"], targets)
        save(root / (case["id"] + "-first.json"), first)
        world = first["arms"]["transport"]
        bench.restore(world["end_cut"])
        answer_span2 = (len(bench.branch.token_ids) - 2, len(bench.branch.token_ids))
        followup2 = runner.append(case["followups"][1])
        sites2 = previous.resolve_sites(program, bench, answer_span2, followup2, copula[0])
        second = bundle(runner, followup2["parent_cut"], sites2, followup2["last_token"], targets[::-1])
        return {"case": case, "model_identity": bench.identity, "program": program.to_dict(),
            "initial_answer_span": answer_span, "initial_native": initial, "first": first,
            "second_candidate_world": second, "second_answer_span": answer_span2,
            "second_world_source_cut": world["end_cut"], "physical_token_steps": runner.token_steps,
            "prediction_events": runner.events, "semantic_cot_qualified": False}
    finally:
        bench.close()


def main():
    import qualification_support as support

    support.MAX_ARCHIVE_FILES = 262144
    support.MAX_EXPANDED_BYTES = 2 << 30
    support.MAX_ARCHIVE_BYTES = 512 << 20
    support.REPORT_MARKER = "MDB_READ_WEIGHT_TRANSPORT_REPORT_V1 "
    support.ARCHIVE_MARKER = "MDB_READ_WEIGHT_TRANSPORT_ARCHIVE_V1 "
    raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request = support._read(Path(os.environ["SATURN_DEBUG_REQUEST"]))
    config = json.loads(raw)
    if (support._sha(raw) != request["config_bytes_sha256"] or request["config"] != config
            or support._request_identity(request) != request["request_sha256"]):
        raise RuntimeError("worker request/config identity changed")
    if config["schema"] != "mdb-read-weight-transport-plan-v1" or config["cases"] != fresh_cases():
        raise RuntimeError("frozen transport plan changed")
    if config["doses"] != list(DOSES):
        raise RuntimeError("frozen dose plan changed")
    job_id = os.environ["MRUN_JOB_ID"]
    root = Path(config["output_root"]) / job_id
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config["parameter_manifest"], root / "parameter-artifacts.json")
    shutil.copyfile(config["retrieval_script"], root / "retrieve_weights.py")
    parameter = support._parameter_identity(config, root / "parameter-artifacts.json")
    spec, model, tokenizer, weights, weight_files, content_sha = support._load_model(config, parameter)
    started = time.monotonic()
    cases = []
    budget = {"used": 0, "limit": config["max_native_token_transitions"]}
    try:
        for case in config["cases"]:
            result = run_case(model, tokenizer, spec, weights, case, root, budget)
            save(root / (case["id"] + ".json"), result)
            cases.append(result)
            print("MDB_READ_WEIGHT_TRANSPORT_CASE " + json.dumps({"id": case["id"], "physical_token_steps": result["physical_token_steps"]}), flush=True)
    except BaseException as failure:
        support._emit_partial_custody(root, request, config, cases, failure, parameter=parameter,
            model_spec=spec, weight_files=weight_files, weight_content_sha=content_sha)
        raise
    report = {"schema": "mdb-read-weight-transport-report-v1", "job_id": job_id,
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
        "model": spec.to_dict(), "weights_sha256": weights, "weight_files": weight_files,
        "weight_content_sha256": content_sha, "parameter_identity": parameter,
        "cases": cases, "elapsed_seconds": time.monotonic() - started,
        "physical_token_steps": budget["used"], "prefill_calls": len(cases),
        "prefill_token_count": sum(len(tokenizer.encode(c["prompt"], add_special_tokens=False)) - 1 for c in config["cases"]),
        "receiver_count": 2 * len(cases), "dose_count": len(DOSES),
        "compilation_scope": "one baseline-only smooth FP32 directional compiler per receiver; original dense parameters paid",
        "consumer_scope": "native A output supplied after weight seal; native attention still executes before replacement",
        "mechanics_status": "valid", "trend_status": "all-rows-retained", "terminal_status": "not-assessed",
        "semantic_cot_qualified": False, "economical_full_query_qualified": False}
    manifest = support._write_output_manifest(root)
    archive, sha, size = support._archive_results(root)
    report["artifact_custody"] = {"schema": "mdb-read-weight-transport-custody-v1", "output_root": str(root),
        "manifest": "output-manifest.json", "manifest_sha256": support._file_sha(root / "output-manifest.json"),
        "archive": archive.name, "archive_sha256": sha, "archive_bytes": size,
        "expanded_bytes": manifest["expanded_bytes"], "file_count": manifest["file_count"], "pretrained_weight_copies": False}
    save(root / "report.json", report)
    support._emit_results(root, report)


if __name__ == "__main__":
    main()
