"""Finite receiver-assisted composition of two previously discovered Qwen reads."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from mdb.bench import BenchSession
from mdb.bench_circuit_discovery import _head_edit
from mdb.bench_circuit_measurement import _closure, _memory
from mdb.bench_hidden_memory import logical_address
from mdb.bench_qwen import tensor_digest


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def error(actual, expected):
    delta = actual.float() - expected.float()
    norm = float(expected.float().norm())
    return {"l2": float(delta.norm()), "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm()) / norm if norm else None,
            "exact": tensor_digest(actual) == tensor_digest(expected)}


class Capture:
    def __init__(self, bench, sites):
        self.bench, self.sites = bench, sites
        self.active = True
        self.live, self.refs = {}, {}

    def keep(self, name, tensor):
        tensor = tensor.detach().clone()
        self.live[name] = tensor
        self.refs[name] = self.bench._tensor_ref(tensor)

    @contextlib.contextmanager
    def installed(self):
        handles = []
        runtime = self.bench.vm.runtime
        native_advance = runtime.advance_token

        def observe_boundary(execution):
            index = execution.next_layer
            observed = self.active and index in (19, 22)
            if observed:
                self.keep(f"input{index}", execution.hidden)
            result = native_advance(execution)
            if observed:
                self.keep(f"exit{index}", execution.hidden)
            return result

        runtime.advance_token = observe_boundary
        try:
            for index in (19, 22):
                layer = self.bench.model.model.layers[index]

                def query(_module, _args, output, index=index):
                    if self.active:
                        self.live[f"rawq{index}"] = output.detach().clone()

                def attention(_module, _args, output, index=index):
                    if self.active:
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
        heads, groups, width = bench.model.config.num_attention_heads, bench.vm.runtime.codec.groups, module.head_dim
        q = self.live.pop(f"rawq{index}").view(1, 1, heads, width).transpose(1, 2)
        position = torch.tensor([[state.length]], device=q.device)
        rope = bench.model.model.rotary_emb(execution.hidden, position)
        q, _ = modeling.apply_rotary_pos_emb(q, q, *rope)
        k = torch.cat((state.ancestor[index][0], state.delta[index][0][:, :, :state.delta_rows + 1]), dim=2)
        v = torch.cat((state.ancestor[index][1], state.delta[index][1][:, :, :state.delta_rows + 1]), dim=2)
        expanded = modeling.repeat_kv(k, heads // groups)
        weights = torch.softmax(q @ expanded.transpose(2, 3) * module.scaling,
                                dim=-1, dtype=torch.float32).to(q.dtype)[0, :, 0]
        for name, value in ((f"q{index}", q), (f"k{index}", k), (f"current_v{index}", v[:, :, -1:]),
                            (f"attention{index}", output), (f"weights{index}", weights)):
            self.keep(name, value)
        if index == 22:
            site = self.sites["B"]
            head, row = site["kv_head"], site["source_row"]
            value = v[0, head, row]
            contribution = torch.zeros(output.shape[-1], dtype=torch.float32, device=output.device)
            for h in range(head * (heads // groups), (head + 1) * (heads // groups)):
                contribution += (module.o_proj.weight[:, h * width:(h + 1) * width].float()
                                 @ value.float()) * weights[h, row].float()
            self.keep("contribution_B", contribution)
            self.keep("source_B", value)


def predict_exit(layer, incoming, attention):
    """Explicit native-width BF16 SwiGLU equations; all dense weights are paid."""
    import torch.nn.functional as functional

    residual = incoming + attention.to(incoming.dtype)
    norm = residual.float()
    norm = norm * torch.rsqrt(norm.pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.variance_epsilon)
    norm = layer.post_attention_layernorm.weight * norm.to(residual.dtype)
    gate = functional.linear(norm, layer.mlp.gate_proj.weight, layer.mlp.gate_proj.bias)
    up = functional.linear(norm, layer.mlp.up_proj.weight, layer.mlp.up_proj.bias)
    mlp = functional.linear(functional.silu(gate) * up, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
    return mlp, residual + mlp


class Runner:
    def __init__(self, bench, tokenizer, budget=None):
        self.bench, self.tokenizer = bench, tokenizer
        self.token_steps = 0
        self.events = []
        self.budget = budget if budget is not None else {"used": 0, "limit": 600}

    def step(self, token=None):
        if token is not None:
            self.bench.stage_input(int(token))
        self.token_steps += 1
        self.budget["used"] += 1
        if self.budget["used"] > self.budget["limit"]:
            raise RuntimeError("declared physical token bound exceeded")
        return self.bench.step()

    def append(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        start = len(self.bench.branch.token_ids)
        for token in ids[:-1]:
            self.step(token)
        return {"token_ids": ids, "start": start, "stop": start + len(ids) - 1,
                "last_token": ids[-1], "parent_cut": self.bench.branch.cut}

    def arm(self, name, parent, sites, last_token, target_ids, *, edits=(), injection=None, capture=True):
        bench = self.bench
        bench.restore(parent)
        acts = []
        for port in edits:
            act, _, locality = _head_edit(bench, sites[port], 0.0)
            acts.append({"port": port, "act": act, "locality": locality})
        if name == "reinstall":
            for port in edits:
                site = sites[port]
                row = bench._tensor_load(acts[list(edits).index(port)]["locality"]["before_ref"])
                selector = {"layers": [site["layer"]], "components": ["value"],
                            "row_start": site["source_row"], "row_stop": site["source_row"] + 1}
                bench.memory_write({"mode": "replace", "selector": selector,
                                    "values": {logical_address(site["layer"], "value", site["source_row"]): row.float().tolist()}})
        observer = Capture(bench, sites)
        rows = []
        with contextlib.ExitStack() as stack:
            if injection is not None:
                def inject(_module, _args, output):
                    return injection.to(output) if observer.active else output
                handle = bench.model.model.layers[22].self_attn.o_proj.register_forward_hook(inject)
                stack.callback(handle.remove)
            if capture:
                stack.enter_context(observer.installed())
            for index in range(3):
                result = self.step(last_token if index == 0 else None)
                logits = bench.resolve_value("logits").float()[0]
                rows.append({"offset": index, "authority": "caller-supplied" if index == 0 else "native-greedy",
                             "consumed_token_id": result["token_id"], "next_token_id": result["next_token_id"],
                             "next_text": self.tokenizer.decode([result["next_token_id"]]),
                             "cut": result["cut"], "receipt_ref": result["receipt_ref"],
                             "logits_sha256": result["logits_sha256"],
                             "target_margin": float(logits[target_ids[0]] - logits[target_ids[1]]),
                             "correct_first_answer": result["next_token_id"] == target_ids[0] if index == 0 else None,
                             "margin_role": "first-object-contrast" if index == 0 else "target-token-diagnostic"})
                observer.active = False
        record = {"name": name, "parent_cut": parent, "acts": acts, "trajectory": rows,
                  "capture": observer.refs, "end_cut": bench.branch.cut,
                  "closure_sha256": _closure(bench, _memory(bench))["sha256"]}
        return record, observer.live


def signature(arm):
    return [(row["consumed_token_id"], row["logits_sha256"]) for row in arm["trajectory"]]


def compare_arms(bench, candidate, reference):
    rows = []
    for a, b in zip(candidate["trajectory"], reference["trajectory"], strict=True):
        av = bench._tensor_load(bench.artifacts.get(a["cut"])["logits"])
        bv = bench._tensor_load(bench.artifacts.get(b["cut"])["logits"])
        rows.append({"offset": a["offset"], "same_consumed": a["consumed_token_id"] == b["consumed_token_id"],
                     "same_next": a["next_token_id"] == b["next_token_id"],
                     "target_margin_change": a["target_margin"] - b["target_margin"], **error(av, bv)})
    return rows


@torch.inference_mode()
def bundle(runner, parent, sites, last_token, target_ids):
    bench = runner.bench
    arms, live = {}, {}
    for name, edits, capture in (("00", (), True), ("sham", (), False), ("A", ("A",), True),
                                 ("B", ("B",), True), ("wrong-row", ("wrong",), False),
                                 ("reinstall", ("A", "B"), False)):
        arms[name], live[name] = runner.arm(name, parent, sites, last_token, target_ids, edits=edits, capture=capture)
    for control in ("sham", "reinstall"):
        if (signature(arms[control]) != signature(arms["00"])
                or arms[control]["closure_sha256"] != arms["00"]["closure_sha256"]):
            raise RuntimeError(control + " failed exact native continuation closure")
    a, zero = live["A"], live["00"]
    predictions, tensors = {}, {}
    for name, contribution in (("conditional", a["contribution_B"]), ("stale", zero["contribution_B"])):
        attention = a["attention22"].float() - contribution
        consumed = attention.to(a["attention22"].dtype)
        mlp, exit_state = predict_exit(bench.model.model.layers[22], a["input22"], consumed)
        tensors[name] = {"attention": attention, "consumed_attention": consumed, "mlp": mlp, "exit": exit_state}
        predictions[name] = {"assistance": "native-A-receiver" if name == "conditional" else "native-A-output-plus-00-read-weights",
                             "input22_ref": arms["A"]["capture"]["input22"],
                             "baseline_attention_ref": arms["A"]["capture"]["attention22"],
                             "contribution_ref": bench._tensor_ref(contribution),
                             **{key + "_ref": bench._tensor_ref(value) for key, value in tensors[name].items()},
                             "dense_mlp_linear_calls": 3, "native_attention_output_assisted": True,
                             "injected_replay_native_attention_omitted": False}
    prediction_record = {"schema": "mdb-two-read-composition-prediction-v1", "parent_cut": parent,
                         "sites": sites, "predictions": predictions, "physical_steps_at_seal": runner.token_steps,
                         "program_status": "frozen-law; supplied-role-constructors; no fitting"}
    prediction_ref = bench.artifacts.put(prediction_record)
    runner.events.append({"kind": "prediction-sealed", "ref": prediction_ref, "physical_steps": runner.token_steps})
    for name in ("conditional", "stale"):
        arms[name], live[name] = runner.arm(name, parent, sites, last_token, target_ids,
                                          edits=("A", "B"), injection=tensors[name]["consumed_attention"])
    runner.events.append({"kind": "native-AB-started", "prediction_ref": prediction_ref,
                          "physical_steps": runner.token_steps})
    arms["AB"], live["AB"] = runner.arm("AB", parent, sites, last_token, target_ids, edits=("A", "B"))
    ab = live["AB"]
    route_exact = (torch.equal(a["q22"], ab["q22"]) and torch.equal(a["k22"], ab["k22"])
                   and torch.equal(a["current_v22"], ab["current_v22"]))
    if not route_exact or not torch.equal(zero["q19"], a["q19"]) or not torch.equal(zero["k19"], a["k19"]):
        raise RuntimeError("value intervention failed its declared fixed-query/key receiver closure")
    errors = {}
    for name in ("conditional", "stale"):
        errors[name] = {key: error(tensors[name][key], ab[native]) for key, native in (
            ("attention", "attention22"), ("mlp", "mlp22"), ("exit", "exit22"))}
        errors[name]["injected_mlp_exact"] = torch.equal(tensors[name]["mlp"], live[name]["mlp22"])
        errors[name]["injected_exit_exact"] = torch.equal(tensors[name]["exit"], live[name]["exit22"])
    interaction = ab["exit22"].float() - a["exit22"].float() - live["B"]["exit22"].float() + zero["exit22"].float()
    result = {"parent_cut": parent, "sites": sites, "semantic_target_ids": target_ids,
              "arms": arms, "prediction_ref": prediction_ref,
              "prediction_record": prediction_record, "local_errors": errors,
              "query22_change": error(a["q22"], zero["q22"]),
              "current_key22_change": error(a["k22"][:, :, -1:], zero["k22"][:, :, -1:]),
              "current_value22_change": error(a["current_v22"], zero["current_v22"]),
              "old_keys22_unchanged": torch.equal(a["k22"][:, :, :-1], zero["k22"][:, :, :-1]),
              "A_vs_AB_query_keys_current_value_exact": route_exact,
              "A_vs_AB_input22_exact": torch.equal(a["input22"], ab["input22"]),
              "A_vs_00_source_B_exact": torch.equal(a["source_B"], zero["source_B"]),
              "layer22_exit_interaction_ref": bench._tensor_ref(interaction),
              "layer22_exit_interaction_l2": float(interaction.norm()),
              "future_comparisons": {name: compare_arms(bench, arms[name], arms["AB"])
                                     for name in ("conditional", "stale")},
              "target_margin_factorial_interaction": (arms["AB"]["trajectory"][0]["target_margin"]
                  - arms["A"]["trajectory"][0]["target_margin"] - arms["B"]["trajectory"][0]["target_margin"]
                  + arms["00"]["trajectory"][0]["target_margin"]),
              "mechanics_status": "valid", "terminal_status": "not-assessed"}
    bench.restore(parent)
    return result


def fresh_cases():
    rows = []
    for a, b, x, y in (("Nina", "Samir", "key", "coin"), ("Ada", "Noah", "book", "cup")):
        for previous in (a, b):
            rows.append({"id": f"{a.lower()}-{previous.lower()}", "owners": [a, b], "objects": [x, y],
                         "previous_owner": previous,
                         "prompt": f"{a} placed the {x} in the box. {b} placed the {y} in the bag. The object {previous} placed is the",
                         "followups": [f" The object {a} placed is the", f" The object {b} placed is the"],
                         "previous_current_owner_relation": "same" if previous == a else "other"})
    return rows


def resolve_sites(program, bench, answer_span, followup, copula):
    resolved = program.resolve(list(bench.branch.token_ids), preceding_answer_span=answer_span,
                               followup_span=(followup["start"], followup["stop"]), copula_token_id=copula)
    if resolved is None or any(site is None for site in resolved):
        raise RuntimeError("frozen locator abstained: copula absent from current follow-up")
    # FrozenReadProgram returns named role-resolved physical addresses.
    sites = {site["port"]: {**site, "source_row": site.get("source_row", site.get("row"))} for site in resolved}
    sites["wrong"] = {**sites["A"], "source_row": followup["start"],
                      "source_token_id": bench.branch.token_ids[followup["start"]], "role": "wrong-row-control"}
    return sites


@torch.inference_mode()
def run_case(model, tokenizer, model_spec, weights_sha, case, root, program, budget=None):
    ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
    copula_ids = tokenizer.encode(" is", add_special_tokens=False)
    object_ids = [tokenizer.encode(" " + word, add_special_tokens=False) for word in case["objects"]]
    if len(copula_ids) != 1 or any(len(tokens) != 1 for tokens in object_ids):
        raise RuntimeError("declared single-token semantic contrast was not tokenized as such")
    target_ids = [tokens[0] for tokens in object_ids]
    bench = BenchSession.from_resident(model, tokenizer, root=root / case["id"], model_id=model_spec.model_id,
                                       revision=model_spec.revision, model_sha256=weights_sha,
                                       session_id=case["id"], attention_backend="native-eager", continuation_capacity=128)
    runner = Runner(bench, tokenizer, budget)
    try:
        bench.prefill(ids[:-1])
        runner.step(ids[-1])
        preceding = (len(bench.branch.token_ids), len(bench.branch.token_ids) + 2)
        initial = [runner.step(), runner.step()]
        followup = runner.append(case["followups"][0])
        sites = resolve_sites(program, bench, preceding, followup, copula_ids[0])
        first = bundle(runner, followup["parent_cut"], sites, followup["last_token"], target_ids)
        save(root / (case["id"] + "-first.json"), first)
        policies = {}
        second = None
        for policy, edits in (("00", ()), ("A", ("A",)), ("B", ("B",)), ("AB", ("A", "B"))):
            world = first["arms"][policy]
            bench.restore(world["end_cut"])
            old_len = len(bench.branch.token_ids)
            answer_span = (old_len - 2, old_len)
            later = runner.append(case["followups"][1])
            later_sites = resolve_sites(program, bench, answer_span, later, copula_ids[0])
            if policy == "AB":
                second = bundle(runner, later["parent_cut"], later_sites, later["last_token"], target_ids[::-1])
                outcome = second["arms"]["AB"]
            else:
                outcome, _ = runner.arm("repeat-" + policy, later["parent_cut"], later_sites,
                                         later["last_token"], target_ids[::-1], edits=edits)
            policies[policy] = {"first_world_cut": world["end_cut"], "preceding_answer_span": answer_span,
                                "resolved_sites": later_sites, "outcome": outcome}
        expected_initial = target_ids[case["owners"].index(case["previous_owner"])]
        return {"case": case, "initial_native": initial, "program": program.to_dict(),
                "copula_token_id": copula_ids[0], "object_token_ids": target_ids,
                "initial_expected_id": expected_initial,
                "initial_answer_correct": initial[0]["token_id"] == expected_initial,
                "initial_answer_span": preceding, "first": first, "second_AB_world": second,
                "recurrent_policies": policies, "prediction_events": runner.events,
                "physical_token_steps": runner.token_steps,
                "model_identity": bench.identity, "numerical_contract": bench.capabilities()["numerical_contract"],
                "semantic_cot_qualified": False, "economical_query_formation_qualified": False}
    finally:
        bench.close()


def main():
    import qualification_support as support

    from mdb.read_program import FrozenReadProgram

    support.MAX_ARCHIVE_FILES = 262144
    support.MAX_EXPANDED_BYTES = 2 << 30
    support.MAX_ARCHIVE_BYTES = 512 << 20
    support.REPORT_MARKER = "MDB_READ_COMPOSITION_REPORT_V1 "
    support.ARCHIVE_MARKER = "MDB_READ_COMPOSITION_ARCHIVE_V1 "

    raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request = support._read(Path(os.environ["SATURN_DEBUG_REQUEST"]))
    config = json.loads(raw)
    if (support._sha(raw) != request["config_bytes_sha256"] or request["config"] != config
            or support._request_identity(request) != request["request_sha256"]):
        raise RuntimeError("worker request/config identity changed")
    if config.get("schema") != "mdb-read-composition-plan-v1":
        raise RuntimeError("unexpected frozen composition plan schema")
    if config.get("cases") != fresh_cases():
        raise RuntimeError("sealed case plan differs from the frozen fresh-context declaration")
    job_id = os.environ["MRUN_JOB_ID"]
    root = Path(config["output_root"]) / job_id
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config["parameter_manifest"], root / "parameter-artifacts.json")
    shutil.copyfile(config["retrieval_script"], root / "retrieve_weights.py")
    parameter = support._parameter_identity(config, root / "parameter-artifacts.json")
    spec, model, tokenizer, weights, weight_files, content_sha = support._load_model(config, parameter)
    program = FrozenReadProgram.standard()
    save(root / "frozen-program.json", program.to_dict())
    started = time.monotonic()
    cases = []
    budget = {"used": 0, "limit": config["max_native_token_transitions"]}
    try:
        for case in config["cases"]:
            result = run_case(model, tokenizer, spec, weights, case, root, program, budget)
            save(root / (case["id"] + ".json"), result)
            cases.append(result)
            print("MDB_READ_COMPOSITION_CASE " + json.dumps({"id": case["id"], "physical_token_steps": result["physical_token_steps"]}), flush=True)
    except BaseException as failure:
        support._emit_partial_custody(root, request, config, cases, failure, parameter=parameter,
                                      model_spec=spec, weight_files=weight_files,
                                      weight_content_sha=content_sha)
        raise
    report = {"schema": "mdb-read-composition-report-v1", "job_id": job_id,
              "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
              "model": spec.to_dict(), "weights_sha256": weights, "weight_files": weight_files,
              "weight_content_sha256": content_sha, "parameter_identity": parameter,
              "program": program.to_dict(), "cases": cases, "elapsed_seconds": time.monotonic() - started,
              "physical_token_steps": sum(case["physical_token_steps"] for case in cases),
              "explicit_dense_mlp_linear_calls": 48,
              "prefill_calls": 4, "mechanics_status": "valid", "trend_status": "all-rows-retained",
              "prefill_token_count": sum(len(tokenizer.encode(case["prompt"], add_special_tokens=False)) - 1 for case in config["cases"]),
              "terminal_status": "not-assessed", "semantic_cot_qualified": False,
              "economical_query_formation_qualified": False,
              "prediction_scope": "native-A-receiver-assisted; native full attention A return and current Q/K/V paid; explicit dense MLP predicted and paid"}
    manifest = support._write_output_manifest(root)
    archive, sha, size = support._archive_results(root)
    report["artifact_custody"] = {"schema": "mdb-read-composition-custody-v1", "output_root": str(root),
        "manifest": "output-manifest.json", "manifest_sha256": support._file_sha(root / "output-manifest.json"),
        "archive": archive.name, "archive_sha256": sha, "archive_bytes": size,
        "expanded_bytes": manifest["expanded_bytes"], "file_count": manifest["file_count"], "pretrained_weight_copies": False}
    save(root / "report.json", report)
    support._emit_results(root, report)


if __name__ == "__main__":
    main()
