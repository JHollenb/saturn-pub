"""Discover and test bounded, occurrence-specific Qwen memory-read circuits.

Native execution owns every token and state transition. Passive hooks observe
actual projection calls; candidate ranking is a diagnostic, never a causal gate.
Head-local edits use the existing Model OS hidden-memory transaction. The result
is a graph of measured dependencies, not a semantic chain-of-thought transcript.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .bench_circuit_measurement import _closure, _memory, atomic_measurement
from .bench_hidden_memory import logical_address
from .bench_qwen import BenchError, tensor_digest
from .model_os_contracts import content_digest

SCHEMA = "saturn-bench-circuit-discovery-v1"


def capabilities(bench):
    return {
        "protocol": SCHEMA,
        "available": (bench.publication.mode == "journal" and bench.vm.runtime.codec.model_type == "qwen2"
                      and bench.vm.runtime.backend == "native-eager"),
        "start_boundary": "committed-token",
        "candidate_source": "native-query-value-weighted-read",
        "carrier_resolution": "layer/kv-head/committed-row/value",
        "continuation_authority": "branch-owned-greedy-with-explicit-input-segments",
        "semantic_cot_qualified": False,
    }


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise BenchError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _validate(bench, value):
    if not capabilities(bench)["available"]:
        raise BenchError("circuit discovery requires journal-mode native-eager Qwen2")
    if not bench.branches or bench._edit or bench._backend().execution is not None:
        raise BenchError("circuit discovery requires a committed token boundary and no pending Act")
    if not isinstance(value, dict):
        raise BenchError("circuit discovery spec must be an object")
    try:
        spec = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        raise BenchError("circuit discovery spec must be finite JSON") from None
    allowed = {"name", "plan", "receiver_offsets", "layers", "row_start", "row_stop",
               "candidates_per_receiver", "scales", "continuation_tokens", "observable", "top_k",
               "max_native_operations", "max_capture_bytes"}
    if set(spec) - allowed:
        raise BenchError("unknown circuit discovery spec fields")
    name = spec.setdefault("name", "read-circuit")
    if not isinstance(name, str) or not name or len(name) > 128:
        raise BenchError("circuit discovery name must be a bounded nonempty string")
    plan = spec.get("plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 16:
        raise BenchError("declare 1..16 finite input/generate plan segments")
    flat = []
    vocab = bench.model.config.vocab_size
    for segment in plan:
        if not isinstance(segment, dict):
            raise BenchError("plan segment must be an object")
        if segment.get("kind") == "generate" and set(segment) == {"kind", "count"}:
            count = _integer(segment["count"], 1, 32, "generation count")
            flat.extend({"kind": "generate"} for _ in range(count))
        elif segment.get("kind") == "input" and set(segment) == {"kind", "token_ids"}:
            ids = segment["token_ids"]
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 32
                or any(type(t) is not int or not 0 <= t < vocab for t in ids)):
                raise BenchError("input segments require 1..32 native token IDs")
            flat.extend({"kind": "input", "token_id": token} for token in ids)
        else:
            raise BenchError("plan segment is input/token_ids or generate/count")
    if not 1 <= len(flat) <= 32:
        raise BenchError("plan must contain 1..32 token transitions")
    offsets = spec.setdefault("receiver_offsets", [0])
    if (not isinstance(offsets, list) or not 1 <= len(offsets) <= 8
        or any(type(i) is not int or not 0 <= i < len(flat) for i in offsets)
        or len(set(offsets)) != len(offsets)):
        raise BenchError("receiver_offsets must select 1..8 distinct plan positions")
    spec["receiver_offsets"] = sorted(offsets)
    depth = bench.vm.runtime.codec.layers
    layers = spec.setdefault("layers", list(range(depth)))
    if (not isinstance(layers, list) or not layers or len(layers) > 128
        or any(type(i) is not int or not 0 <= i < depth for i in layers)
        or len(set(layers)) != len(layers)):
        raise BenchError("layers must select distinct native decoder depths")
    spec["layers"] = sorted(layers)
    for offset in offsets:
        rows = len(bench.branch.token_ids) + offset
        start = spec.get("row_start", max(0, rows - 256))
        stop = spec.get("row_stop", rows)
        if (type(start) is not int or type(stop) is not int
            or not 0 <= start < stop <= rows or stop - start > 256):
            raise BenchError("source row window must select 1..256 committed rows at every receiver")
    count = _integer(spec.setdefault("candidates_per_receiver", 1), 1, 4, "candidates_per_receiver")
    scales = spec.setdefault("scales", [0.0, 0.5])
    if (not isinstance(scales, list) or not 1 <= len(scales) <= 4
        or any(type(s) not in {int, float} or not 0 <= s <= 2 or not math.isfinite(s) for s in scales)
        or len(set(scales)) != len(scales)):
        raise BenchError("scales require 1..4 distinct finite doses in [0, 2]")
    spec["scales"] = [float(s) for s in scales]
    horizon = _integer(spec.setdefault("continuation_tokens", len(flat)), 1, 32, "continuation_tokens")
    _integer(spec.setdefault("top_k", min(5, vocab)), 1, min(20, vocab), "top_k")
    observable = spec.get("observable")
    if observable is not None and (
        not isinstance(observable, dict)
        or set(observable) != {"kind", "token_id", "against_token_id"}
        or observable["kind"] != "token_margin"
        or any(type(observable[k]) is not int or not 0 <= observable[k] < vocab
               for k in ("token_id", "against_token_id"))
        or observable["token_id"] == observable["against_token_id"]
    ):
        raise BenchError("observable must declare two distinct native token-margin IDs")
    if bench.cursor()["token"] + len(flat) > bench.capacity:
        raise BenchError("circuit discovery exceeds session continuation capacity")
    rows = len(bench.branch.token_ids) + len(flat)
    if rows > bench.model.config.max_position_embeddings:
        raise BenchError("circuit discovery exceeds model context capacity")
    # Baseline with and without observer, each dose, and one exact reinstall
    # per selected carrier. No effect-size or semantic gate enters admission.
    operations = depth * (2 * len(flat) + count * (len(scales) + 1)
                          * sum(min(horizon, len(flat) - i) for i in offsets))
    budget = _integer(spec.setdefault("max_native_operations", 4096), 1, 100000, "max_native_operations")
    if operations > budget:
        raise BenchError(f"circuit discovery native budget exceeded ({operations} layer-token upper bound)")
    size = next(bench.model.parameters()).element_size()
    heads = bench.model.config.num_attention_heads
    hidden = bench.model.config.hidden_size
    capture = int((8 + 4 * 8 / size) * rows * bench.vm.runtime.codec.row_bytes
                  + 4 * heads * (rows * bench.vm.runtime.codec.head_dim + 256 * hidden) * size
                  + 4 * vocab * size + 4 * min(1024, vocab) * hidden)
    limit = _integer(spec.setdefault("max_capture_bytes", 256 << 20), 1, 1 << 30, "max_capture_bytes")
    if capture > limit:
        raise BenchError(f"circuit discovery capture budget exceeded ({capture} byte upper bound)")
    return spec, flat, operations, capture


def discover_circuit(bench, spec):
    normalized, flat, operations, capture = _validate(bench, spec)
    return atomic_measurement(bench, _discover, normalized, flat, operations, capture)


class _ReadObserver:
    """Passive native projection hooks; no model output is replaced."""

    def __init__(self, bench, spec):
        self.bench, self.spec = bench, spec
        self.offset = None
        self.parent = None
        self.query = {}
        self.candidates = []
        self.coverage = []

    @contextlib.contextmanager
    def installed(self):
        handles = []
        try:
            for index in self.spec["layers"]:
                attention = self.bench.model.model.layers[index].self_attn

                def query_hook(_module, _args, output, index=index):
                    if self.offset is not None:
                        self.query[index] = output.detach().clone()

                def output_hook(_module, _args, index=index):
                    if self.offset is not None:
                        self._observe(index, _args[0])

                def return_hook(_module, _args, output, index=index):
                    if self.offset is not None:
                        self.coverage[-1]["attention_output_ref"] = self.bench._tensor_ref(output)
                        self.coverage[-1]["attention_output_sha256"] = tensor_digest(output)

                handles.append(attention.q_proj.register_forward_hook(query_hook))
                handles.append(attention.o_proj.register_forward_pre_hook(output_hook))
                handles.append(attention.o_proj.register_forward_hook(return_hook))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.query.clear()

    def _observe(self, index, native_attended):
        from transformers.models.qwen2 import modeling_qwen2 as modeling

        bench = self.bench
        execution = bench._backend().execution
        if execution is None or execution.next_layer != index or index not in self.query:
            raise BenchError("read observer lost its native projection occurrence")
        state = execution.state
        attention = bench.model.model.layers[index].self_attn
        heads = bench.model.config.num_attention_heads
        groups = bench.vm.runtime.codec.groups
        width = attention.head_dim
        query = self.query.pop(index).view(1, 1, heads, width).transpose(1, 2)
        position = torch.tensor([[state.length]], device=query.device)
        rope = bench.model.model.rotary_emb(execution.hidden, position)
        query, _ = modeling.apply_rotary_pos_emb(query, query, *rope)
        keys = torch.cat((state.ancestor[index][0], state.delta[index][0][:, :, :state.delta_rows + 1]), dim=2)
        values = torch.cat((state.ancestor[index][1], state.delta[index][1][:, :, :state.delta_rows + 1]), dim=2)
        keys = modeling.repeat_kv(keys, heads // groups)
        weights = torch.softmax(query @ keys.transpose(2, 3) * attention.scaling, dim=-1,
                                dtype=torch.float32).to(query.dtype)[0, :, 0]
        if not bool(torch.isfinite(weights).all()):
            raise BenchError("non-finite native read-map observation")
        reconstruction = (weights.unsqueeze(0).unsqueeze(2)
                          @ modeling.repeat_kv(values, heads // groups)).transpose(1, 2).reshape(1, 1, -1)
        error = reconstruction.float() - native_attended.float()
        start = self.spec.get("row_start", max(0, state.length - 256))
        stop = self.spec.get("row_stop", state.length)
        self.coverage.append({"offset": self.offset, "parent_cut": self.parent, "layer": index,
                              "row_start": start, "row_stop": stop,
                              "receiver_position": state.length,
                              "query_sha256": tensor_digest(query),
                              "key_history_sha256": tensor_digest(keys),
                              "softmax_context_rows": state.length + 1,
                              "backend": bench.vm.runtime.backend,
                              "attention_reconstruction_l2": float(error.norm()),
                              "attention_reconstruction_max_abs": float(error.abs().max()),
                              "attention_reconstruction_relative_l2": float(error.norm() / native_attended.float().norm())
                              if float(native_attended.float().norm()) else None,
                              "reconstruction_status": "diagnostic; does not replace native attention",
                              "current_uncommitted_row_excluded_from_candidates": True})
        repeat = heads // groups
        proposed = []
        for group in range(groups):
            value = values[0, group, start:stop]
            per_head, write = [], torch.zeros((stop - start, attention.o_proj.weight.shape[0]),
                                             dtype=torch.float32, device=value.device)
            score = torch.zeros(stop - start, dtype=torch.float32, device=value.device)
            for head in range(group * repeat, (group + 1) * repeat):
                # Projection is an explanatory FP32 contraction of native
                # operands; native BF16 reduction/rounding remains authoritative.
                projected = value.float() @ attention.o_proj.weight[:, head * width:(head + 1) * width].float().T
                weighted = projected * weights[head, start:stop].float().unsqueeze(-1)
                norms = weighted.norm(dim=-1)
                score += norms
                write += weighted
                per_head.append((head, norms))
            for local in torch.argsort(score, descending=True, stable=True)[:self.spec["candidates_per_receiver"]].tolist():
                row = start + local
                proposed.append({"offset": self.offset, "parent_cut": self.parent, "layer": index,
                                 "kv_head": group, "source_row": row,
                                 "source_token_id": bench.branch.token_ids[row],
                                 "value_sha256": tensor_digest(values[:, group:group + 1, row:row + 1]),
                                 "key_sha256": tensor_digest(keys[:, group * repeat:group * repeat + 1, row:row + 1]),
                                 "read_score": float(score[local]),
                                 "group_write_l2": float(write[local].norm()),
                                 "query_heads": [{"head": head,
                                                  "query_sha256": tensor_digest(query[:, head:head + 1]),
                                                  "attention": float(weights[head, row]),
                                                  "weighted_write_l2": float(norms[local])}
                                                 for head, norms in per_head],
                                 "_write": write[local].cpu()})
        self.candidates.extend(proposed)
        # Only a bounded shortlist of write vectors survives each layer.
        self.candidates.sort(key=lambda c: (-c["read_score"], c["layer"], c["kv_head"], c["source_row"]))
        del self.candidates[self.spec["candidates_per_receiver"]:]


def _readout(bench, spec):
    logits = bench.resolve_value("logits").float()[0]
    if not bool(torch.isfinite(logits).all()):
        raise BenchError("circuit discovery native consumer produced non-finite logits")
    observable = spec.get("observable")
    return {"logits_sha256": tensor_digest(bench.resolve_value("logits")),
            "next_token_id": int(logits.argmax()),
            "margin": float(logits[observable["token_id"]] - logits[observable["against_token_id"]])
            if observable else None}


def _run(bench, spec, flat, *, offset=0, observer=None):
    rows, observations = [], []
    for index in range(offset, len(flat)):
        instruction = flat[index]
        parent = bench.branch.cut
        observed = observer is not None and index in spec["receiver_offsets"]
        if observed:
            observer.offset, observer.parent = index, parent
            observer.candidates, observer.coverage = [], []
        if instruction["kind"] == "input":
            bench.stage_input(instruction["token_id"])
        result = bench.step("token")
        rows.append({"offset": index, "parent_cut": parent, "cut_id": bench.branch.cut,
                     "input_authority": "caller-supplied" if instruction["kind"] == "input" else "native-greedy",
                     "consumed_token_id": result["token_id"], "receipt_ref": result["receipt_ref"],
                     "history_sha256": content_digest(bench.branch.token_ids), **_readout(bench, spec)})
        if observed:
            observations.append({"offset": index, "parent_cut": parent,
                                 "receiver_token_id": result["token_id"],
                                 "coverage": observer.coverage, "candidates": observer.candidates})
        if observer is not None:
            observer.offset = None
    return {"trajectory": rows, "closure": _closure(bench, _memory(bench))}, observations


def _compare(bench, actual, reference):
    rows = []
    for a, b in zip(actual["trajectory"], reference["trajectory"], strict=True):
        acut, bcut = bench.artifacts.get(a["cut_id"]), bench.artifacts.get(b["cut_id"])
        av, bv = bench._tensor_load(acut["logits"]).float(), bench._tensor_load(bcut["logits"]).float()
        difference = av - bv
        rows.append({"offset": a["offset"], "same_consumed_token": a["consumed_token_id"] == b["consumed_token_id"],
                     "same_history": a["history_sha256"] == b["history_sha256"],
                     "same_logits": a["logits_sha256"] == b["logits_sha256"],
                     "same_next_token": a["next_token_id"] == b["next_token_id"],
                     "logit_l2": float(difference.norm()), "logit_max_abs": float(difference.abs().max()),
                     "signed_margin_effect": a["margin"] - b["margin"] if a["margin"] is not None else None})
    return {"alignment": "plan-offset; generated tokens remain branch-owned", "steps": rows,
            "closure_exact": actual["closure"]["sha256"] == reference["closure"]["sha256"],
            "same_consumed_tokens": [r["consumed_token_id"] for r in actual["trajectory"]]
            == [r["consumed_token_id"] for r in reference["trajectory"]]}


def _vocabulary(bench, write, top_k):
    device = next(bench.model.parameters()).device
    direction = write.to(device) * bench.model.model.norm.weight.float()
    # Avoid materializing a full FP32 copy of a pretrained vocabulary head.
    head = bench.model.lm_head.weight
    scores = torch.cat([head[start:start + 1024].float() @ direction
                        for start in range(0, head.shape[0], 1024)])
    ids = torch.argsort(scores, descending=True, stable=True)[:top_k].tolist()
    return {"interpretation": "decoded-candidates", "semantic_cot_qualified": False,
            "projection": "FP32 final-norm-gain times read contribution through unembedding; ranks only",
            "tokens": [{"token_id": i, "score": float(scores[i]),
                        "text": bench.tokenizer.decode([i]) if bench.tokenizer is not None else None} for i in ids]}


def _head_edit(bench, candidate, scale):
    layer, row, head = candidate["layer"], candidate["source_row"], candidate["kv_head"]
    value = bench.resolve_value(f"kv_cache.layer_{layer:03d}.value")[:, :, row:row + 1].cpu()
    before = value.clone()
    value[:, head:head + 1].mul_(scale)
    if not torch.equal(value[:, :head], before[:, :head]) or not torch.equal(value[:, head + 1:], before[:, head + 1:]):
        raise BenchError("head-local memory edit changed an unselected head")
    selector = {"layers": [layer], "row_start": row, "row_stop": row + 1, "components": ["value"]}
    act = bench.memory_write({"mode": "replace", "selector": selector,
                              "values": {logical_address(layer, "value", row): value.float().tolist()}})
    actual = bench.resolve_value(f"kv_cache.layer_{layer:03d}.value")[:, :, row:row + 1].cpu()
    if not torch.equal(value, actual):
        raise BenchError("head-local memory edit failed exact payload installation")
    mask = torch.ones(before.shape[1], dtype=torch.bool)
    mask[head] = False
    locality = {"selected_head": head, "before_ref": bench._tensor_ref(before),
                "after_ref": bench._tensor_ref(actual),
                "untouched_heads_before_sha256": tensor_digest(before[:, mask]),
                "untouched_heads_after_sha256": tensor_digest(actual[:, mask]),
                "selected_before_sha256": tensor_digest(before[:, head:head + 1]),
                "selected_after_sha256": tensor_digest(actual[:, head:head + 1])}
    return act, selector, locality


def _graph(observations, tests):
    nodes, edges = [], []
    payloads = {}
    for observation in observations:
        receiver = f"receiver:{observation['offset']}"
        nodes.append({"id": receiver, "kind": "native-receiver", "parent_cut": observation["parent_cut"],
                      "offset": observation["offset"], "token_id": observation["receiver_token_id"]})
        for candidate in observation["candidates"]:
            nodes.append({"id": candidate["id"], "kind": "versioned-memory-value", **candidate})
            edges.append({"source": candidate["id"], "target": receiver, "kind": "observed-native-read",
                          "read_score": candidate["read_score"], "causal_dependency_tested": False})
            payload = (candidate["layer"], candidate["kv_head"], candidate["source_row"], candidate["value_sha256"])
            previous = payloads.get(payload)
            if previous is not None:
                edges.append({"source": previous, "target": candidate["id"],
                              "kind": "same-payload-observation", "value_sha256": candidate["value_sha256"],
                              "causal_dependency_tested": False})
            payloads[payload] = candidate["id"]
    for test in tests:
        target = f"future:{test['id']}"
        nodes.append({"id": target, "kind": "native-future", "cut_id": test["candidate"]["trajectory"][-1]["cut_id"]})
        edges.append({"source": test["candidate_id"], "target": target,
                      "kind": "measured-intervention-response", "test_id": test["id"],
                      "scale": test["scale"], "effect": test["effect"],
                      "interpretation": "conditional response at this receiver; no uniqueness or semantic role claim"})
    for left, right in zip(observations, observations[1:]):
        edges.append({"source": f"receiver:{left['offset']}", "target": f"receiver:{right['offset']}",
                      "kind": "native-execution-order", "causal_dependency_tested": False})
    return {"nodes": nodes, "edges": edges, "semantic_cot_qualified": False,
            "composition_status": "occurrence-graph; edges do not certify a composed minimal circuit"}


@torch.inference_mode()
def _discover(bench, spec, flat, operations, capture):
    started = time.monotonic()
    parent, branch = bench.branch.cut, bench.active
    selected_closure = _closure(bench, _memory(bench))
    siblings = {name: item.cut for name, item in bench.branches.items() if name != branch}
    tests = []
    try:
        observer = _ReadObserver(bench, spec)
        with observer.installed():
            baseline, observations = _run(bench, spec, flat, observer=observer)
        bench.restore(parent)
        unobserved, _ = _run(bench, spec, flat)
        parity = _compare(bench, baseline, unobserved)
        if not parity["closure_exact"] or any(not row["same_logits"] for row in parity["steps"]):
            raise BenchError("passive circuit observer changed native execution")
        for observation in observations:
            offset = observation["offset"]
            stop = min(len(flat), offset + spec["continuation_tokens"])
            reference_rows = baseline["trajectory"][offset:stop]
            bench.restore(reference_rows[-1]["cut_id"])
            reference = {"trajectory": reference_rows, "closure": _closure(bench, _memory(bench))}
            for candidate in observation["candidates"]:
                write = candidate.pop("_write")
                candidate["id"] = "carrier:" + content_digest({k: candidate[k] for k in
                    ("parent_cut", "layer", "kv_head", "source_row")})[:24]
                candidate["vocabulary"] = _vocabulary(bench, write, spec["top_k"])
                candidate["contribution_ref"] = bench._tensor_ref(write)
                source_read = next(row for row in observation["coverage"] if row["layer"] == candidate["layer"])
                for scale in spec["scales"]:
                    bench.restore(observation["parent_cut"])
                    act, selector, locality = _head_edit(bench, candidate, scale)
                    edited_parent = bench.branch.cut
                    native_output = bench._tensor_load(source_read["attention_output_ref"]).float()
                    predicted = native_output + (scale - 1) * write.to(native_output.device)
                    prediction = {"schema": "saturn-circuit-read-write-prediction-v1",
                                  "candidate_id": candidate["id"], "scale": scale,
                                  "baseline_output_ref": source_read["attention_output_ref"],
                                  "contribution_ref": candidate["contribution_ref"],
                                  "predicted_output_ref": bench._tensor_ref(predicted),
                                  "scope": "first receiver attention output; FP32 decomposition with native rounding residual"}
                    prediction_ref = bench.artifacts.put(prediction)
                    witness_spec = {**spec, "receiver_offsets": [offset], "layers": [candidate["layer"]],
                                    "row_start": candidate["source_row"], "row_stop": candidate["source_row"] + 1,
                                    "candidates_per_receiver": 1}
                    witness = _ReadObserver(bench, witness_spec)
                    with witness.installed():
                        future, witnessed = _run(bench, witness_spec, flat[:stop], offset=offset, observer=witness)
                    actual_read = witnessed[0]["coverage"][0]
                    exact_route = (actual_read["query_sha256"] == source_read["query_sha256"]
                                   and actual_read["key_history_sha256"] == source_read["key_history_sha256"])
                    if not exact_route:
                        raise BenchError("value-only discovery changed its first receiver query or keys")
                    actual_output = bench._tensor_load(actual_read["attention_output_ref"]).float()
                    error = actual_output - predicted
                    local_prediction = {"prediction_ref": prediction_ref, "actual_read": actual_read,
                                        "query_and_keys_exact": exact_route,
                                        "error_l2": float(error.norm()), "error_max_abs": float(error.abs().max()),
                                        "relative_l2": float(error.norm() / actual_output.norm())
                                        if float(actual_output.norm()) else None,
                                        "interpretation": "native rounding and contraction error; diagnostic only"}
                    test_id = "test:" + content_digest({"candidate": candidate["id"], "scale": scale})[:24]
                    tests.append({"id": test_id, "candidate_id": candidate["id"], "receiver_offset": offset,
                                  "scale": scale, "act": act, "edited_parent_cut": edited_parent,
                                  "head_locality_exact": True, "head_locality": locality,
                                  "local_prediction": local_prediction,
                                  "reference": reference, "candidate": future,
                                  "effect": _compare(bench, future, reference)})
                # Reinstall the original row into the last candidate before any
                # consumer runs. This tests exact edit reversal, not semantic rescue.
                bench.restore(edited_parent)
                reinstall_act = bench.memory_write({"mode": "copy", "donor_cut": observation["parent_cut"],
                                                     "selector": selector})
                restored, _ = _run(bench, spec, flat[:stop], offset=offset)
                control = _compare(bench, restored, reference)
                if not control["closure_exact"] or any(not row["same_logits"] for row in control["steps"]):
                    raise BenchError("circuit discovery exact-reinstall native closure parity failed")
                candidate["reinstall"] = {"act": reinstall_act, "future": restored, "effect": control}
    finally:
        bench.active = branch
        bench.restore(parent)
    if _closure(bench, _memory(bench)) != selected_closure or bench.branch.cut != parent:
        raise BenchError("circuit discovery failed to restore selected parent")
    if siblings != {name: item.cut for name, item in bench.branches.items() if name != branch}:
        raise BenchError("circuit discovery changed sibling selection")
    source = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    source_ref = bench.artifacts.put({"schema": "saturn-instrument-source-v1", "module": __name__,
                                      "source_sha256": source_hash, "source_text": source.decode()})
    report = {"schema": SCHEMA, "name": spec["name"], "session_id": bench.session_id,
              "model_identity": bench.identity, "lane": bench.lane,
              "numerical_contract": bench.capabilities()["numerical_contract"],
              "parent_cut": parent, "spec": spec, "spec_sha256": content_digest(spec),
              "instrument_identity": {"module": __name__, "source_sha256": source_hash, "source_ref": source_ref},
              "input_authority": {"input_segments": "caller-supplied", "generated_segments": "branch-owned-native-greedy",
                                  "first_pending_token": "already selected at parent unless replaced by explicit input"},
              "baseline": baseline, "observer_control": {"future": unobserved, "effect": parity},
              "receivers": observations, "tests": tests, "graph": _graph(observations, tests),
              "selected_parent_restored_exact": True, "sibling_selections_preserved": True,
              "mechanics_status": "valid", "instrument_status": "native-consumers-executed",
              "trend_status": "observed", "terminal_status": "not-assessed", "semantic_cot_qualified": False,
              "operation_upper_bound": operations, "capture_upper_bound_bytes": capture,
              "operation_budget_unit": "native layer-token invocations; excludes observer and publication work",
              "elapsed_seconds": time.monotonic() - started,
              "limitations": ["bounded candidate search; ranking is not proof of causal importance",
                              "only newly executed reads are observed; prefill is not retrospectively decoded",
                              "write projections are FP32 explanatory contractions, not native low-precision decompositions",
                              "head-value scaling does not establish a semantic variable or a unique/minimal circuit",
                              "later receiver parents come from the baseline history; each tested suffix owns its generated tokens",
                              "separate intervention responses do not certify causal composition across receiver edges",
                              "full native model and dense execution remain in use"]}
    ref = bench.artifacts.put(report)
    bench.events.append({"kind": "circuit-discovered", "parent": parent, "report_ref": ref})
    bench._publish()
    return {"op": "discover_circuit", "cut_id": parent, "report_ref": ref, "report": report}


def inspect_discoveries(bench):
    return [{"report_ref": e["report_ref"], "parent_cut": e["parent"],
             "semantic_cot_qualified": False} for e in bench.events if e.get("kind") == "circuit-discovered"]
