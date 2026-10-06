"""Paired native reads of an admitted hidden-memory edit from one frozen parent."""

from __future__ import annotations

import json
import math
import re
import time
from hashlib import sha256
from pathlib import Path

import torch

from .bench_circuit_measurement import _closure, _memory, atomic_measurement
from .bench_qwen import BenchError, tensor_digest
from .model_os_contracts import content_digest

SCHEMA = "saturn-hidden-memory-probe-v1"


def _validate(bench, value):
    if (
        not bench.branches
        or bench._edit
        or bench._backend().execution is not None
        or bench.cursor()["phase"] != "token"
    ):
        raise BenchError("memory probe requires a committed token boundary and no pending Act")
    if bench.publication.mode != "journal":
        raise BenchError("memory probe requires journal publication")
    if not isinstance(value, dict):
        raise BenchError("memory probe spec must be an object")
    try:
        spec = json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError):
        raise BenchError("memory probe spec must be finite JSON") from None
    allowed = {
        "name",
        "intervention",
        "query",
        "query_token_ids",
        "continuation_tokens",
        "observable",
        "max_native_operations",
        "max_capture_bytes",
        "top_k",
    }
    if set(spec) - allowed:
        raise BenchError("unknown memory probe spec fields")
    name = spec.setdefault("name", "memory-read")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*", name):
        raise BenchError("memory probe name must be a safe identifier")
    if ("query" in spec) == ("query_token_ids" in spec):
        raise BenchError("declare exactly one fixed query or query_token_ids")
    if "query" in spec:
        if not isinstance(spec["query"], str) or bench.tokenizer is None:
            raise BenchError("text memory queries require a resident tokenizer")
        query = bench.tokenizer.encode(spec["query"], add_special_tokens=False)
    else:
        query = spec["query_token_ids"]
    if (
        not isinstance(query, list)
        or not 1 <= len(query) <= 64
        or any(type(t) is not int or not 0 <= t < bench.model.config.vocab_size for t in query)
    ):
        raise BenchError("fixed memory query must contain 1..64 native vocabulary IDs")
    spec["query_token_ids"] = query
    horizon = spec.setdefault("continuation_tokens", 4)
    if type(horizon) is not int or not 1 <= horizon <= 32:
        raise BenchError("memory probe continuation_tokens must be 1..32")
    top_k = spec.setdefault("top_k", 5)
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise BenchError("memory probe top_k must be 1..20")
    observable = spec.get("observable")
    if observable is not None:
        if (
            not isinstance(observable, dict)
            or set(observable) != {"kind", "token_id", "against_token_id"}
            or observable["kind"] != "token_margin"
            or any(
                type(observable[k]) is not int
                or not 0 <= observable[k] < bench.model.config.vocab_size
                for k in ("token_id", "against_token_id")
            )
            or observable["token_id"] == observable["against_token_id"]
        ):
            raise BenchError("observable must declare two distinct native token-margin IDs")
    intervention = spec.get("intervention")
    if not isinstance(intervention, dict):
        raise BenchError("memory probe requires an intervention object")
    # Validate the edit before a reference future executes or rollback begins.
    from .bench_hidden_memory import normalize_write_spec

    spec["intervention"] = normalize_write_spec(bench, intervention)
    layers = bench.model.config.num_hidden_layers
    consumed = len(query) + horizon
    upper = 3 * layers * consumed
    budget = spec.setdefault("max_native_operations", 4096)
    if type(budget) is not int or not 1 <= budget <= 100000 or upper > budget:
        raise BenchError(f"memory probe native budget exceeded ({upper} layer-token upper bound)")
    if bench.cursor()["token"] + consumed > bench.capacity:
        raise BenchError("memory probe exceeds session continuation capacity")
    rows = len(bench.branch.token_ids) + consumed
    if rows > bench.model.config.max_position_embeddings:
        raise BenchError("memory probe exceeds model context capacity")
    element_size = next(bench.model.parameters()).element_size()
    # Full native closure captures and double-precision comparison workspace.
    capture = int(
        (6 + 4 * 8 / element_size)
        * (rows * bench.vm.runtime.codec.row_bytes + bench.model.config.vocab_size * element_size)
    )
    limit = spec.setdefault("max_capture_bytes", 64 << 20)
    if type(limit) is not int or not 1 <= limit <= 1 << 30 or capture > limit:
        raise BenchError(f"memory probe capture budget exceeded ({capture} byte upper bound)")
    return spec, upper, capture


def atomic_probe(bench, spec):
    normalized, operations, capture = _validate(bench, spec)
    return atomic_measurement(bench, memory_probe, normalized, operations, capture)


def _decode(bench, ids):
    return bench.tokenizer.decode(ids) if bench.tokenizer is not None else None


def _readout(bench, spec):
    logits = bench.resolve_value("logits").float()
    if not bool(torch.isfinite(logits).all()):
        raise BenchError("native memory consumer produced non-finite logits")
    scores, ids = torch.topk(logits[0], min(spec["top_k"], logits.shape[-1]))
    observable = spec.get("observable")
    margin = (
        float(logits[0, observable["token_id"]] - logits[0, observable["against_token_id"]])
        if observable
        else None
    )
    return {
        "cut_id": bench.branch.cut,
        "logits_sha256": tensor_digest(bench.resolve_value("logits")),
        "raw_logit_margin": margin,
        "top_tokens": [
            {"token_id": int(t), "text": _decode(bench, [int(t)]), "raw_logit": float(v)}
            for t, v in zip(ids, scores, strict=True)
        ],
        "pending_token_id": bench.branch.pending,
    }


def _consume(bench, spec, role):
    steps = []
    for token in spec["query_token_ids"]:
        bench.stage_input(token)
        steps.append(bench.step("token"))
    first = _readout(bench, spec)
    answers, readouts = [], []
    for _ in range(spec["continuation_tokens"]):
        readouts.append(_readout(bench, spec))
        answers.append(bench.branch.pending)
        steps.append(bench.step("token"))
    closure = _closure(bench, _memory(bench))
    return {
        "role": role,
        "query_token_ids": spec["query_token_ids"],
        "answer_token_ids": answers,
        "answer_text": _decode(bench, answers),
        "first_readout": first,
        "answer_readouts": readouts,
        "endpoint_readout": _readout(bench, spec),
        "closure": closure,
        "endpoint_cut": bench.branch.cut,
        "native_steps": steps,
        "native_output_measured": True,
    }


def _compare(candidate, native):
    actual = candidate["first_readout"]["raw_logit_margin"]
    baseline = native["first_readout"]["raw_logit_margin"]
    return {
        "signed_effect": actual - baseline if actual is not None else None,
        "observable_boundary": "first-answer-readout-after-fixed-query",
        "same_answer_tokens": candidate["answer_token_ids"] == native["answer_token_ids"],
        "same_first_logits": candidate["first_readout"]["logits_sha256"]
        == native["first_readout"]["logits_sha256"],
        "closure_exact": candidate["closure"]["sha256"] == native["closure"]["sha256"],
        "native_output_measured": True,
    }


@torch.inference_mode()
def memory_probe(bench, spec, operations, capture):
    started = time.monotonic()
    parent, branch = bench.branch.cut, bench.active
    parent_closure = _closure(bench, _memory(bench))
    sibling_cuts = {name: item.cut for name, item in bench.branches.items() if name != branch}
    act = reinstall = None
    try:
        native = _consume(bench, spec, "native")
        bench.restore(parent)
        act = bench.memory_write(spec["intervention"])
        candidate_cut = bench.branch.cut
        candidate = _consume(bench, spec, "candidate")
        bench.restore(candidate_cut)
        reinstall = bench.memory_write(
            {"mode": "copy", "donor_cut": parent, "selector": act["selector"]}
        )
        restored = _consume(bench, spec, "reinstall")
        control = _compare(restored, native)
        if not control["closure_exact"]:
            raise BenchError("memory probe exact-reinstall native closure parity failed")
    finally:
        bench.active = branch
        bench.restore(parent)
    selected_closure = _closure(bench, _memory(bench))
    if selected_closure != parent_closure or bench.branch.cut != parent:
        raise BenchError("memory probe failed to restore selected parent")
    if sibling_cuts != {name: item.cut for name, item in bench.branches.items() if name != branch}:
        raise BenchError("memory probe changed sibling selection")
    source_bytes = Path(__file__).read_bytes()
    source_hash = sha256(source_bytes).hexdigest()
    source_ref = bench.artifacts.put(
        {
            "schema": "saturn-instrument-source-v1",
            "module": "mdb.bench_memory_probe",
            "source_sha256": source_hash,
            "source_text": source_bytes.decode("utf-8"),
        }
    )
    report = {
        "schema": SCHEMA,
        "name": spec["name"],
        "session_id": bench.session_id,
        "model_identity": bench.identity,
        "lane": bench.lane,
        "numerical_contract": bench.capabilities()["numerical_contract"],
        "parent_cut": parent,
        "spec": spec,
        "spec_sha256": content_digest(spec),
        "instrument_identity": {
            "module": "mdb.bench_memory_probe",
            "source_sha256": source_hash,
            "source_ref": source_ref,
        },
        "input_authority": {
            "query": "caller-supplied-fixed-input",
            "answer": "native-greedy",
            "initial_pending_token": "replaced-by-fixed-query-before-consumption",
        },
        "native": native,
        "candidate": candidate,
        "reinstall": restored,
        "act": act,
        "reinstall_act": reinstall,
        "effect": _compare(candidate, native),
        "reinstall_effect": control,
        "selected_parent_restored_exact": True,
        "sibling_selections_preserved": True,
        "mechanics_status": "valid",
        "instrument_status": "native-consumer-executed",
        "trend_status": "observed",
        "terminal_status": "not-assessed",
        "operation_upper_bound": operations,
        "capture_upper_bound_bytes": capture,
        "elapsed_seconds": time.monotonic() - started,
    }
    ref = bench.artifacts.put(report)
    bench.events.append({"kind": "hidden-memory-probe", "parent": parent, "report_ref": ref})
    bench._publish()
    return {"op": "memory_probe", "cut_id": parent, "report_ref": ref, "report": report}


def verify_artifact(value, artifacts, manifest, vm_artifacts=None):
    """Validate frozen probe references and same-parent paired output provenance."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return False
    if value["model_identity"] != manifest["identity"] or value["lane"] != manifest["lane"]:
        raise ValueError("memory probe model/lane identity differs from manifest")
    if content_digest(value["spec"]) != value["spec_sha256"]:
        raise ValueError("memory probe spec digest mismatch")
    instrument = value["instrument_identity"]
    source = artifacts.get(instrument["source_ref"])
    if (
        source["module"] != instrument["module"]
        or source["source_sha256"] != instrument["source_sha256"]
        or sha256(source["source_text"].encode("utf-8")).hexdigest() != instrument["source_sha256"]
    ):
        raise ValueError("memory probe instrument source custody mismatch")
    parent = artifacts.get(value["parent_cut"])
    if parent["identity"] != manifest["identity"] or parent["lane"] != manifest["lane"]:
        raise ValueError("memory probe parent identity mismatch")
    from .bench_hidden_memory import verify_write_act

    verify_write_act(value["act"], artifacts, vm_artifacts)
    verify_write_act(value["reinstall_act"], artifacts, vm_artifacts)
    for role in ("native", "candidate", "reinstall"):
        arm = value[role]
        if arm["query_token_ids"] != value["spec"]["query_token_ids"]:
            raise ValueError("memory probe paired query mismatch")
        endpoint = artifacts.get(arm["endpoint_cut"])
        if endpoint["identity"] != parent["identity"] or endpoint["lane"] != parent["lane"]:
            raise ValueError("memory probe endpoint identity mismatch")
        expected_tokens = parent["token_ids"] + arm["query_token_ids"] + arm["answer_token_ids"]
        if (
            endpoint["token_ids"] != expected_tokens
            or arm["closure"]["token_ids"] != expected_tokens
        ):
            raise ValueError("memory probe native answer/history mismatch")
        if len(arm["native_steps"]) != len(arm["query_token_ids"]) + len(arm["answer_token_ids"]):
            raise ValueError("memory probe native step coverage mismatch")
        closure = dict(arm["closure"])
        digest = closure.pop("sha256")
        if content_digest(closure) != digest:
            raise ValueError("memory probe closure digest mismatch")
        if (
            arm["closure"]["logits"]["sha256"] != endpoint["logits"]["sha256"]
            or arm["closure"]["cursor"]["pending_token"] != endpoint["pending"]
        ):
            raise ValueError("memory probe endpoint readout binding mismatch")
        from .vm_payload_cache import PayloadRef, PayloadStore

        if vm_artifacts is None:
            raise ValueError("memory probe requires native VM artifact closure")
        payloads = PayloadStore(vm_artifacts.root.parent / "radix/cas")
        for readout in [arm["first_readout"], *arm["answer_readouts"], arm["endpoint_readout"]]:
            cut = artifacts.get(readout["cut_id"])
            payload = PayloadRef.from_dict(cut["logits"])
            if (
                readout["logits_sha256"] != payload.sha256
                or readout["pending_token_id"] != cut["pending"]
            ):
                raise ValueError("memory probe readout/cut mismatch")
            raw = payloads.hydrate(
                payload, native_consumer="hidden-memory-probe-verification"
            ).payload
            dtype = getattr(torch, payload.dtype.removeprefix("torch."))
            logits = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(payload.shape).float()[0]
            observable = value["spec"].get("observable")
            margin = (
                float(logits[observable["token_id"]] - logits[observable["against_token_id"]])
                if observable
                else None
            )
            if readout["raw_logit_margin"] != margin:
                raise ValueError("memory probe raw margin/cut mismatch")
            claimed = readout["top_tokens"]
            actual_scores = torch.topk(
                logits, min(value["spec"]["top_k"], logits.numel())
            ).values.tolist()
            # Equal-score vocabulary ties may have different order on CPU/CUDA.
            if [row["raw_logit"] for row in claimed] != actual_scores or len(
                {row["token_id"] for row in claimed}
            ) != len(actual_scores):
                raise ValueError("memory probe native top-k ranking mismatch")
            for token in readout["top_tokens"]:
                if float(logits[token["token_id"]]) != token["raw_logit"]:
                    raise ValueError("memory probe top-token logit/cut mismatch")
        if any(
            not math.isfinite(float(row["raw_logit"]))
            for readout in [arm["first_readout"], *arm["answer_readouts"], arm["endpoint_readout"]]
            for row in readout["top_tokens"]
        ):
            raise ValueError("memory probe non-finite native readout")
        input_tokens = arm["query_token_ids"] + arm["answer_token_ids"]
        for index, step in enumerate(arm["native_steps"]):
            cut = artifacts.get(step["cut"])
            if (
                cut["token_ids"] != parent["token_ids"] + input_tokens[: index + 1]
                or step["token_id"] != input_tokens[index]
                or step["next_token_id"] != cut["pending"]
            ):
                raise ValueError("memory probe step input authority mismatch")
            receipt = step["receipt_ref"]
            try:
                artifacts.get(receipt)
            except Exception:
                if vm_artifacts is None:
                    raise
                vm_artifacts.get(receipt)
    if value["effect"] != _compare(value["candidate"], value["native"]):
        raise ValueError("memory probe signed effect mismatch")
    if value["reinstall_effect"] != _compare(value["reinstall"], value["native"]):
        raise ValueError("memory probe reinstall effect mismatch")
    if not value["reinstall_effect"]["closure_exact"]:
        raise ValueError("memory probe reinstall closure mismatch")
    return True
