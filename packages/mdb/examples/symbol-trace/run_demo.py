"""Qualify a longitudinal symbol trace on a random tiny native Qwen3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SESSION_ID = "symbol-trace"
SYMBOL = "demo.carrier"
REPORT_SCHEMA = "mdb-symbol-trace-demo-v1"
NATIVE_TRANSITION_LIMIT = 256


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cut(project: Any, cut_id: str) -> dict[str, Any]:
    return project.cut_descriptor(SESSION_ID, cut_id)


def _slot_descriptor(project: Any, cut_id: str, slot: str) -> Any:
    payload = (_cut(project, cut_id).get("manifest") or {}).get("payload") or {}
    for item in payload.get("items", ()):
        key = (item.get("key") or {}).get("value")
        if key == slot:
            return item.get("value")
    return None


def _cut_observation(project: Any, cut_id: str) -> dict[str, Any]:
    descriptor = _cut(project, cut_id)
    manifest = descriptor["manifest"]
    return {
        "cut_id": cut_id,
        "parent_cut": descriptor.get("parent"),
        "boundary": descriptor.get("boundary"),
        "execution_point": descriptor.get("execution_point"),
        "surface": descriptor.get("surface"),
        "payload_descriptor": manifest.get("payload"),
    }


def _receipt_for(project: Any, cut_id: str, *, operation: str | None = None) -> Any:
    rows = []
    for receipt in project.receipts(SESSION_ID):
        body = receipt.to_dict()
        if body.get("result") != cut_id or body.get("status") == "refused":
            continue
        if operation is not None and body.get("operation") != operation:
            continue
        rows.append(receipt)
    if not rows:
        raise RuntimeError(f"no retained native receipt produced cut {cut_id}")
    return rows[-1]


def _retained_transition_rows(project: Any) -> int:
    """Count distinct transition rows that survived receipt deduplication."""

    return sum(
        _receipt_transition_count(receipt.to_dict())
        for receipt in project.receipts(SESSION_ID)
    )


def _receipt_transition_count(body: dict[str, Any]) -> int:
    """Count native transitions in a receipt, including macro micro-receipts.

    ``Session.step(granularity='token')`` deliberately stores one outer receipt
    and its native transition receipts below ``micro_receipts``.  Counting only
    the outer ``transitions`` field would therefore under-report physical work.
    """

    transitions = body.get("transitions")
    count = len(transitions) if isinstance(transitions, list) else 0
    micro_receipts = body.get("micro_receipts")
    if isinstance(micro_receipts, list):
        count += sum(
            _receipt_transition_count(item)
            for item in micro_receipts
            if isinstance(item, dict)
        )
    return count


def _physical_work(native_transitions: int) -> dict[str, int]:
    """Report physical native work, including replay executions."""

    return {
        "native_transitions": native_transitions,
        "interventions": 1,
        "bindings": 2,
    }


def _native_logit_endpoint(
    adapter: Any, debug: Any, *, output: Path, label: str
) -> dict[str, Any]:
    """Capture the public readout endpoint while the native session is live."""

    import numpy as np

    descriptor = debug.inspect("logits")["value"]
    value = adapter.session.read("logits")
    array = value.detach().cpu().numpy()
    vector = array.reshape(-1)
    artifact = output / "consumer-logits" / f"{label}.npy"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact, array, allow_pickle=False)
    return {
        "descriptor": descriptor,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "full_vector_sha256": descriptor.get("sha256"),
        "argmax": int(vector.argmax().item()),
        "artifact": {
            "path": str(artifact.relative_to(output)),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "_vector": vector,
    }


def _logit_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    delta = left["_vector"] - right["_vector"]
    return {
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "full_vector_equal": bool(np.array_equal(left["_vector"], right["_vector"])),
        "argmax_equal": left["argmax"] == right["argmax"],
    }


def _run_to_readout(adapter: Any) -> tuple[Any, int]:
    """Run one native token transaction through the full suffix to logits.

    The MDB debugger's transition step persists and re-captures every layer.
    That is useful for a layer-by-layer debugging session, but makes a full
    decoder suffix disproportionately expensive.  The public Saturn Session
    already exposes the same exact execution as one token-granularity
    transaction.  We use that native API, then let ``debug.checkpoint`` retain
    only the endpoint cut.
    """

    receipt = adapter.session.step(
        granularity="token", max_steps=NATIVE_TRANSITION_LIMIT
    )
    body = receipt.to_dict()
    if body.get("operation") != "step":
        raise RuntimeError(f"native token transaction returned {body.get('operation')!r}")
    transitions = _receipt_transition_count(body)
    if transitions <= 0:
        raise RuntimeError("native token transaction retained no native transitions")
    return receipt, transitions


def _mark_phase(
    timings: list[dict[str, Any]], started: float, phase: str, **details: Any
) -> None:
    """Emit a small flushed progress marker and retain it in the report."""

    row = {"phase": phase, "elapsed_s": round(time.perf_counter() - started, 6), **details}
    timings.append(row)
    print("[symbol-trace] " + json.dumps(row, sort_keys=True), flush=True)


def _clock(
    binding: Any,
    source_cut: str,
    reading: dict[str, Any],
    label: str,
    *,
    consumer: str,
) -> Any:
    from mdb.symbol_lifecycle import ClockWitness

    ref = "cas://sha256/" + source_cut
    family = str(reading.get("family") or "native")
    return ClockWitness(
        clock_id=label,
        kind="execution-point",
        domain="native/" + family,
        reading=reading,
        provenance=(ref,),
        validity_status="supported",
        validity_support=(ref,),
        lifetime={"kind": "bounded-suffix", "reset": "common-parent"},
        status="observed",
        metadata={
            "source_cut": source_cut,
            "model_execution": binding.execution,
            "consumer": consumer,
            "reset": "common-parent",
        },
    )


def _observation(source_cut: str, result_cut: str, project: Any) -> dict[str, Any]:
    source = _cut_observation(project, source_cut)
    result = _cut_observation(project, result_cut)
    return {
        "observable": "native-successor-state-descriptor",
        "source_cut": source_cut,
        "result_cut": result_cut,
        "source_boundary": source["boundary"],
        "result_boundary": result["boundary"],
        "source_payload_descriptor": source["payload_descriptor"],
        "result_payload_descriptor": result["payload_descriptor"],
        "changed_cut_identity": source_cut != result_cut,
    }


def _make_lifecycle(
    binding: Any, occurrence: Any, use: Any, *, execution_info: dict[str, Any]
) -> Any:
    from mdb.symbol_lifecycle import SymbolLifecycle

    return SymbolLifecycle.define(
        binding,
        definitions=(occurrence,),
        uses=(use,),
        mechanics_status="valid",
        instrument_status="valid",
        trend_status="not-assessed",
        terminal_status="not-assessed",
        metadata={
            "specimen": {
                "family": execution_info.get("family"),
                "device": (execution_info.get("device") or {}).get("type"),
                "dtype": execution_info.get("dtype"),
            },
            "consumer": use.consumer,
            "semantic_cot_observation": "not-assessed",
        },
    )


def _lifecycle_for(
    binding: Any,
    *,
    occurrence_id: str,
    use_id: str,
    source_cut: str,
    result_cut: str,
    branch: str,
    definition_branch: str,
    receipt: Any,
    intervention: str,
    project: Any,
) -> Any:
    from mdb.symbol_lifecycle import SymbolOccurrence, SymbolUse

    fingerprint = binding.to_dict()["fingerprint"]
    source_point = _cut(project, source_cut)["execution_point"]
    result_point = _cut(project, result_cut)["execution_point"]
    execution_info = dict(
        ((_cut(project, source_cut).get("manifest") or {}).get("execution") or {})
    )
    consumer = (_cut(project, source_cut).get("surface") or {}).get(
        "consumer", "native-suffix"
    )
    ref = "cas://sha256/"
    occurrence = SymbolOccurrence.for_binding(
        binding,
        occurrence_id=occurrence_id,
        session_id=SESSION_ID,
        source_cut=source_cut,
        locations=binding.locations,
        branch=definition_branch,
        execution=binding.execution,
        clock=_clock(
            binding, source_cut, source_point, f"{occurrence_id}-clock", consumer=consumer
        ),
        evidence_refs=(ref + source_cut,),
        status="observed",
        validity_status="supported",
        validity_support=(ref + source_cut,),
    )
    receipt_id = receipt.fingerprint
    use = SymbolUse(
        use_id,
        fingerprint,
        SESSION_ID,
        consumer,
        source_cut=source_cut,
        parent_cut=receipt.to_dict().get("parent"),
        result_cut=result_cut,
        branch=branch,
        execution=binding.execution,
        receipt=receipt_id,
        locations=binding.locations,
        # A consumer use is observed at its successor/result cut.  Keep the
        # clock's provenance and metadata aligned with that same retained cut;
        # the source cut remains the signed dataflow association below.
        clock=_clock(binding, result_cut, result_point, f"{use_id}-clock", consumer=consumer),
        signed_observations=(_observation(source_cut, result_cut, project),),
        evidence_refs=(ref + receipt_id,),
        status="observed",
        validity_status="supported",
        validity_support=(ref + receipt_id,),
        metadata={"intervention": intervention, "cot_observation": "not-assessed"},
    )
    return _make_lifecycle(binding, occurrence, use, execution_info=execution_info)


def _offline_reopen(root: Path, selected_cut: str) -> dict[str, Any]:
    """Reopen the metadata plane in a new process without model frameworks."""

    code = r'''
import importlib.abc
import json
import sys

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'diffusers', 'saturn', 'common'}:
            raise ImportError('offline check attempted forbidden import: ' + fullname)

sys.meta_path.insert(0, Guard())
from mdb import Project
from mdb.symbol_trace import trace_symbol, verify_symbol_trace

project = Project(sys.argv[1])
sid = sys.argv[2]
symbol = sys.argv[3]
selected_before = project.session(sid)['selected']
project.store.load = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('symbol trace hydrated a native payload')
)
trace = trace_symbol(project, sid, symbol)
verify_symbol_trace(trace)
selected_after = project.session(sid)['selected']
assert selected_before == selected_after
print(json.dumps({
    'frameworks_blocked': True,
    'native_payload_reads': 0,
    'trace_verified': True,
    'selected_before': selected_before,
    'selected_after': selected_after,
    'expected_selected': sys.argv[4],
}, sort_keys=True))
'''
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(root / "project"),
            SESSION_ID,
            SYMBOL,
            selected_cut,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"offline reopen failed: {result.stderr}\n{result.stdout}")
    return json.loads(result.stdout)


def qualify(
    native_adapter: Any, token_ids: list[int] | tuple[int, ...], root: str | Path
) -> dict[str, Any]:
    """Run the smallest native symbol-to-consumer-to-successor qualification.

    native_adapter is a caller-owned DecoderAdapter. The function creates a
    new Project directory and never downloads or constructs a model. The
    returned report deliberately records CoT observation as untested.
    """

    from saturn_pub import StateAddress
    from saturn_pub.symbols import SymbolBinding, SymbolTable

    from mdb import BenchSession, Project
    from mdb.symbol_trace import trace_symbol, verify_symbol_trace

    output = Path(root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    project_root = output / "project"
    started = time.perf_counter()
    timings: list[dict[str, Any]] = []
    _mark_phase(timings, started, "setup_start")
    native_session = native_adapter.session(tuple(int(item) for item in token_ids))

    with BenchSession.from_saturn_pub(
        native_session, project=Project(project_root), session_id=SESSION_ID
    ) as owner:
        project = owner.adapter.project
        adapter = owner.adapter
        debug = owner.debugger()
        main_branch = adapter.public.active
        execution_info = dict(adapter.session.adapter.execution)
        _mark_phase(
            timings,
            started,
            "adapter_init",
            family=execution_info.get("family"),
            model_id=adapter.session.adapter.model_identity,
            device=(execution_info.get("device") or {}).get("type"),
            dtype=execution_info.get("dtype"),
        )

        # Embed plus layer 0: the first exact symbol cut.
        prefix_result = debug.step(2)
        prefix_transitions = len(prefix_result.get("transitions", ())) or 2
        source_a = debug.checkpoint("source-a")["cut_id"]
        binding_a = SymbolBinding.bind(
            adapter.session,
            SYMBOL,
            (StateAddress("hidden", role="carrier"),),
            context="layer-0",
            context_slots=("tokens", "hidden"),
            evidence=("cas://sha256/" + source_a,),
            claim="Supplied carrier at the post-embedding/layer-0 native cut.",
        )

        # Next native layer: same supplied name, a distinct exact binding.
        layer_result = debug.step()
        layer_transitions = len(layer_result.get("transitions", ())) or 1
        source_b = debug.checkpoint("source-b")["cut_id"]
        binding_b = SymbolBinding.bind(
            adapter.session,
            SYMBOL,
            (StateAddress("hidden", role="carrier"),),
            context="layer-1",
            context_slots=("tokens", "hidden"),
            evidence=("cas://sha256/" + source_b,),
            claim="Supplied carrier at the next native layer cut.",
        )
        adapter.symbols = SymbolTable((binding_a, binding_b))
        adapter.symbol_context = "layer-1"
        debug.checkpoint("qualified-bindings")
        _mark_phase(
            timings,
            started,
            "bindings",
            source_a=source_a,
            source_b=source_b,
            bindings=2,
            native_transitions=prefix_transitions + layer_transitions,
        )

        # The causal swap and no-op control start from source_a. The only
        # expression is evaluated inside the native debugger worker, so the
        # hidden tensor crosses no Python boundary here.
        debug.restore(source_a)
        debug.fork("zero-carrier")
        debug.set_expression("hidden", 'zeros_like("hidden")')
        zero_commit = debug.commit()
        zero_candidate = zero_commit["cut_id"]
        _mark_phase(timings, started, "zero_suffix_start", branch="zero-carrier")
        _, zero_transitions = _run_to_readout(adapter)
        zero_result = debug.checkpoint("zero-successor")["cut_id"]
        zero_logits = _native_logit_endpoint(adapter, debug, output=output, label="zero")
        zero_receipt = _receipt_for(project, zero_result, operation="step")
        _mark_phase(
            timings,
            started,
            "zero_suffix",
            branch="zero-carrier",
            native_transitions=zero_transitions,
            result_cut=zero_result,
        )

        adapter.optional_operation("checkout", main_branch)
        debug.status()
        debug.restore(source_a)
        debug.fork("copy-control")
        # A native fork with no write is the copy/no-op control. Keeping it
        # intervention free makes the single zero-carrier write explicit in
        # the physical-work accounting.
        copy_candidate = debug.checkpoint("copy-parent")["cut_id"]
        _mark_phase(timings, started, "copy_suffix_start", branch="copy-control")
        _, copy_transitions = _run_to_readout(adapter)
        copy_result = debug.checkpoint("copy-successor")["cut_id"]
        copy_logits = _native_logit_endpoint(adapter, debug, output=output, label="copy")
        _mark_phase(
            timings,
            started,
            "copy_suffix",
            branch="copy-control",
            native_transitions=copy_transitions,
            result_cut=copy_result,
        )

        adapter.optional_operation("checkout", main_branch)
        debug.status()
        debug.restore(source_b)
        _mark_phase(timings, started, "main_suffix_start", branch=main_branch)
        _, main_transitions = _run_to_readout(adapter)
        main_result = debug.checkpoint("main-successor")["cut_id"]
        main_logits = _native_logit_endpoint(adapter, debug, output=output, label="main")
        main_receipt = _receipt_for(project, main_result, operation="step")
        _mark_phase(
            timings,
            started,
            "main_suffix",
            branch=main_branch,
            native_transitions=main_transitions,
            result_cut=main_result,
        )

        replay_receipt = debug.replay(source_b)
        debug.restore(source_b)
        _mark_phase(timings, started, "restored_suffix_start", branch=main_branch)
        _, restored_transitions = _run_to_readout(adapter)
        replay_result = debug.checkpoint("restored-successor")["cut_id"]
        debug.restore(main_result)
        selected_before_trace = project.session(SESSION_ID)["selected"]
        _mark_phase(
            timings,
            started,
            "replay_restore",
            replay_steps=int(replay_receipt.get("steps") or 0),
            replay_native_transitions=2 * int(replay_receipt.get("steps") or 0),
            restored_transitions=restored_transitions,
            result_cut=replay_result,
        )

        _mark_phase(timings, started, "lifecycles_start")
        lifecycle_a = _lifecycle_for(
            binding_a,
            occurrence_id="definition-a",
            use_id="use-zero-successor",
            source_cut=source_a,
            result_cut=zero_result,
            branch="zero-carrier",
            definition_branch=main_branch,
            receipt=zero_receipt,
            intervention="zero-carrier",
            project=project,
        )
        lifecycle_b = _lifecycle_for(
            binding_b,
            occurrence_id="definition-b",
            use_id="use-main-successor",
            source_cut=source_b,
            result_cut=main_result,
            branch=main_branch,
            definition_branch=main_branch,
            receipt=main_receipt,
            intervention="native-main",
            project=project,
        )
        saved_a = project.save_symbol_lifecycle(SESSION_ID, lifecycle_a)
        saved_b = project.save_symbol_lifecycle(SESSION_ID, lifecycle_b)
        _mark_phase(
            timings,
            started,
            "lifecycles",
            lifecycle_count=2,
            occurrence_count=2,
            use_count=2,
        )

        _mark_phase(timings, started, "trace_start")
        trace_all = trace_symbol(project, SESSION_ID, SYMBOL)
        trace_at_b = trace_symbol(project, SESSION_ID, SYMBOL, cut_id=source_b)
        trace_main = trace_symbol(project, SESSION_ID, SYMBOL, cut_id=main_result)
        trace_zero = trace_symbol(project, SESSION_ID, SYMBOL, branch="zero-carrier")
        for trace in (trace_all, trace_at_b, trace_main, trace_zero):
            verify_symbol_trace(trace)
        _mark_phase(
            timings,
            started,
            "trace",
            trace_count=4,
            binding_groups=len(trace_all["binding_groups"]),
        )

        assert len(trace_all["binding_groups"]) == 2
        assert binding_a.to_dict()["fingerprint"] != binding_b.to_dict()["fingerprint"]
        assert any(row.get("branch") == "zero-carrier" for row in trace_zero["uses"])
        assert any(row.get("branch") == main_branch for row in trace_main["uses"])
        assert all(row.get("branch") != "zero-carrier" for row in trace_main["uses"])
        assert trace_main["filtered_events"]
        assert _slot_descriptor(project, source_a, "hidden") == _slot_descriptor(
            project, copy_candidate, "hidden"
        )
        assert _slot_descriptor(project, source_a, "hidden") != _slot_descriptor(
            project, zero_candidate, "hidden"
        )
        assert replay_result == main_result
        assert replay_receipt.get("exact") is True
        assert selected_before_trace == main_result
        physical_transition_count = (
            prefix_transitions
            + layer_transitions
            + zero_transitions
            + copy_transitions
            + main_transitions
            + (2 * int(replay_receipt.get("steps") or 0))
            + restored_transitions
        )
        physical_work = _physical_work(physical_transition_count)
        zero_copy_delta = _logit_delta(zero_logits, copy_logits)
        copy_main_delta = _logit_delta(copy_logits, main_logits)
        zero_main_delta = _logit_delta(zero_logits, main_logits)
        if physical_work["native_transitions"] > NATIVE_TRANSITION_LIMIT:
            raise RuntimeError(
                "assay exceeded the declared native transition limit: "
                f"{physical_work['native_transitions']} > {NATIVE_TRANSITION_LIMIT}"
            )

        retained_receipts = project.receipts(SESSION_ID)
        raw = {
            "cuts": [_cut_observation(project, cut) for cut in project.cut_ids(SESSION_ID)],
            "receipts": [row.to_dict() for row in retained_receipts],
            "lifecycles": [saved_a, saved_b],
        }
        _write_json(output / "raw.json", raw)
        _write_json(output / "lifecycle-a.json", lifecycle_a.to_dict())
        _write_json(output / "lifecycle-b.json", lifecycle_b.to_dict())
        _write_json(output / "trace-all.json", trace_all)
        _write_json(output / "trace-at-source-b.json", trace_at_b)
        _write_json(output / "trace-at-main-result.json", trace_main)
        _write_json(output / "trace-zero-branch.json", trace_zero)
        consumer_endpoints = {
            name: {key: value for key, value in endpoint.items() if key != "_vector"}
            for name, endpoint in (
                ("zero", zero_logits),
                ("copy", copy_logits),
                ("main", main_logits),
            )
        }
        _write_json(output / "consumer-endpoints.json", consumer_endpoints)

        controls = {
            "zero": {
                "candidate_cut": zero_candidate,
                "result_cut": zero_result,
                "receipt": zero_receipt.fingerprint,
                "logits_artifact": zero_logits["artifact"],
                "logit_vector_sha256": zero_logits["full_vector_sha256"],
                "logit_argmax": zero_logits["argmax"],
            },
            "copy": {
                "kind": "native-copy/no-op",
                "candidate_cut": copy_candidate,
                "result_cut": copy_result,
                "hidden_descriptor_equal_to_parent": True,
                "logits_artifact": copy_logits["artifact"],
                "logit_vector_sha256": copy_logits["full_vector_sha256"],
                "logit_argmax": copy_logits["argmax"],
            },
            "main": {
                "source_cut": source_b,
                "result_cut": main_result,
                "logits_artifact": main_logits["artifact"],
                "logit_vector_sha256": main_logits["full_vector_sha256"],
                "logit_argmax": main_logits["argmax"],
            },
            "restored_successor_cut": replay_result,
            "one_step_replay_exact": bool(replay_receipt.get("exact"))
            and int(replay_receipt.get("steps") or 0) == 1,
            "full_restored_suffix_cut_equal": replay_result == main_result,
            "replay_steps": int(replay_receipt.get("steps") or 0),
            "consumer_comparison": {
                "same_parent_zero_vs_copy": zero_copy_delta,
                "native_prefix_copy_vs_main": copy_main_delta,
                "zero_vs_main": zero_main_delta,
                "endpoint": "logits",
                "generation_budget": 1,
                "prefix_mask": "full-prefix",
            },
        }
        report = {
            "schema": REPORT_SCHEMA,
            "specimen": {
                "family": execution_info.get("family"),
                "kind": "native-public",
                "device": (execution_info.get("device") or {}).get("type"),
                "dtype": execution_info.get("dtype"),
                "model_id": adapter.session.adapter.model_identity,
                "token_ids": list(token_ids),
            },
            "session_id": SESSION_ID,
            "symbol": SYMBOL,
            "physical_work": physical_work,
            "retained_transition_rows": _retained_transition_rows(project),
            "retained_receipt_count": len(retained_receipts),
            "physical_transition_breakdown": {
                "prefix": prefix_transitions,
                "binding_layer": layer_transitions,
                "zero": zero_transitions,
                "copy": copy_transitions,
                "main": main_transitions,
                "replay": 2 * int(replay_receipt.get("steps") or 0),
                "restored": restored_transitions,
            },
            "limits": {
                "native_transitions": NATIVE_TRANSITION_LIMIT,
                "logical_branches": 4,
                "logical_operations": NATIVE_TRANSITION_LIMIT,
            },
            "source_cuts": {"post-layer-0": source_a, "post-layer-1": source_b},
            "binding_fingerprints": [
                binding_a.to_dict()["fingerprint"],
                binding_b.to_dict()["fingerprint"],
            ],
            "lifecycle_ids": [saved_a["lifecycle_id"], saved_b["lifecycle_id"]],
            "controls": controls,
            "trace_fingerprints": {
                "all": trace_all["fingerprint"],
                "at_source_b": trace_at_b["fingerprint"],
                "at_main_result": trace_main["fingerprint"],
                "zero_branch": trace_zero["fingerprint"],
            },
            "trace_checks": {
                "multiple_bindings_preserved": len(trace_all["binding_groups"]) == 2,
                "sibling_excluded_by_main_cut": all(
                    row.get("branch") != "zero-carrier" for row in trace_main["uses"]
                ),
                "selected_pointer_unchanged": selected_before_trace == main_result,
                "native_payload_reads": 0,
            },
            "cot_observation": {
                "status": "not-assessed",
                "tested": False,
                "cot_refs": [],
                "reason": (
                    "This demo observes native successor state only; it does not decode "
                    "or qualify semantic CoT."
                ),
            },
            "statuses": {
                "mechanics": "valid",
                "instrument": "valid",
                "trend": "not-assessed",
                "terminal": "not-assessed",
            },
            "authority": "read-only-evidence-trace",
            "causal_effect": "not-established",
            "claim_boundary": (
                "The demo joins two exact qualified bindings to native successor uses and "
                "retained dataflow metadata. It establishes no semantic CoT role or causal effect."
            ),
            "phase_timings": timings,
        }

    _mark_phase(timings, started, "offline_reopen_start")
    offline = _offline_reopen(output, report["controls"]["main"]["result_cut"])
    _mark_phase(
        timings,
        started,
        "offline_reopen",
        frameworks_blocked=offline.get("frameworks_blocked"),
        native_payload_reads=offline.get("native_payload_reads"),
        trace_verified=offline.get("trace_verified"),
    )
    report["offline_reopen"] = offline
    report["phase_timings"] = timings
    _write_json(output / "phase-timings.json", timings)
    _write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="new output directory; project bytes stay below it")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--token-ids", default="1,2,3")
    args = parser.parse_args(argv)
    from saturn_pub.adapters.decoder import DecoderAdapter

    token_ids = [int(value) for value in args.token_ids.split(",") if value.strip()]
    report = qualify(DecoderAdapter.tiny("qwen3", seed=args.seed), token_ids, args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
