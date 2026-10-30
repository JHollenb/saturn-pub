"""Focused mechanics for the read-only longitudinal symbol trace."""

from __future__ import annotations

import copy
import json

import pytest
from saturn_pub import StateAddress
from saturn_pub.symbols import SymbolBinding, SymbolTable
from test_saturn_pub_adapter import opened

from mdb import Project
from mdb.cli import main as cli_main
from mdb.project_navigation import command as navigation_command
from mdb.symbol_lifecycle import SymbolLifecycle, SymbolOccurrence, SymbolUse
from mdb.symbol_trace import (
    SymbolTraceError,
    trace_symbol,
    verify_symbol_trace,
)


def _binding(adapter, name="trace.carrier"):
    return SymbolBinding.bind(
        adapter.session,
        name,
        (StateAddress("value"),),
        context="trace-context",
        context_slots=(),
        evidence=("fixture://trace-binding",),
        claim="Supplied carrier location; no semantic inference.",
    )


def _occurrence(binding, source, *, branch="main", cot_refs=(), occurrence_id="occurrence"):
    return SymbolOccurrence.for_binding(
        binding,
        occurrence_id=occurrence_id,
        session_id="experiment",
        source_cut=source,
        locations=binding.locations,
        branch=branch,
        execution=binding.execution,
        cot_refs=cot_refs,
        evidence_refs=(f"cas://sha256/{source}",),
        status="observed",
    )


def _linear_lifecycle(tmp_path, *, cot_refs=()):
    project, adapter, owner, debug = opened(tmp_path)
    binding = _binding(adapter)
    adapter.symbols = SymbolTable((binding,))
    adapter.symbol_context = "trace-context"
    source = adapter.checkpoint("trace-source")["cut_id"]
    debug.step()
    result = adapter.checkpoint("trace-result")["cut_id"]
    receipt = next(
        item
        for item in project.receipts("experiment")
        if item.to_dict().get("result") == result and item.to_dict().get("operation") == "continue"
    )
    occurrence = _occurrence(binding, source, cot_refs=cot_refs)
    use = SymbolUse(
        "use-1",
        binding.to_dict()["fingerprint"],
        "experiment",
        "counter-native",
        source_cut=source,
        parent_cut=source,
        result_cut=result,
        branch="main",
        execution=binding.execution,
        receipt=receipt.fingerprint,
        locations=binding.locations,
        signed_observations=({"observable": "value-change", "delta": 1},),
        evidence_refs=(f"cas://sha256/{receipt.fingerprint}",),
        status="observed",
    )
    lifecycle = SymbolLifecycle.define(binding, definitions=(occurrence,), uses=(use,))
    project.save_symbol_lifecycle("experiment", lifecycle)
    return project, adapter, owner, lifecycle, source, result


def test_trace_groups_later_exact_bindings_without_merging_native_clocks(tmp_path):
    # The clock-organism fixture deliberately rebinds the same supplied name at
    # two native cuts.  Each lifecycle is exact; the trace is the longitudinal
    # index that joins the supplied name while keeping both identities.
    from test_symbol_lifecycle_clocks import _open_two_bank

    project, _, owner, _, records, before, after = _open_two_bank(tmp_path)
    try:
        trace = trace_symbol(project, "experiment", "organism.a")
        assert trace["schema"] == "mdb-symbol-trace-v1"
        assert len(trace["binding_groups"]) == 2
        assert len(trace["occurrences"]) == 2
        assert {
            row["binding_fingerprint"] for row in trace["occurrences"]
        } == {
            records["a0"].binding_fingerprint,
            records["a2"].binding_fingerprint,
        }
        assert {row["source_cut"] for row in trace["occurrences"]} == {before, after}
        assert trace["causal_effect"] == "not-established"
        assert trace["authority"] == "read-only-evidence-trace"
        assert verify_symbol_trace(trace)["passed"] is True
    finally:
        owner.close()


def test_duplicate_use_ids_keep_distinct_binding_links(tmp_path):
    # Native consumers may use a local label such as "read" repeatedly.  The
    # sealed event fingerprint, rather than that label, is the trace identity.
    from test_symbol_lifecycle_clocks import _open_two_bank

    project, _, owner, _, records, before, after = _open_two_bank(tmp_path)
    try:
        receipts = {
            item.to_dict()["result"]: item
            for item in project.receipts("experiment")
            if item.to_dict().get("operation") == "continue"
        }
        middle = next(result for result in receipts if result not in {before, after})
        for key, source, result in (("a0", before, middle), ("a2", after, after)):
            binding = records[key].binding
            # Reuse the exact retained definition for each binding so the
            # additional lifecycle snapshot contributes a use without making
            # the source definition itself ambiguous.
            occurrence = records[key].definitions[0]
            receipt = receipts[result]
            receipt_body = receipt.to_dict()
            use = SymbolUse(
                "read",
                binding.to_dict()["fingerprint"],
                "experiment",
                "two-bank-native",
                source_cut=source,
                parent_cut=receipt_body["parent"],
                result_cut=result,
                branch="main",
                execution=binding.execution,
                receipt=receipt.fingerprint,
                locations=binding.locations,
                signed_observations=(("read", key),),
                evidence_refs=(f"cas://sha256/{receipt.fingerprint}",),
                status="observed",
            )
            project.save_symbol_lifecycle(
                "experiment", SymbolLifecycle.define(binding, definitions=(occurrence,), uses=(use,))
            )

        trace = trace_symbol(project, "experiment", "organism.a")
        assert len(trace["uses"]) == 2
        assert {row["use_id"] for row in trace["uses"]} == {"read"}
        assert len({row["event_fingerprint"] for row in trace["uses"]}) == 2
        assert len({row["definition_link"]["definition_fingerprint"] for row in trace["uses"]}) == 2
        assert {row["use_fingerprint"] for row in trace["links"]} == {
            row["event_fingerprint"] for row in trace["uses"]
        }
    finally:
        owner.close()


def test_cut_and_branch_filters_exclude_sibling_events_and_keep_unknown_links(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    try:
        binding = _binding(adapter)
        adapter.symbols = SymbolTable((binding,))
        adapter.symbol_context = "trace-context"
        source = adapter.checkpoint("root")["cut_id"]
        debug.fork("sibling")
        debug.step()
        sibling = adapter.checkpoint("sibling-result")["cut_id"]
        receipt = next(
            item
            for item in project.receipts("experiment")
            if item.to_dict().get("result") == sibling and item.to_dict().get("operation") == "continue"
        )
        receipt_body = receipt.to_dict()
        occurrence = _occurrence(binding, source, occurrence_id="root-definition")
        use = SymbolUse(
            "sibling-use",
            binding.to_dict()["fingerprint"],
            "experiment",
            "counter-native",
            source_cut=source,
            parent_cut=receipt_body["parent"],
            result_cut=sibling,
            branch="sibling",
            execution=binding.execution,
            receipt=receipt.fingerprint,
            locations=binding.locations,
            signed_observations=({"observable": "sibling-result", "value": 1},),
            evidence_refs=(f"cas://sha256/{receipt.fingerprint}",),
            status="observed",
        )
        lifecycle = SymbolLifecycle.define(binding, definitions=(occurrence,), uses=(use,))
        project.save_symbol_lifecycle("experiment", lifecycle)

        root_trace = trace_symbol(project, "experiment", "trace.carrier", cut_id=source)
        assert [row["occurrence_id"] for row in root_trace["occurrences"]] == ["root-definition"]
        assert root_trace["uses"] == []
        assert root_trace["coverage"]["filtered_events"] == 1

        sibling_trace = trace_symbol(project, "experiment", "trace.carrier", branch="sibling")
        assert sibling_trace["occurrences"] == []
        assert [row["use_id"] for row in sibling_trace["uses"]] == ["sibling-use"]
        assert sibling_trace["links"][0]["status"] == "unknown"
        assert sibling_trace["links"][0]["reason"] == "no-exact-source-cut-and-binding-definition"
    finally:
        owner.close()


def test_duplicate_lifecycle_snapshots_deduplicate_events_and_retain_invalidations(tmp_path):
    project, _, owner, lifecycle, _, _ = _linear_lifecycle(tmp_path)
    try:
        original_id = lifecycle.fingerprint
        invalidated = lifecycle.invalidate(
            scope="occurrence:occurrence",
            reason="supplied invalidation retained as a later snapshot",
            evidence_refs=("policy://trace-test",),
        )
        later_id = project.save_symbol_lifecycle("experiment", invalidated)["lifecycle_id"]
        trace = trace_symbol(project, "experiment", "trace.carrier")
        assert len(trace["occurrences"]) == 1
        assert trace["occurrences"][0]["snapshot_count"] == 2
        assert set(trace["occurrences"][0]["lifecycle_ids"]) == {original_id, later_id}
        assert len(trace["invalidations"]) == 1
        assert trace["invalidations"][0]["lifecycle_ids"] == [later_id]
        assert {row["lifecycle_id"] for row in trace["lifecycle_snapshots"]} == {
            original_id,
            later_id,
        }
        # Loading the original sealed snapshot still returns the pre-invalidation body.
        assert project.symbol_lifecycle("experiment", original_id)["lifecycle"]["invalidations"] == []
    finally:
        owner.close()


def test_mismatched_cot_is_locator_unknown_and_never_causal(tmp_path):
    cot = {
        "schema": "mdb-cot-register-observation-v1",
        "source_cut": "cas://sha256/" + "f" * 64,
        "semantic_cot_qualified": True,
        "causal_dependency_tested": True,
        "decoded_candidates": [{"token_id": 1, "score": 0.9}],
    }
    project, _, owner, _, _, _ = _linear_lifecycle(tmp_path, cot_refs=(cot,))
    try:
        trace = trace_symbol(project, "experiment", "trace.carrier")
        assert trace["causal_effect"] == "not-established"
        assert trace["cot_refs"][0]["locator_status"] == "unknown"
        assert trace["cot_refs"][0]["semantic_status"] == "supplied-unverified"
        assert trace["cot_refs"][0]["raw_payload"]["semantic_cot_qualified"] is True
        assert trace["cot_refs"][0]["raw_payload"]["causal_dependency_tested"] is True
        assert trace["cot_refs"][0]["semantic_cot_qualified"] is False
        assert trace["cot_refs"][0]["causal_dependency_tested"] is False
        assert trace["cot_refs"][0]["authority"] == "locator-only"
        assert trace["occurrences"][0]["cot_refs"][0]["causal_effect"] == "not-established"
    finally:
        owner.close()


def test_trace_is_metadata_only_reopens_reproducibly_and_verifies_seal(tmp_path, monkeypatch):
    project, _, owner, _, _, _ = _linear_lifecycle(tmp_path)
    try:
        monkeypatch.setattr(project.store, "load", lambda *args, **kwargs: pytest.fail("payload hydration"))
        first = trace_symbol(project, "experiment", "trace.carrier")
        assert first["chronology"]["cuts"]
        assert first["signed_observations"][0]["observation"]["observable"] == "value-change"
        assert first["evidence_refs"]
        assert first["links"][0]["status"] == "linked"
        owner.close()
        reopened = Project(project.root)
        second = trace_symbol(reopened, "experiment", "trace.carrier")
        assert second == first
        assert verify_symbol_trace(second)["passed"] is True
        tampered = copy.deepcopy(second)
        tampered["causal_effect"] = "causal"
        with pytest.raises(SymbolTraceError, match="fingerprint"):
            verify_symbol_trace(tampered)
    finally:
        if getattr(owner, "_closed", False) is False:
            owner.close()


def test_trace_output_is_json_and_mermaid_uses_sealed_graph(tmp_path):
    project, _, owner, _, _, _ = _linear_lifecycle(tmp_path)
    try:
        trace = trace_symbol(project, "experiment", "trace.carrier")
        encoded = json.dumps(trace, sort_keys=True)
        assert json.loads(encoded)["fingerprint"] == trace["fingerprint"]
        graph = trace["graph"]
        assert any(node["kind"] == "supplied-symbol" for node in graph["nodes"])
        assert any(node["kind"] == "cut" for node in graph["nodes"])
        assert any(edge["relation"] == "supplied-name-association" for edge in graph["edges"])
        assert any(edge["relation"] == "native-parent" for edge in graph["edges"])
        from mdb.symbol_trace import trace_mermaid

        mermaid = trace_mermaid(trace)
        assert mermaid.startswith("graph LR\n")
        assert "joined_slice" in mermaid
    finally:
        owner.close()


def test_cli_and_navigation_symbol_history_apply_cut_and_branch_filters(tmp_path, capsys):
    from test_symbol_lifecycle_project import recorded

    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    try:
        project.save_symbol_lifecycle("experiment", lifecycle)
        assert cli_main(["symbol-trace", str(project.root), "counter.value"]) == 0
        cli_trace = json.loads(capsys.readouterr().out)
        assert cli_trace["schema"] == "mdb-symbol-trace-v1"
        assert cli_trace["symbol"] == "counter.value"
        assert verify_symbol_trace(cli_trace)["passed"] is True

        selected = navigation_command(
            project,
            "experiment",
            ["symbol-history", "counter.value", "--cut", source, "--branch", "main"],
        )
        assert selected["filters"] == {"branch": "main", "cut_id": source}
        assert [row["branch"] for row in selected["occurrences"]] == ["main"]
        assert selected["uses"] == []  # result cut is outside the selected ancestry

        excluded = navigation_command(
            project,
            "experiment",
            ["symbol-history", "counter.value", "--branch", "other"],
        )
        assert excluded["occurrences"] == []
        assert excluded["uses"] == []
        assert excluded["coverage"]["filtered_events"] == 2
    finally:
        owner.close()


class _MetadataTraceProject:
    """Minimal retained-descriptor surface for ancestry frontier mechanics."""

    def __init__(self, descriptors):
        self.descriptors = descriptors
        self._session = {
            "identity": json.dumps("metadata-trace-model"),
            "execution": json.dumps({"adapter": "metadata-trace"}),
            "selected": next(iter(descriptors)),
        }

    def session(self, session_id):
        assert session_id == "experiment"
        return self._session

    def cut_ids(self, session_id):
        assert session_id == "experiment"
        return list(self.descriptors)

    def cut_descriptor(self, session_id, cut_id):
        assert session_id == "experiment"
        return self.descriptors[cut_id]

    def receipts(self, session_id):
        assert session_id == "experiment"
        return []

    def symbol_lifecycles(self, session_id, *, symbol=None):
        assert session_id == "experiment"
        return []


def test_long_and_cyclic_ancestry_stays_iterative_and_exposes_frontier():
    descriptors = {}
    previous = None
    for index in range(1101):
        identifier = f"chain-{index}"
        descriptors[identifier] = {
            "fingerprint": identifier,
            "parent": previous,
            "boundary": identifier,
            "execution_point": {"logical_step": index, "index": index},
            "surface": {"state_schema": "metadata-trace-v1", "consumer": "trace"},
        }
        previous = identifier
    descriptors["chain-0"]["parent"] = "missing-parent"
    descriptors.update(
        {
            "cycle-a": {
                "fingerprint": "cycle-a",
                "parent": "cycle-b",
                "boundary": "cycle-a",
                "execution_point": {"logical_step": 2000, "index": 2000},
                "surface": {"state_schema": "metadata-trace-v1", "consumer": "trace"},
            },
            "cycle-b": {
                "fingerprint": "cycle-b",
                "parent": "cycle-a",
                "boundary": "cycle-b",
                "execution_point": {"logical_step": 2001, "index": 2001},
                "surface": {"state_schema": "metadata-trace-v1", "consumer": "trace"},
            },
        }
    )
    project = _MetadataTraceProject(descriptors)

    long_trace = trace_symbol(project, "experiment", "counter.value", cut_id="chain-1100")
    assert len(long_trace["chronology"]["selected_ancestry"]) == 1102
    assert any(item["kind"] == "unretained-cut" for item in long_trace["frontier"])
    assert verify_symbol_trace(long_trace)["passed"] is True

    cyclic_trace = trace_symbol(project, "experiment", "counter.value", cut_id="cycle-a")
    assert any(item["kind"] == "cut-cycle" for item in cyclic_trace["frontier"])
    assert verify_symbol_trace(cyclic_trace)["passed"] is True
