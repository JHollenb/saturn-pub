"""CPU fixture coverage for the native symbol lifecycle assay."""

from __future__ import annotations

from argparse import Namespace

import pytest
import worker

from mdb import bench_loading
from mdb.ar.model import ArModelSpec
from mdb.bench_loading import BenchModelSpec


class _FixtureTokenizer:
    def __init__(self) -> None:
        self._words: dict[str, int] = {}

    def __call__(self, text, *, add_special_tokens=False, return_tensors=None):
        del add_special_tokens
        ids = []
        for word in str(text).split():
            ids.append(self._words.setdefault(word, len(self._words) + 1))
        if return_tensors is not None:
            import torch

            return type("Encoded", (), {"input_ids": torch.tensor([ids])})()
        return {"input_ids": ids}

    def decode(self, ids):
        reverse = {value: key for key, value in self._words.items()}
        return reverse[int(ids[0])] if ids else ""


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_native_tiny_cpu_lifecycle_roundtrip(monkeypatch, tmp_path):
    """Exercise real Session/Project joins with two KV heads and two rows."""

    adapter_class = worker._memory_adapter_class()
    fixture_adapter = adapter_class.tiny("qwen2", seed=113, device="cpu", granularity="operation")
    assert fixture_adapter.num_kv_heads >= 2
    tokenizer = _FixtureTokenizer()
    spec = BenchModelSpec("fixture-qwen2", "fixture/qwen2", "fixture", "qwen2")
    monkeypatch.setattr(bench_loading, "resolve_bench_model", lambda name: spec)
    monkeypatch.setattr(
        bench_loading,
        "load_bench_model",
        lambda name, device="cuda": (
            ArModelSpec(spec.registry_name, spec.registry_name, spec.model_id,
                        spec.revision, spec.family), fixture_adapter.model, tokenizer),
    )

    args = Namespace(
        model="fixture-qwen2",
        root=str(tmp_path / "session"),
        project_root=str(tmp_path / "project"),
        export_root=str(tmp_path / "export"),
        device="cpu",
        tiny_fixture=True,
    )
    report = worker.run_native(args)

    assert report["mechanics_status"] == "valid"
    assert report["instrument_status"] == "valid"
    assert report["trend_status"] == "observations-only"
    assert report["terminal_status"] == "not-assessed"
    assert report["exported_project"]["exact_lifecycle"] is True
    assert report["exported_project"]["lifecycle"]["definitions"] == 1
    assert report["exported_project"]["lifecycle"]["uses"] == 7
    assert report["native_project_archive"]["bytes"] > 0
    assert len(report["interventions"]) == 7
    assert {row["name"] for row in report["interventions"]} >= {
        "matched_source_donor",
        "wrong_row_control",
        "parent_copy",
    }

    members = report["source_coalition"]["members"]
    assert len(members) > 1
    assert [row["selector"] for row in members] == [
        [1, 0, 0, 0],
        [1, 0, 0, 1],
    ]
    assert len(report["source_coalition"]["locations"]) == 8
    assert all(len(row["selector"]) == 4 for row in report["source_coalition"]["locations"])

    uses = {row["use_id"]: row for row in report["symbol_uses"]}
    for arm in report["interventions"]:
        probe = arm["probe"]
        use = uses[f"use:{arm['name']}"]
        assert use["parent_cut"] == probe["candidate_receipt_parent_cut"]
        assert probe["baseline_margin"]["status"] == "completed"
        assert probe["candidate_margin"]["status"] == "completed"
        assert "Berlin" in probe["baseline_margin"]["answer_logits"]
        assert probe["exact_candidate_input"] is True

    assert report["lifecycle"]["join"]["listed_count"] == 1
    # symbols_at returns the matching occurrence once; its eight source
    # locations are retained inside that occurrence rather than expanded.
    assert report["lifecycle"]["join"]["located_count"] == 1
    assert report["spectral_address"]["mapping_status"].startswith("supplied-native-exact")
    assert report["spectral_queries"]["correct"]["result"]["selected"] is not None
    assert report["spectral_queries"]["wrong_instance"]["result"]["selected"] is None
    assert report["spectral_queries"]["wrong_role"]["result"]["selected"] is None
    assert report["vm_bundle"]["model_identity"]["id"] == report["model_identity"]
    custody = report["vm_descriptor_custody"]
    assert len(custody["files"]) >= 4
    assert all((tmp_path / "project" / row["path"]).is_file() for row in custody["files"])
    assert report["native_budget"]["counted_operations"] <= report["native_budget"]["max_operations"]
    assert report["native_budget"]["shared_baseline"] is True
    assert report["native_budget"]["actual_native_transitions"] == 8 * (fixture_adapter.layers + 5)
    assert report["native_budget"]["actual_act_operations"] == 11
