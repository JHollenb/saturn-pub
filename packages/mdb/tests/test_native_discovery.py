"""Mechanics: no pretrained downloads, semantic labels, patches, or local research runs."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdb.native_discovery import (
    DiscoveryError,
    checked,
    load_corpus,
    prepare_corpus,
    read_arrays,
    save_arrays,
    save_json,
    sealed,
)


def documents():
    return [
        {"id": "a", "text": "Clouds gather above a quiet mountain lake."},
        {"id": "b", "text": "A compiler turns source code into executable instructions."},
        {"id": "c", "text": "Fresh bread cools beside the open kitchen window."},
        {"id": "d", "text": "The telescope records distant stars during winter nights."},
        {"id": "e", "text": "A gardener plants flowers along the stone footpath."},
        {"id": "f", "text": "The library lends books and preserves historical manuscripts."},
    ]


def test_corpus_rejects_authored_answers_and_source_spans():
    for name in ("expected_answer", "source_span", "roles", "label"):
        rows = documents()
        rows[0][name] = "temperature"
        with pytest.raises(DiscoveryError, match="no labels or spans"):
            prepare_corpus(rows)


def test_grouping_and_dedup_precede_split_and_are_repeatable(tmp_path):
    rows = documents()
    rows.extend([{**rows[0], "id": "duplicate"},
                 {"id": "other", "group": "b", "text": "Programs allocate temporary storage for intermediate results."}])
    result = prepare_corpus(rows, seed=7)
    assert result == prepare_corpus(list(reversed(rows)), seed=7)
    assert len(result["deduplicated"]) == 1
    groups = {}
    for row in result["documents"]:
        assert groups.setdefault(row["group"], row["split"]) == row["split"]
    save_json(tmp_path / "corpus.json", result)
    assert load_corpus(tmp_path / "corpus.json") == result
    result["documents"][0]["text"] += " changed"
    save_json(tmp_path / "changed.json", result)
    with pytest.raises(DiscoveryError, match="fingerprint"):
        load_corpus(tmp_path / "changed.json")


def test_safe_arrays_are_immutable_and_hash_bound(tmp_path):
    descriptor = save_arrays(tmp_path / "a.npz", state=np.ones((2, 3), np.float32))
    assert read_arrays(tmp_path, descriptor)["state"].shape == (2, 3)
    with pytest.raises(FileExistsError):
        save_arrays(tmp_path / "a.npz", state=np.ones((2, 3)))
    (tmp_path / "a.npz").write_bytes(b"changed")
    with pytest.raises(DiscoveryError, match="changed"):
        read_arrays(tmp_path, descriptor)
    with pytest.raises(DiscoveryError, match="non-object"):
        save_arrays(tmp_path / "objects.npz", state=np.array([object()]))


def tiny_model():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=64, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=128)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval().requires_grad_(False)


class TinyTokenizer:
    def __call__(self, text, *, max_length=16, return_offsets_mapping=False, **kwargs):
        import re
        matches = list(re.finditer(r"\S+", text))[:max_length]
        result = {"input_ids": [int(hashlib.sha256(m.group().encode()).hexdigest()[:8], 16) % 63
                                for m in matches]}
        if return_offsets_mapping:
            result["offset_mapping"] = [m.span() for m in matches]
        return result

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


def test_native_capture_is_passive_and_cleans_up_hooks():
    import torch

    from mdb.native_discovery_capture import capture
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False, output_attentions=True).logits.clone()
    # Transformers lazily installs its own output-capture hooks on first use.
    hooks = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    arrays = capture(model, ids, boundaries=(0, 1, 2), neuron_budget=4)
    with torch.inference_mode():
        actual = model(ids, use_cache=False).logits
    assert torch.equal(expected, actual)
    assert arrays["states"].shape == (3, 4, 8)
    assert arrays["routes"].shape == (2, 2, 4, 4)
    assert arrays["neurons"].shape == (2, 4, 4)
    assert np.allclose(arrays["routes"].sum(-1), 1)
    assert hooks == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    model.train()
    with pytest.raises(DiscoveryError, match="evaluation"):
        capture(model, ids, boundaries=(0, 2))


def test_forecast_never_uses_later_native_states():
    import torch

    from mdb.native_discovery_learning import StatePredictor, forecast_arrays
    model = StatePredictor(width=8, features=4, heads=2, value_width=4,
                           stages=2, kind="sparse_consumer")
    with torch.no_grad():
        model.spans[0].local_write.bias.fill_(1)
        model.spans[1].local_write.bias.fill_(2)
    frontier = np.zeros((3, 8), np.float32)
    prediction = forecast_arrays(model, frontier)
    np.testing.assert_allclose(prediction["state"][0], 1)
    np.testing.assert_allclose(prediction["state"][1], 3)
    assert np.all(np.triu(prediction["routes"], 1) == 0)


def test_fitter_refuses_heldout_rows_before_any_training(tmp_path):
    from mdb.native_discovery_learning import fit_predictors
    record = sealed({"split": "heldout", "binding": {"id": "a"}, "boundaries": [0, 1]})
    with pytest.raises(DiscoveryError, match="split"):
        fit_predictors([(record, {})], [(record, {})], root=tmp_path / "fit",
                       binding={"id": "a"}, boundaries=(0, 1), steps=1)
    assert not (tmp_path / "fit").exists()


def test_full_pipeline_seals_future_and_exports_real_symbol_records(tmp_path):
    from mdb.native_discovery import verify_run
    from mdb.native_discovery_learning import load_predictors, load_snapshot
    from mdb.native_discovery_run import run_on_model
    from mdb.symbol_lifecycle import SymbolLifecycle
    corpus = prepare_corpus(documents(), seed=19)
    save_json(tmp_path / "corpus.json", corpus)
    root = tmp_path / "analysis"
    model = tiny_model()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    result = run_on_model(model, TinyTokenizer(), corpus=tmp_path / "corpus.json",
        out=root, binding={"model_identity": "tiny-random-mechanics", "execution": {"device": "cpu", "dtype": "fp32"}},
        max_tokens=12, stages=2, features=4, steps=2, interval_steps=1)
    assert result["interventions"] == 0
    import torch
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    from mdb.knowledge_interchange import CONTEXT, RESOLVED, VALUE, KnowledgeQuery
    from mdb.native_discovery_export import index_observations, load_catalog_bank
    bank = load_catalog_bank(root / "knowledge-bank.json")
    address = bank.records[0].logical_tuple.address
    assert bank.resolve(KnowledgeQuery(address, kind=CONTEXT)).status == RESOLVED
    assert bank.resolve(KnowledgeQuery(address, kind=VALUE)).status == "context-as-value"
    report = json.loads((root / "report.json").read_text())
    comparisons = sum(a["output_consumer_comparisons"] for a in report["work"]["training"].values())
    assert comparisons > 0
    assert report["work"]["paid_last_token_norm_head_calls"] == 6 * len(report["heldout"]) + 2 * comparisons
    for row in report["heldout"]:
        assert set(row["native_output_consumer"]) == {"linear_features", "sparse_reconstruction", "sparse_consumer"}
        assert row["mean_write"]
    verification = verify_run(root)
    assert verification["mechanics_status"] == "valid"
    # Additive position-robust metrics and mandatory trivial route baselines.
    for row in report["heldout"]:
        assert len(row["position_split_mean_write"]) == len(row["mean_write"])
        assert "position_0_energy_fraction" in row["mean_write"][0]
        assert {"position_0", "previous_token", "uniform"} <= set(row["route_top1_trivial_baselines"][0])
        for spans in row["arms"].values():
            assert "centered_relative_mse" in spans[0]["state"]
            assert spans[0]["source_top1"]["baselines"]["position_0"]["top_source_agreement"] >= 0
            assert spans[0]["source_top1"]["nominator"]["top_source_agreement"] == spans[0]["source_top1_agreement"]
    assert report["role_candidates_nonempty"] <= report["role_candidates"]
    visible = json.loads((root / "visible-inputs.json").read_text())
    assert all(v["longest_cross_split_shared_prefix"] < 8 for v in visible)
    projection = index_observations(root, tmp_path / "index-input")
    assert projection["observations"] == 15
    assert len((tmp_path / "index-input/observations.jsonl").read_text().splitlines()) == 15
    with pytest.raises(DiscoveryError, match="outside immutable"):
        index_observations(root, root / "index")
    metadata, models = load_predictors(root / "predictor")
    assert len(models) == 3
    assert not set(metadata["fitted_document_ids"]) & {
        r["id"] for r in corpus["documents"] if r["split"] == "heldout"}
    for path in (root / "predictor").glob("*/candidates/*"):
        snapshot = load_snapshot(path)
        snapshot.verify()
    for path in (root / "symbols").glob("*/lifecycles.json"):
        for row in json.loads(path.read_text()):
            lifecycle = SymbolLifecycle.from_dict(row)
            assert lifecycle.binding.symbol.startswith("component_")
            assert lifecycle.uses
            assert lifecycle.authority["causal_effect"] == "not-established"
    for forecast in (root / "forecasts").iterdir():
        events = [json.loads(p.read_text()) for p in sorted((forecast / "events").glob("*.json"))]
        kinds = [e["kind"] for e in events]
        assert kinds.index("globally_sealed") < kinds.index("observation_persisted")
    report = json.loads((root / "report.json").read_text())
    assert report["work"]["physical_document_forwards"] == len(corpus["documents"])
    assert report["bindings"] > 0
    manifest = json.loads((root / "manifest.json").read_text())
    changed = copy.deepcopy(manifest)
    changed["files"]["report.json"] = "0" * 64
    checked(manifest)
    assert changed != manifest
    (root / "report.json").write_text("{}")
    with pytest.raises(DiscoveryError, match="artifact changed"):
        verify_run(root)


def test_cli_help_does_not_import_model_frameworks():
    code = """import sys
from mdb.native_discovery import main
try: main(['--help'])
except SystemExit: pass
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})


def test_cross_split_shared_token_prefixes_are_refused():
    from mdb.native_discovery import cross_split_prefixes
    base = list(range(20))
    assert cross_split_prefixes([("discovery", base), ("heldout", [99] + base[1:])]) == [0, 0]
    assert cross_split_prefixes([("discovery", base), ("discovery", base)]) == [0, 0]
    assert cross_split_prefixes([("discovery", base), ("heldout", base[:7] + [99] * 13)]) == [7, 7]
    with pytest.raises(DiscoveryError, match="first 8 tokens"):
        cross_split_prefixes([("discovery", base), ("heldout", base[:8] + [99] * 12)])
    with pytest.raises(DiscoveryError, match="share"):
        cross_split_prefixes([("discovery", base), ("development", base)])
    with pytest.raises(DiscoveryError, match="share"):  # a whole shorter window prefixing a longer one
        cross_split_prefixes([("discovery", base), ("heldout", base[:3])])
    assert cross_split_prefixes([("discovery", base), ("heldout", base[:3] + [99])], refuse_at=5) == [3, 3]


def test_run_refuses_shared_visible_prefix_before_capture(tmp_path):
    from mdb.native_discovery_run import run_on_model
    shared = "alpha beta gamma delta epsilon zeta eta theta"
    rows = [{"id": k, "text": shared + " " + t} for k, t in
            (("a", "one two three"), ("b", "four five six"), ("c", "seven eight nine"),
             ("d", "ten eleven twelve"), ("e", "thirteen fourteen fifteen"))]
    rows = [{**r, "text": r["text"] + " " + " ".join(r["id"] * (i + 2) for i in range(6))} for r in rows]
    corpus = prepare_corpus(rows, seed=3, near_duplicate=1.0)
    save_json(tmp_path / "corpus.json", corpus)
    with pytest.raises(DiscoveryError, match="share their first"):
        run_on_model(tiny_model(), TinyTokenizer(), corpus=tmp_path / "corpus.json", out=tmp_path / "run",
                     binding={"model_identity": "tiny", "execution": {"device": "cpu"}},
                     max_tokens=12, stages=2, features=4, steps=1, interval_steps=1)
    assert not (tmp_path / "run" / "traces").exists()


def test_corpus_extractor_groups_byte_identical_files_into_one_split(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "native_corpus", Path(__file__).parents[1] / "experiments/native_discovery/corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(0)
    vocabulary = [f"w{i}" for i in range(4000)]

    def text():
        return "\n".join(" ".join(rng.choice(vocabulary, 12)) for _ in range(220))

    docs, code = tmp_path / "src/packages/mdb/docs", tmp_path / "src/src/saturn_pub"
    docs.mkdir(parents=True)
    code.mkdir(parents=True)
    duplicate = text()
    (docs / "copy_a.md").write_text(duplicate)
    (code / "copy_b.py").write_text(duplicate)
    for i in range(10):
        (docs / f"doc{i}.md").write_text(text())
    out = tmp_path / "corpus.json"
    module.build(tmp_path / "src", out, count=12, seed=5)
    corpus = load_corpus(out)
    rows = {r["source"]: r for r in corpus["documents"]}
    a, b = rows["packages/mdb/docs/copy_a.md"], rows["src/saturn_pub/copy_b.py"]
    assert a["text"] != b["text"]  # different windows of the same bytes
    assert a["group"] == b["group"] and a["group"].startswith("sha256:")
    assert a["split"] == b["split"]
    sources = json.loads(out.with_suffix(".sources.json").read_text())
    assert "content sha256" in sources["split_group"]
    # Exclusion by an earlier corpus also catches a byte-identical file under a new path.
    (docs / "copy_c.md").write_text(duplicate)
    for i in range(5):
        (docs / f"fresh{i}.md").write_text(text())
    module.build(tmp_path / "src", tmp_path / "next.json", count=5, seed=5, exclude_corpus=(out,))
    chosen = {r["source"] for r in load_corpus(tmp_path / "next.json")["documents"]}
    assert chosen == {f"packages/mdb/docs/fresh{i}.md" for i in range(5)}
