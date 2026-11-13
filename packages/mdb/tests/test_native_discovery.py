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
    exported = json.loads((tmp_path / "index-input/observations.json").read_text())
    learned = [r for r in exported if "source_top1_agreement" in r["metrics"]]
    assert learned and all(r["metrics"]["source_top1_baselines"]["baselines"]["position_0"] for r in learned)
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


def test_export_writes_baselines_beside_top1_or_explicit_null():
    from mdb.native_discovery_export import source_top1_metrics
    t = np.zeros((4, 4))
    t[np.arange(4), [0, 1, 1, 3]] = 1.0
    from mdb.attention_ranking import evaluate_route_forecast
    record = evaluate_route_forecast(t, t, name="x")
    full = source_top1_metrics({"source_top1_agreement": 1.0, "source_top1": record})
    assert full["source_top1_agreement"] == 1.0
    assert {"position_0", "previous_token", "uniform"} <= set(full["source_top1_baselines"]["baselines"])
    old = source_top1_metrics({"source_top1_agreement": 0.5})  # a saved run without baselines
    assert old == {"source_top1_agreement": 0.5, "source_top1_baselines": None}


def test_shared_prefix_threshold_can_be_changed_or_disabled(tmp_path, monkeypatch):
    from mdb.native_discovery_run import run_on_model
    shared = "alpha beta gamma delta epsilon zeta eta theta"
    rows = [{"id": k, "text": shared + " " + t} for k, t in
            (("a", "one two three"), ("b", "four five six"), ("c", "seven eight nine"),
             ("d", "ten eleven twelve"), ("e", "thirteen fourteen fifteen"))]
    rows = [{**r, "text": r["text"] + " " + " ".join(r["id"] * (i + 2) for i in range(6))} for r in rows]
    corpus = prepare_corpus(rows, seed=3, near_duplicate=1.0)
    save_json(tmp_path / "corpus.json", corpus)
    run_on_model(tiny_model(), TinyTokenizer(), corpus=tmp_path / "corpus.json", out=tmp_path / "run",
                 binding={"model_identity": "tiny", "execution": {"device": "cpu"}},
                 max_tokens=12, stages=2, features=4, steps=1, interval_steps=1,
                 refuse_shared_prefix=None)
    visible = json.loads((tmp_path / "run/visible-inputs.json").read_text())
    assert max(v["longest_cross_split_shared_prefix"] for v in visible) >= 8  # recorded, not refused
    assert all(v["shared_prefix_refusal_tokens"] is None for v in visible)
    # The CLI forwards the option; 0 means disabled, the default stays 8.
    import mdb.native_discovery as nd
    import mdb.native_discovery_run as ndr
    seen = []
    monkeypatch.setenv("MRUN_JOB_ID", "x")
    monkeypatch.setattr(ndr, "run_pretrained", lambda **kw: seen.append(kw) or {})
    base = ["run", "--corpus", "c", "--model", "m", "--out", "o"]
    nd.main(base)
    nd.main(base + ["--shared-prefix-refusal", "20"])
    nd.main(base + ["--shared-prefix-refusal", "0"])
    assert [k["refuse_shared_prefix"] for k in seen] == [8, 20, None]


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


# ----------------------------------------------------------------- position-0-robust options

def _sink_samples(n=4, length=5, width=6, stages=2, heads=4, value_width=4, seed=0):
    """Tiny traces whose position 0 carries a massive activation after the first boundary."""
    import torch
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        states = g.normal(size=(stages + 1, length, width)).astype(np.float32)
        states[1:, 0] += 500.0
        routes = np.tril(g.random((stages, heads, length, length))) + 1e-3 * np.tril(np.ones((length, length)))
        routes = (routes / routes.sum(-1, keepdims=True)).astype(np.float32)
        out.append({"states": torch.tensor(states), "routes": torch.tensor(routes),
                    "values": torch.tensor(g.normal(size=(stages, length, value_width)).astype(np.float32))})
    return out


def test_split_position_mode_initialises_at_the_position_split_mean_write():
    import torch

    from mdb.analysis_metrics import fit_mean_writes, mean_write_forecast
    from mdb.native_discovery_learning import StatePredictor, forecast_arrays
    samples = _sink_samples()
    fitted = fit_mean_writes([s["states"].numpy() for s in samples])
    for mode, split in (("split", True), ("global", False)):
        net = StatePredictor(width=6, features=4, heads=4, value_width=4, stages=2,
                             kind="sparse_reconstruction", position_mode=mode)
        net.initialize_scales(samples)
        frontier = samples[0]["states"][0].numpy()
        predicted = forecast_arrays(net, frontier)["state"]
        np.testing.assert_allclose(predicted, mean_write_forecast(frontier, fitted, split=split),
                                   rtol=1e-4, atol=1e-3)
    # Per-class write scales: position 0's scale is its own, so its huge write cannot dominate.
    span = net.spans[0]
    assert span.row_write_scale(5).shape == ()
    split_net = StatePredictor(width=6, features=4, heads=4, value_width=4, stages=2,
                               kind="linear_features", position_mode="split")
    split_net.initialize_scales(samples)
    scale = split_net.spans[0].row_write_scale(5)
    assert scale.shape == (5, 1) and float(scale[0]) > 100 * float(scale[1])
    assert torch.all(scale[1:] == scale[1])


def test_split_mode_loss_normalises_each_position_class_separately():
    import torch

    from mdb.native_discovery_learning import StatePredictor, sample_loss
    samples = _sink_samples(n=3)
    net = StatePredictor(width=6, features=4, heads=4, value_width=4, stages=2,
                         kind="sparse_reconstruction", position_mode="split")
    net.initialize_scales(samples)
    sample = {k: v.clone() for k, v in samples[0].items()}
    with torch.no_grad():
        base = float(sample_loss(net, sample, auxiliary=False)[1][0]["state_loss"])
        # An error of one position-0 write scale at position 0 adds exactly 1/T (one row, one unit).
        sample["states"][1][0] += net.spans[0].write_scale[0]
        moved = float(sample_loss(net, sample, auxiliary=False)[1][0]["state_loss"])
    assert moved - base == pytest.approx(1 / 5, rel=0.2)


def test_robust_development_metric_is_the_shared_centred_position0_excluded_error():
    import torch

    from mdb.analysis_metrics import relative_error
    from mdb.native_discovery_learning import StatePredictor, evaluate
    samples = _sink_samples(n=2)
    net = StatePredictor(width=6, features=4, heads=4, value_width=4, stages=2, kind="linear_features")
    net.initialize_scales(samples)
    result = evaluate(net, samples)
    expected = []
    with torch.no_grad():
        for s in samples:
            rows = net.forecast(s["states"][0])
            expected.append(np.mean([relative_error(r["state"].numpy(), s["states"][i + 1].numpy())
                                     ["centered_relative_mse_excluding_position_0"] for i, r in enumerate(rows)]))
    assert result["centered_error_excluding_position_0"] == pytest.approx(np.mean(expected), rel=1e-6)


def test_kv_group_read_keeps_groups_separate():
    import torch

    from mdb.native_discovery_learning import SpanPredictor
    t, heads, groups = 3, 4, 2
    route = torch.zeros(heads, t, t)
    route[:2, :, 0] = 1.0                     # KV group 0 heads read source 0
    route[2:] = torch.eye(t)                  # KV group 1 heads read the current token
    value = torch.arange(t * 4, dtype=torch.float32).reshape(t, 4)  # two groups of width 2
    grouped = SpanPredictor(5, 3, heads, 4, kind="linear_features", read_groups=groups).read(route, value)
    np.testing.assert_allclose(grouped[:, :2], value[[0, 0, 0], :2])
    np.testing.assert_allclose(grouped[:, 2:], value[:, 2:])
    averaged = SpanPredictor(5, 3, heads, 4, kind="linear_features").read(route, value)
    np.testing.assert_allclose(averaged, 0.5 * (value[[0, 0, 0]] + value))
    with pytest.raises(DiscoveryError, match="divide"):
        SpanPredictor(5, 3, 4, 4, kind="linear_features", read_groups=3)


def test_arm_presets_are_explicit_and_original_by_default():
    from mdb.native_discovery_learning import ARM_PRESETS, KINDS, resolve_arms
    assert [a["name"] for a in resolve_arms(None)] == list(KINDS)
    assert all(a["position_mode"] == "global" and a["selection"] == "original" and a["read_groups"] == 1
               for a in resolve_arms(None))
    robust = resolve_arms("original+p0robust", kv_groups=2)
    assert len(robust) == 6 and {a["read_groups"] for a in robust} == {1, 2}
    assert len(ARM_PRESETS["original+p0robust"]) == 6
    with pytest.raises(DiscoveryError, match="KV"):
        resolve_arms("original+p0robust")
    with pytest.raises(DiscoveryError, match="unknown arm option"):
        resolve_arms([{"kind": "linear_features", "loss": "x"}])


def test_contrast_nominator_recovers_per_head_rules_without_learning():
    from mdb.native_discovery_nominator import (
        fit_contrast_nominator,
        nominate_sources,
        one_hot_routes,
        score_nominations,
    )
    length = 6
    q = np.arange(length)
    sources = np.stack([np.zeros(length, np.int64), np.maximum(q - 1, 0), q])  # sink, previous, current
    routes = 0.9 * one_hot_routes(sources[None]) + 0.1 * np.tril(np.ones((length, length))) / (q + 1)[:, None]
    fitted = fit_contrast_nominator([routes, routes], offsets=3)
    assert [h["rule"] for h in fitted["heads"]] == ["position_0", "offset_1", "offset_0"]
    assert fitted["learning"] == "none"
    nominated = nominate_sources(fitted, length)
    np.testing.assert_array_equal(nominated[0], sources)
    scored = score_nominations(nominated, routes)
    record = scored[0]["source_top1"]
    assert record["nominator"]["top_source_agreement"] == 1.0
    assert {"position_0", "previous_token", "uniform"} <= set(record["baselines"])
    # One trivial rule cannot cover three different heads; the per-head rule beats every baseline.
    assert record["beats_every_trivial_baseline"]
    with pytest.raises(ValueError, match="causal"):
        one_hot_routes(np.array([[1, 0]]))


def test_position_robust_pipeline_seals_split_baseline_and_nominator(tmp_path):
    import importlib.util

    from mdb.native_discovery import verify_run
    from mdb.native_discovery_learning import load_predictors
    from mdb.native_discovery_run import run_on_model
    corpus = prepare_corpus(documents(), seed=19)
    save_json(tmp_path / "corpus.json", corpus)
    root = tmp_path / "analysis"
    run_on_model(tiny_model(), TinyTokenizer(), corpus=tmp_path / "corpus.json", out=root,
                 binding={"model_identity": "tiny-random-mechanics", "execution": {"device": "cpu"}},
                 max_tokens=12, stages=2, features=4, steps=2, interval_steps=1,
                 arms="original+p0robust", sealed_position_split=True, contrast_nominator=True)
    assert verify_run(root)["mechanics_status"] == "valid"
    report = json.loads((root / "report.json").read_text())
    names = {"linear_features", "sparse_reconstruction", "sparse_consumer"}
    names |= {n + ".p0robust" for n in names}
    assert report["options"]["sealed_position_split_mean_write"] and report["options"]["contrast_nominator"]
    for row in report["heldout"]:
        assert set(row["native_output_consumer"]) == names
        assert len(row["split_mean_write"]) == 2 and "native_to_predicted_kl" in row["split_mean_write_consumer"]
        # The sealed split baseline is the same forecast as the post-hoc one (fit on discovery only).
        for sealed_row, posthoc in zip(row["split_mean_write"], row["position_split_mean_write"], strict=True):
            assert sealed_row["relative_mse"] == pytest.approx(posthoc["relative_mse"], rel=1e-4)
        assert {"position_0", "previous_token", "uniform"} <= set(
            row["contrast_nominator"][0]["source_top1"]["baselines"])
    for forecast in (root / "forecasts").iterdir():
        kinds = [json.loads(p.read_text()) for p in sorted((forecast / "events").glob("*.json"))]
        order = [e["kind"] for e in kinds]
        sealed_names = {e["detail"]["name"] for e in kinds if e["kind"] == "forecast_persisted"}
        assert {"split_mean_write", "contrast_nominator"} <= sealed_names
        assert order.index("globally_sealed") < order.index("observation_persisted")
    metadata, models = load_predictors(root / "predictor")
    assert set(models) == names
    robust = metadata["arms"]["sparse_consumer.p0robust"]
    assert robust["arm_options"]["position_mode"] == "split"
    assert robust["arm_options"]["selection"] == "position_robust"
    assert "arm_options" not in metadata["arms"]["sparse_consumer"]
    receipts = list((root / "predictor/sparse_consumer.p0robust/receipts").rglob("*.json"))
    assert any("centered_error_excluding_position_0" in p.read_text() for p in receipts)
    nominator = json.loads((root / "contrast-nominator.json").read_text())
    assert nominator["learning"] == "none" and set(nominator["documents"]) == set(metadata["fitted_document_ids"])
    # The artifact-only audit re-scores every sealed arm, including the new ones.
    spec = importlib.util.spec_from_file_location(
        "native_audit", Path(__file__).parents[1] / "experiments/native_discovery/audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "evidence").symlink_to(root)
    import tarfile
    with tarfile.open(stage / "evidence.tar.gz", "w:gz") as stream:
        stream.add(root, arcname="evidence")
    save_json(stage / "result.json", {"artifact": {"sha256": __import__("mdb.native_discovery", fromlist=["x"]).file_hash(stage / "evidence.tar.gz")}})
    assert audit.audit(stage)["metric_checks"] > 0
    spec = importlib.util.spec_from_file_location(
        "native_rerun_report", Path(__file__).parents[1] / "experiments/native_discovery/rerun_report.py")
    tables_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables_module)
    tables = tables_module.tables(stage)
    assert set(tables["output_kl"]) == names | {"hold_state", "mean_write", "split_mean_write"}
    assert "sparse_reconstruction.p0robust vs split_mean_write: output KL" in tables["paired"]
    assert any(k.startswith("contrast_nominator vs") for k in tables["paired"])
    row = tables["paired"]["sparse_consumer vs split_mean_write: output KL"]
    assert row["candidate_better"] + row["candidate_worse"] + row["ties"] == row["n"] == len(tables["documents"])
    assert row["bootstrap_95"][0] <= row["mean_difference"] <= row["bootstrap_95"][1]
