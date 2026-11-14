"""Planted-route recovery for the native-discovery learner (CPU, tiny synthetic data, ~15 s).

Audit 2026-11-13 finding 4: 0/288 learned route cells beat trivial baselines on Qwen, but no test
showed that the learner can recover a route that is known to be there. Here the route is planted:
every head of the single span reads the PREVIOUS token (row 0 reads itself). Each row of the state
is ``token embedding + position code`` (sin/cos at two frequencies, identical across documents),
so a q.k route over the encoded state CAN express "previous position"; the next state is
``x + W_o (route @ values)``. Held-out documents are scored by top-1 agreement with the planted
route (uniform expectation 0.29, position 0 0.20, previous token 1.0).

MEASURED 2026-11-13 (this file, torch 2.14 CPU; agreement on 8 held-out documents):

| run | linear_features | sparse_consumer | sparse_reconstruction (no route loss) |
|---|---|---|---|
| default budget: 60 steps, 60 accepted updates | 0.56 | 0.41 | 0.34 |
| 200 steps, 200 / 200 / 180 accepted | 1.00 | 0.94 | 0.43 |
| 600 steps, 310 / 320 / 420 accepted | 1.00 | 0.97 | 0.44 |
| 600 steps, position code removed from the state | 0.29 | 0.33 | 0.28 |

So the route form recovers a planted previous-token route only when (a) the input carries position,
(b) the arm has the route loss, and (c) it gets on the order of 200 accepted updates. The Qwen
rerun had none of (a), route loss on 4 of 6 arms, and 10-40 accepted updates.
"""
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("saturn_pub.training")

from mdb_lanes.attention_ranking import evaluate_route_forecast, trivial_route_baselines  # noqa: E402
from mdb_lanes.native_discovery import sealed  # noqa: E402

T, WIDTH, HEADS, VALUE_WIDTH, VOCAB = 10, 8, 2, 4, 6
BINDING = {"id": "planted-previous-token"}
BOUNDARIES = (0, 1)
RECOVERED = 0.9          # top-1 agreement with the planted route that counts as recovery (a choice)


def planted_routes():
    route = np.zeros((T, T), np.float32)
    route[0, 0] = 1.0
    for q in range(1, T):
        route[q, q - 1] = 1.0
    return np.stack([route] * HEADS)


def position_code():
    t = np.arange(T)[:, None]
    freq = np.array([2 * np.pi / T, 2 * np.pi / (T / 2)])[None, :]
    return np.concatenate([np.sin(t * freq), np.cos(t * freq)], 1).astype(np.float32)   # [T, 4]


def documents(n, seed):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(VOCAB, WIDTH - 4)).astype(np.float32)
    wv = rng.normal(size=(WIDTH, VALUE_WIDTH)).astype(np.float32) / np.sqrt(WIDTH)
    wo = rng.normal(size=(VALUE_WIDTH, WIDTH)).astype(np.float32) / np.sqrt(VALUE_WIDTH)
    pos, routes = position_code(), planted_routes()
    out = []
    for i in range(n):
        tokens = np.random.default_rng(seed * 1000 + i).integers(0, VOCAB, T)
        x0 = np.concatenate([emb[tokens], pos], 1)
        values = x0 @ wv
        x1 = x0 + (routes.mean(0) @ values) @ wo
        out.append({"states": np.stack([x0, x1]).astype(np.float32), "routes": routes[None].copy(),
                    "values": values[None].astype(np.float32)})
    return out


def rows(split, arrays, offset):
    out = []
    for i, a in enumerate(arrays):
        doc = f"{split}-{offset + i}"
        out.append((sealed({"split": split, "binding": BINDING, "boundaries": list(BOUNDARIES), "group": doc,
                            "text_sha256": hashlib.sha256(doc.encode()).hexdigest(), "document_id": doc}), a))
    return out


def all_docs():
    # one generator: the token embeddings and maps are shared; documents differ by tokens
    docs = documents(28, seed=11)
    return docs[:16], docs[16:20], docs[20:]


def score(routes_pred, routes_true):
    return evaluate_route_forecast(routes_pred, routes_true, name="learned")["nominator"]["top_source_agreement"]


def test_planted_data_are_what_they_claim():
    _, _, heldout = all_docs()
    truth = np.stack([d["routes"][0] for d in heldout])                       # [docs, heads, T, T]
    assert score(truth, truth) == 1.0
    base = trivial_route_baselines(truth)
    assert base["previous_token"]["top_source_agreement"] == pytest.approx(1.0)
    assert base["uniform"]["top_source_agreement"] < 0.4
    # the next state really depends on the planted read (not a no-op write)
    d = heldout[0]
    assert np.abs(d["states"][1] - d["states"][0]).max() > 0.1


def fit_and_score(tmp_path, steps, *, position=True):
    from mdb_lanes.native_discovery_learning import fit_predictors, forecast_arrays
    train, dev, heldout = all_docs()
    if not position:                                   # the control: same data, position code zeroed
        def strip(docs):
            out = [{k: v.copy() for k, v in d.items()} for d in docs]
            for d in out:
                d["states"][:, :, WIDTH - 4:] = 0.0
            return out
        train, dev, heldout = strip(train), strip(dev), strip(heldout)
    meta, models = fit_predictors(rows("discovery", train, 0), rows("development", dev, 100), root=tmp_path / "fit",
                                  binding=BINDING, boundaries=BOUNDARIES, features=16, steps=steps, interval_steps=10)
    truth = np.stack([d["routes"][0] for d in heldout])
    out = {}
    for name, model in models.items():
        pred = np.stack([forecast_arrays(model, d["states"][0])["routes"][0] for d in heldout])
        out[name] = (score(pred, truth), meta["arms"][name]["accepted_updates"])
    return out


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED 2026-11-13: at the default budget (60 steps, 60 accepted updates) no arm recovers the planted "
    "previous-token route (best 0.56, linear_features); see the module docstring and "
    "packages/mdb/experiments/native_discovery/FINDINGS.md. Strict: if a later learner recovers it at this budget, update this."))
def test_learner_recovers_planted_route_at_the_default_budget(tmp_path):
    scores = fit_and_score(tmp_path, steps=60)
    assert max(v for v, _ in scores.values()) >= RECOVERED, scores


def test_learner_recovers_planted_route_with_position_route_loss_and_200_updates(tmp_path):
    scores = fit_and_score(tmp_path, steps=200)
    assert scores["linear_features"][0] >= RECOVERED and scores["sparse_consumer"][0] >= RECOVERED, scores
    assert scores["linear_features"][1] >= 150 and scores["sparse_consumer"][1] >= 150, scores
    # the arm trained without the route loss does not recover it at the same budget
    assert scores["sparse_reconstruction"][0] < 0.6, scores


def test_without_position_in_the_input_no_arm_recovers_the_planted_route(tmp_path):
    scores = fit_and_score(tmp_path, steps=600, position=False)
    assert max(v for v, _ in scores.values()) < 0.45, scores           # uniform expectation is 0.29
