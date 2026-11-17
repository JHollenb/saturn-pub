"""Light CPU checks of sup.py on synthetic block-disjoint data (laptop-safe: < 5 s)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sup  # noqa: E402


def blocks(n=600, K=3, w=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    cls = torch.randint(K, (n,), generator=g)
    X = torch.zeros(n, K * w)
    bits = (torch.rand(n, w, generator=g) < 0.4).float()
    bits[:, 0] = 1.0                                        # bit 0 of each class block always on
    for c in range(K):
        X[cls == c, c * w:(c + 1) * w] = bits[cls == c]
    L = torch.nn.functional.one_hot(cls, K).bool()
    return X, cls, L


def test_best_cut_control_raw_is_exact():
    X, _, L = blocks()
    r = sup.best_cut(X, L)
    assert torch.allclose(r["iou"], torch.ones(3)) and torch.allclose(r["precision"], torch.ones(3))


def test_best_cut_never_splits_ties():
    Z = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    L = torch.tensor([[True], [False], [False], [False]])
    r = sup.best_cut(Z, L)
    assert abs(float(r["iou"][0]) - 0.5) < 1e-6           # cut must take both tied rows


def test_rotation_orthogonal_and_pca_invariant():
    X, _, L = blocks()
    R = sup.rotation(X.shape[1], 1)
    assert torch.allclose(R @ R.T, torch.eye(len(R)), atol=1e-5)
    a = sup.best_cut(sup.pca(X, 4)[0], L)["iou"]
    b = sup.best_cut(sup.pca(sup.mix(X, R), 4)[0], L)["iou"]
    assert torch.allclose(a, b, atol=1e-4)


def test_rotation_hurts_raw_neurons():
    X, _, L = blocks()
    raw = sup.best_cut(sup.mix(X, sup.rotation(X.shape[1], 2)), L)["iou"].mean()
    assert raw < 0.95


def test_kmeans_native_group():
    X, cls, L = blocks()
    B = sup.kmeans(X, 3, seed=0)
    nat = sup.native(B.bool(), L)
    assert float(nat["group_iou"].mean()) > 0.9
    vs = sup.value_split(B, cls, X.sum(1).long(), nat)
    assert vs["pure_units"] >= 3 and 0 < vs["value_purity"] <= 1


def test_ica_unmixes_laplace_sources():
    g = torch.Generator().manual_seed(0)
    S = torch.distributions.Laplace(0, 1).sample((4000, 3))
    A = torch.randn(3, 3, generator=g)
    est = sup.ica(S @ A.T, 3, seed=0)
    C = torch.corrcoef(torch.cat([S.T, est.T]))[:3, 3:].abs()
    assert float(C.max(1).values.min()) > 0.95


def test_projection_and_noise_shapes():
    X, _, _ = blocks()
    Y = sup.add_noise(sup.mix(X, sup.projection(5, X.shape[1], 0)), 0.3, 0)
    assert Y.shape == (len(X), 5)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
