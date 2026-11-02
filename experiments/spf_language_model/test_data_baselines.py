"""Small mechanics checks: byte custody, replayable input stream, and causality."""

import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import build_baseline, parameter_count
from data import ByteCorpus, make_batches


@pytest.fixture
def corpus():
    return ByteCorpus(bytes(range(256)) * 16)


def test_disjoint_spans_and_exact_next_byte(corpus):
    manifest = corpus.manifest()
    raw = bytes(range(256)) * 16
    previous_end = 0
    for split, row in manifest["splits"].items():
        start, stop = row["start_inclusive"], row["stop_exclusive"]
        assert start == previous_end
        assert row["sha256"] == hashlib.sha256(raw[start:stop]).hexdigest()
        x, y, offsets = corpus.sample_batch(
            split, seq_len=32, batch_size=10, allow_sealed=True, return_offsets=True
        )
        assert torch.equal(x[:, 1:], y[:, :-1])
        for i, offset in enumerate(offsets):
            assert start <= offset and offset + 33 <= stop
            assert bytes(x[i].tolist()) == raw[offset : offset + 32]
            assert bytes(y[i].tolist()) == raw[offset + 1 : offset + 33]
        previous_end = stop
    assert previous_end == len(raw)


def test_cursor_rewind_and_independent_model_streams(corpus):
    a = make_batches(corpus, batch_size=2, seq_len=16, seed=19, num_batches=3)
    b = make_batches(corpus, batch_size=2, seq_len=16, seed=19, num_batches=3)
    state = a.snapshot()
    first = next(a)
    next(a)
    assert all(torch.equal(x, y) for x, y in zip(first, next(b)))
    a.restore(state)
    assert all(torch.equal(x, y) for x, y in zip(first, next(a)))
    next(a)
    next(a)
    with pytest.raises(StopIteration):
        next(a)
    with pytest.raises(ValueError, match="same corpus and split"):
        corpus.cursor("feedback").restore(state["cursor"])


def test_sealed_test_requires_explicit_opt_in(corpus):
    for split in ("test", "sealed_test"):
        with pytest.raises(PermissionError):
            corpus.sample_batch(split)
        with pytest.raises(PermissionError):
            make_batches(corpus, split)
    # A previously authorized cursor does not waive the access check.
    test_cursor = corpus.cursor("sealed_test", allow_sealed=True)
    with pytest.raises(PermissionError):
        corpus.sample_batch("sealed_test", cursor=test_cursor)


def test_bad_checksum_and_short_span_rejected(tmp_path, corpus):
    path = tmp_path / "fixture.txt"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="checksum"):
        ByteCorpus.from_file(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="too short"):
        corpus.sample_batch("feedback", seq_len=2048)
    cursor = corpus.cursor()
    with pytest.raises(ValueError, match="cursor or seed"):
        corpus.sample_batch(cursor=cursor, seed=0)


@pytest.mark.parametrize("kind", ["transformer", "mamba"])
def test_baseline_causality_gradients_and_seed_closure(kind):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        before = torch.random.get_rng_state().clone()
        model = build_baseline(kind, d_model=8, n_layers=1, state_size=4, seed=81)
        assert torch.equal(before, torch.random.get_rng_state())
        twin = build_baseline(kind, d_model=8, n_layers=1, state_size=4, seed=81)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), twin.parameters()))
        assert model.architecture["parameters"] == parameter_count(model)
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 120) % 256
        model.eval()
        original_logits, changed_logits = model(x), model(changed)
        assert original_logits.shape == (2, 8, 256)
        assert torch.isfinite(original_logits).all()
        torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4], rtol=0, atol=0)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), x.roll(-1, 1).flatten())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    finally:
        torch.set_num_threads(old_threads)
