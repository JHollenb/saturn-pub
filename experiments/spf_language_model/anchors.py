"""Train-only frozen raw-byte frequency anchors for the neural LM pilot.

These are descriptive anchors, not parameter- or compute-matched competitors.
All counts come from the disjoint training span. Smoothing is fixed before any
feedback/test scores are read. A bigram uses only the current byte, whereas a
unigram ignores even that byte; neither uses an oracle syntax input.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn

VOCAB_SIZE = 256


def _sealed_access(split: str, allow_sealed: bool) -> None:
    if split in {"test", "sealed_test"} and not allow_sealed:
        raise PermissionError("sealed anchor scores require allow_sealed=True")
    if split not in {"train", "feedback", "val", "validation", "test", "sealed_test"}:
        raise ValueError(f"unknown split: {split}")


def _byte_tensor(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.dtype not in (torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must be an integer [batch,time] tensor")
    if value.numel() == 0 or bool((value < 0).any()) or bool((value >= VOCAB_SIZE).any()):
        raise ValueError(f"{name} must contain nonempty raw-byte IDs in [0,255]")


class FrozenByteAnchor(nn.Module):
    """An immutable fitted categorical distribution with no trainable weights."""

    def __init__(self, kind: str, counts: np.ndarray, *, smoothing: float, training_identity: dict):
        super().__init__()
        if kind not in {"unigram", "bigram"}:
            raise ValueError("anchor kind must be unigram or bigram")
        expected_shape = (VOCAB_SIZE,) if kind == "unigram" else (VOCAB_SIZE, VOCAB_SIZE)
        if counts.shape != expected_shape or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("counts do not match the declared anchor")
        if np.any(counts < 0) or counts.sum() <= 0:
            raise ValueError("anchor counts must be nonnegative and nonempty")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("additive smoothing must be finite and positive")
        # Float64 avoids needless accumulation error in a cheap frozen anchor.
        values = torch.from_numpy(counts.astype(np.int64, copy=True))
        mass = values.to(dtype=torch.float64) + float(smoothing)
        denominator = mass.sum() if kind == "unigram" else mass.sum(-1, keepdim=True)
        self.register_buffer("counts", values)
        self.register_buffer("log_probabilities", (mass / denominator).log())
        self.kind = kind
        self.metadata = {
            "kind": f"train_only_{kind}_raw_byte_anchor",
            "vocab_size": VOCAB_SIZE,
            "history_used": "none" if kind == "unigram" else "current_byte_only",
            "smoothing": {
                "kind": "fixed_additive",
                "alpha": float(smoothing),
                "selected_using_feedback_or_test": False,
            },
            "count_observations": int(values.sum()),
            "trainable_parameters": 0,
            "categorical_entries": int(values.numel()),
            "fitted_storage_bytes": values.numel() * values.element_size()
            + self.log_probabilities.numel() * self.log_probabilities.element_size(),
            "counts_sha256": hashlib.sha256(values.cpu().numpy().tobytes()).hexdigest(),
            "training_identity": dict(training_identity),
            "limitations": "Frequency anchor; categorical capacity is not matched to the neural arms. Bigram advantage does not establish use of earlier context.",
        }
        self.eval()

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Return normalized log-probabilities as [B,T,256] model logits."""
        _byte_tensor(byte_ids, "byte_ids")
        table = self.log_probabilities.to(device=byte_ids.device)
        if self.kind == "unigram":
            return table.view(1, 1, VOCAB_SIZE).expand(*byte_ids.shape, VOCAB_SIZE)
        return table[byte_ids.long()]

    def target_bits(self, byte_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _byte_tensor(byte_ids, "byte_ids")
        _byte_tensor(targets, "targets")
        if byte_ids.shape != targets.shape:
            raise ValueError("byte IDs and next-byte targets must have identical shape")
        table = self.log_probabilities.to(device=targets.device)
        if self.kind == "unigram":
            selected = table[targets.long()]
        else:
            selected = table[byte_ids.to(device=targets.device, dtype=torch.long), targets.long()]
        return -selected / math.log(2)


def fit_train_anchors(corpus, *, smoothing: float = 1.0) -> dict[str, FrozenByteAnchor]:
    """Count only raw bytes and adjacent pairs wholly inside corpus.train.

    The currently public ByteCorpus contract supplies spans, a manifest, and
    an immutable raw specimen. Access its raw backing once here to avoid either
    random-window counting bias or a silently different corpus on disk.
    """
    manifest = corpus.manifest()
    train = manifest["splits"]["train"]
    start, stop = train["start_inclusive"], train["stop_exclusive"]
    raw_train = corpus._raw[start:stop]
    digest = hashlib.sha256(raw_train).hexdigest()
    if digest != train["sha256"] or len(raw_train) != train["raw_bytes"]:
        raise ValueError("training span does not match the corpus manifest")
    if len(raw_train) < 2:
        raise ValueError("training span needs at least two bytes")
    values = np.frombuffer(raw_train, dtype=np.uint8)
    unigram = np.bincount(values, minlength=VOCAB_SIZE).astype(np.int64)
    pair_ids = values[:-1].astype(np.int64) * VOCAB_SIZE + values[1:].astype(np.int64)
    bigram = np.bincount(pair_ids, minlength=VOCAB_SIZE**2).reshape(VOCAB_SIZE, VOCAB_SIZE)
    identity = {
        "raw_corpus_sha256": manifest["raw_sha256"],
        "split": "train",
        "start_inclusive": start,
        "stop_exclusive": stop,
        "raw_train_sha256": digest,
        "raw_train_bytes": len(raw_train),
        "cross_split_pairs": 0,
        "feedback_bytes_counted": 0,
        "test_bytes_counted": 0,
    }
    return {
        kind: FrozenByteAnchor(kind, counts, smoothing=smoothing, training_identity=identity)
        for kind, counts in (("unigram", unigram), ("bigram", bigram))
    }


@torch.no_grad()
def score_anchors_on_batch(
    anchors: Mapping[str, FrozenByteAnchor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    split: str = "feedback",
    raw_offsets=None,
    allow_sealed: bool = False,
) -> dict:
    """Score exactly supplied frozen windows; this never samples or fits."""
    _sealed_access(split, allow_sealed)
    _byte_tensor(x, "byte_ids")
    _byte_tensor(y, "targets")
    if x.shape != y.shape:
        raise ValueError("input and target windows must have identical shape")
    if raw_offsets is not None and len(raw_offsets) != x.shape[0]:
        raise ValueError("one raw offset is required for each supplied window")
    rows = {}
    for name, anchor in anchors.items():
        bits = anchor.target_bits(x, y)
        rows[name] = {
            "bpb": float(bits.mean()),
            "per_window_bpb": bits.mean(-1).cpu().tolist(),
            "per_token_bits": bits.cpu().tolist(),
            "target_bytes": y.numel(),
        }
    return {
        "split": "sealed_test" if split == "test" else split,
        "raw_offsets": None if raw_offsets is None else [int(offset) for offset in raw_offsets],
        "byte_ids_sha256": hashlib.sha256(
            x.detach().cpu().numpy().astype(np.uint8).tobytes()
        ).hexdigest(),
        "target_ids_sha256": hashlib.sha256(
            y.detach().cpu().numpy().astype(np.uint8).tobytes()
        ).hexdigest(),
        "batch_size": x.shape[0],
        "sequence_length": x.shape[1],
        "target_bytes": y.numel(),
        "anchors": rows,
        "heldout_used_for_fit_or_selection": False,
    }


def frozen_windows_score(
    corpus,
    anchors: Mapping[str, FrozenByteAnchor],
    *,
    split: str = "feedback",
    seed: int = 811,
    batch_size: int = 8,
    seq_len: int = 64,
    num_batches: int = 1,
    allow_sealed: bool = False,
) -> dict:
    """One shared input stream for every frozen anchor, with exact offsets.

    Defaults reproduce the pilot's feedback evaluator. Its sealed test requires
    split='sealed_test', seed=991, batch_size=4, num_batches=16 and explicit
    allow_sealed=True (using the request's actual length/batch when changed).
    """
    _sealed_access(split, allow_sealed)
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    for anchor in anchors.values():
        if anchor.metadata["training_identity"]["raw_corpus_sha256"] != corpus.sha256:
            raise ValueError("anchor was fitted to a different corpus")
    cursor = corpus.cursor(split, seed=seed, allow_sealed=allow_sealed)
    rows = []
    for index in range(num_batches):
        x, y, offsets = corpus.sample_batch(
            split,
            batch_size=batch_size,
            seq_len=seq_len,
            cursor=cursor,
            return_offsets=True,
            allow_sealed=allow_sealed,
        )
        row = score_anchors_on_batch(
            anchors, x, y, split=split, raw_offsets=offsets, allow_sealed=allow_sealed
        )
        row["batch"] = index
        rows.append(row)
    total_targets = sum(row["target_bytes"] for row in rows)
    aggregate = {
        name: {
            "bpb": sum(row["anchors"][name]["bpb"] * row["target_bytes"] for row in rows)
            / total_targets,
            "target_bytes": total_targets,
        }
        for name in anchors
    }
    return {
        "schema": "spf-byte-lm-frozen-frequency-anchors-v1",
        "split": cursor.split,
        "sampler_seed": seed,
        "sampler_final": cursor.snapshot(),
        "fit": {name: anchor.metadata for name, anchor in anchors.items()},
        "aggregate": aggregate,
        "rows": rows,
        "mechanics_status": "valid",
        "trend_status": "observations-retained",
        "terminal_status": "not-assessed",
        "limitations": "Shared sampled windows can overlap; train counts and category capacity are not a neural optimization budget.",
    }
