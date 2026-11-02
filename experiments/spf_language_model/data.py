"""Raw-byte next-token data with hash-bound, non-overlapping evaluation spans.

No syntax roles, target-derived features, normalization, or text decoding enter a
model input. The sealed test is accessible only through an explicit opt-in.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

VOCAB_SIZE = 256
TINY_SHAKESPEARE_REVISION = "370cbcd448eb7daf32f21a6be560b70e0b33c4e3"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    f"{TINY_SHAKESPEARE_REVISION}/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
TINY_SHAKESPEARE_BYTES = 1_115_394
TINY_SHAKESPEARE_SOURCE = {
    "name": "tiny_shakespeare",
    "url": TINY_SHAKESPEARE_URL,
    "revision": TINY_SHAKESPEARE_REVISION,
    "sha256": TINY_SHAKESPEARE_SHA256,
    "bytes": TINY_SHAKESPEARE_BYTES,
    "rights": "Underlying Shakespeare works are public domain; source repository supplies no separate corpus/formatting license.",
    "license": "public-domain-literary-text; distribution-license-unspecified",
    "license_scope": "Do not infer a corpus license from nanoGPT's MIT-licensed preparation code.",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class BatchCursor:
    """Stateless hash sampling with a restorable count of sampled windows.

    The cursor owns no global RNG state. Advancing or restoring one model's
    cursor cannot change another model's input stream.
    """

    corpus_sha256: str
    split: str
    seed: int = 0
    position: int = 0

    def snapshot(self) -> dict:
        return dict(
            corpus_sha256=self.corpus_sha256,
            split=self.split,
            seed=self.seed,
            position=self.position,
        )

    def restore(self, state: dict) -> None:
        if state["corpus_sha256"] != self.corpus_sha256 or state["split"] != self.split:
            raise ValueError("cursor restore requires the same corpus and split")
        if int(state["position"]) < 0:
            raise ValueError("negative cursor position")
        self.seed = int(state["seed"])
        self.position = int(state["position"])

    state_dict = snapshot
    load_state_dict = restore


class ByteCorpus:
    """One immutable byte specimen and three disjoint contiguous spans."""

    def __init__(
        self,
        raw: bytes,
        *,
        source: dict | None = None,
        train_fraction: float = 0.90,
        feedback_fraction: float = 0.05,
    ):
        if not raw:
            raise ValueError("empty corpus")
        if not (
            0 < train_fraction < 1
            and 0 < feedback_fraction < 1
            and train_fraction + feedback_fraction < 1
        ):
            raise ValueError("train/feedback fractions must leave a positive test span")
        self._raw = bytes(raw)
        self.sha256 = sha256(raw)
        self.source = dict(source or {"name": "provided_raw_bytes"})
        self.split_fractions = {
            "train": train_fraction,
            "feedback": feedback_fraction,
            "sealed_test": 1 - train_fraction - feedback_fraction,
        }
        train_end = int(len(raw) * train_fraction)
        feedback_end = train_end + int(len(raw) * feedback_fraction)
        self.spans = {
            "train": (0, train_end),
            "feedback": (train_end, feedback_end),
            "sealed_test": (feedback_end, len(raw)),
        }
        if any(stop - start < 2 for start, stop in self.spans.values()):
            raise ValueError("each split needs at least two raw bytes")
        self._byte_buffer = bytearray(raw)
        self._tokens = torch.frombuffer(self._byte_buffer, dtype=torch.uint8)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        source: dict | None = None,
        **kwargs,
    ) -> "ByteCorpus":
        raw = Path(path).read_bytes()
        if expected_sha256 is not None and sha256(raw) != expected_sha256:
            raise ValueError("corpus checksum mismatch")
        return cls(raw, source=source, **kwargs)

    def manifest(self) -> dict:
        return {
            "schema": "spf-byte-corpus-v1",
            "source": self.source,
            "raw_sha256": self.sha256,
            "raw_bytes": len(self._raw),
            "tokenizer": {
                "kind": "identity-raw-byte",
                "vocab_size": VOCAB_SIZE,
                "normalization": "none",
                "special_tokens": [],
            },
            "split_policy": "contiguous-disjoint-raw-spans-90-5-5-default",
            "split_fractions": self.split_fractions,
            "splits": {
                name: {
                    "start_inclusive": start,
                    "stop_exclusive": stop,
                    "raw_bytes": stop - start,
                    "sha256": sha256(self._raw[start:stop]),
                    "sealed": name == "sealed_test",
                }
                for name, (start, stop) in self.spans.items()
            },
            "limitations": "Contiguous text holdout; repeated phrases and characters can occur across spans; not a document- or author-generalization claim.",
        }

    @staticmethod
    def _split_name(split: str) -> str:
        return {"test": "sealed_test", "validation": "feedback", "val": "feedback"}.get(
            split, split
        )

    def cursor(
        self, split: str = "train", *, seed: int = 0, allow_sealed: bool = False
    ) -> BatchCursor:
        split = self._split_name(split)
        self._check_split(split, allow_sealed)
        return BatchCursor(self.sha256, split, int(seed))

    def _check_split(self, split: str, allow_sealed: bool) -> None:
        if split not in self.spans:
            raise ValueError(f"unknown split: {split}")
        if split == "sealed_test" and not allow_sealed:
            raise PermissionError("sealed test requires allow_sealed=True at terminal evaluation")

    def sample_batch(
        self,
        split: str = "train",
        *,
        batch_size: int = 4,
        seq_len: int = 64,
        cursor: BatchCursor | None = None,
        seed: int | None = None,
        device="cpu",
        allow_sealed: bool = False,
        return_offsets: bool = False,
    ):
        split = self._split_name(split)
        self._check_split(split, allow_sealed)
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if cursor is not None and seed is not None:
            raise ValueError("provide cursor or seed, not both")
        if cursor is None:
            cursor = self.cursor(split, seed=0 if seed is None else seed, allow_sealed=allow_sealed)
        if cursor.corpus_sha256 != self.sha256 or cursor.split != split:
            raise ValueError("cursor belongs to a different corpus or split")
        start, stop = self.spans[split]
        choices = stop - start - seq_len
        if choices <= 0:
            raise ValueError("split is too short for seq_len+1 bytes")
        offsets = []
        for row in range(batch_size):
            key = f"{self.sha256}:{split}:{cursor.seed}:{cursor.position + row}".encode("ascii")
            offsets.append(
                start + int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % choices
            )
        x = torch.stack([self._tokens[s : s + seq_len] for s in offsets]).to(
            device=device, dtype=torch.long
        )
        y = torch.stack([self._tokens[s + 1 : s + seq_len + 1] for s in offsets]).to(
            device=device, dtype=torch.long
        )
        cursor.position += batch_size
        return (x, y, offsets) if return_offsets else (x, y)

    def make_batches(self, split: str = "train", **kwargs) -> "BatchStream":
        return make_batches(self, split, **kwargs)


class BatchStream(Iterator):
    def __init__(
        self,
        corpus: ByteCorpus,
        split: str,
        *,
        cursor: BatchCursor,
        num_batches: int | None = None,
        **batch_kwargs,
    ):
        if num_batches is not None and num_batches < 0:
            raise ValueError("negative batch count")
        self.corpus, self.split, self.cursor = corpus, split, cursor
        self.num_batches, self.batch_kwargs, self.emitted = num_batches, batch_kwargs, 0

    def __next__(self):
        if self.num_batches is not None and self.emitted >= self.num_batches:
            raise StopIteration
        result = self.corpus.sample_batch(self.split, cursor=self.cursor, **self.batch_kwargs)
        self.emitted += 1
        return result

    def snapshot(self) -> dict:
        return {"cursor": self.cursor.snapshot(), "emitted": self.emitted}

    def restore(self, state: dict) -> None:
        emitted = int(state["emitted"])
        if emitted < 0:
            raise ValueError("negative emitted count")
        self.cursor.restore(state["cursor"])
        self.emitted = emitted

    state_dict = snapshot
    load_state_dict = restore


def make_batches(
    corpus: ByteCorpus,
    split: str = "train",
    *,
    seed: int = 0,
    cursor: BatchCursor | None = None,
    num_batches: int | None = None,
    allow_sealed: bool = False,
    **batch_kwargs,
) -> BatchStream:
    split = corpus._split_name(split)
    corpus._check_split(split, allow_sealed)
    if cursor is None:
        cursor = corpus.cursor(split, seed=seed, allow_sealed=allow_sealed)
    return BatchStream(
        corpus,
        split,
        cursor=cursor,
        num_batches=num_batches,
        allow_sealed=allow_sealed,
        **batch_kwargs,
    )


def load_tiny_shakespeare(data_dir: str | Path, *, download: bool = False) -> ByteCorpus:
    """Read a verified local artifact; downloading is an explicit setup action."""
    data_dir = Path(data_dir)
    path = data_dir / "tiny_shakespeare.txt"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"missing {path}; acquire once with download=True")
        request = urllib.request.Request(
            TINY_SHAKESPEARE_URL, headers={"User-Agent": "spf-byte-lm-research"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(TINY_SHAKESPEARE_BYTES + 1)
        if len(raw) != TINY_SHAKESPEARE_BYTES or sha256(raw) != TINY_SHAKESPEARE_SHA256:
            raise ValueError("download does not match the pinned corpus")
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    corpus = ByteCorpus.from_file(
        path, expected_sha256=TINY_SHAKESPEARE_SHA256, source=TINY_SHAKESPEARE_SOURCE
    )
    manifest_path = data_dir / "corpus_manifest.json"
    manifest = json.dumps(corpus.manifest(), indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text() != manifest:
            raise ValueError("existing corpus manifest differs from verified split contract")
    else:
        manifest_path.write_text(manifest)
    return corpus
