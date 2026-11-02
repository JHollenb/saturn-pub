"""Byte-only controlled memory organisms, separate from natural-text evidence.

Recall:   ^ A <IID lowercase distractors> ?       -> next byte A
Revision: ^ A <gap> ! B <IID distractors> ?       -> next byte B

Only x reaches a model. y, query masks, source positions and counterfactual
payload labels are loss/evidence metadata. There is no padding. Separate RNG
identities quarantine the cold audit; the finite grammar can repeat exact strings
across splits, especially at delay zero, so no text-disjoint claim is made.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

DEFAULT_DELAYS = (0, 4, 8, 16, 32, 64)
PAYLOAD_BYTES = tuple(b"ABCD")
DEFAULT_SPLIT_SEEDS = {"train": 17011, "feedback": 81103, "sealed_audit": 99107}


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _split(split: str) -> str:
    return {
        "val": "feedback",
        "validation": "feedback",
        "test": "sealed_audit",
        "audit": "sealed_audit",
        "cold_audit": "sealed_audit",
        "sealed_test": "sealed_audit",
    }.get(split, split)


def _access(split: str, allow_sealed: bool) -> str:
    split = _split(split)
    if split not in DEFAULT_SPLIT_SEEDS:
        raise ValueError(f"unknown memory split: {split}")
    if split == "sealed_audit" and not allow_sealed:
        raise PermissionError("cold audit requires explicit allow_sealed=True")
    return split


@dataclass
class MemoryCursor:
    task_sha256: str
    split: str
    seed: int
    position: int = 0
    batch_position: int = 0

    def snapshot(self) -> dict:
        return dict(
            task_sha256=self.task_sha256,
            split=self.split,
            seed=self.seed,
            position=self.position,
            batch_position=self.batch_position,
        )

    def restore(self, state: dict) -> None:
        if state["task_sha256"] != self.task_sha256 or state["split"] != self.split:
            raise ValueError("memory cursor restore requires the same task and split")
        position, batches = int(state["position"]), int(state["batch_position"])
        if min(position, batches) < 0:
            raise ValueError("negative memory cursor count")
        self.seed, self.position, self.batch_position = int(state["seed"]), position, batches

    state_dict = snapshot
    load_state_dict = restore


class MemoryTask:
    def __init__(
        self, kind: str = "delayed_recall", *, revision_gap: int = 4, max_delay: int = 128
    ):
        kind = {
            "recall": "delayed_recall",
            "revision": "latest_revision",
            "latest_value_revision": "latest_revision",
        }.get(kind, kind)
        if kind not in {"delayed_recall", "latest_revision"}:
            raise ValueError("unknown raw-byte memory task")
        if (
            not isinstance(revision_gap, int)
            or isinstance(revision_gap, bool)
            or not 0 <= revision_gap <= 128
        ):
            raise ValueError("revision_gap must be an integer in [0,128]")
        if (
            not isinstance(max_delay, int)
            or isinstance(max_delay, bool)
            or not 0 <= max_delay <= 128
        ):
            raise ValueError("max_delay must lie in [0,128]")
        self.kind, self.revision_gap, self.max_delay = kind, revision_gap, max_delay
        contract = {
            "schema": "spf-raw-byte-memory-organism-v1",
            "kind": kind,
            "revision_gap": revision_gap,
            "max_delay": max_delay,
            "payload_bytes": list(PAYLOAD_BYTES),
            "distractors": "IID uniform a-z",
            "delimiters": {"initial_write": ord("^"), "revision": ord("!"), "query": ord("?")},
            "prediction": "raw next-byte at query only",
            "vocabulary": 256,
            "padding": "none",
            "generator": "domain-separated-sha256-seeded-python-random-v1",
        }
        self.task_sha256 = _digest(contract)
        self._contract = contract

    def manifest(self) -> dict:
        return {
            **self._contract,
            "task_sha256": self.task_sha256,
            "default_delays": list(DEFAULT_DELAYS),
            "default_split_seeds": DEFAULT_SPLIT_SEEDS.copy(),
            "balanced_classes": "Every consecutive block of four row indices has one shuffled A/B/C/D target; evaluation batches must be multiples of four.",
            "revision_pairs": "old!=new; cyclic shifts cover all 12 ordered pairs over three aligned four-row blocks",
            "oracle_inputs": False,
            "chance_query_accuracy": 0.25,
            "uniform_four_payload_bits": 2.0,
            "revision_old_only_reference": {
                "conditional_accuracy": 1 / 3,
                "conditional_bits": math.log2(3),
                "scope": "Uniform new value given old!=new; this is not a no-history query model.",
            },
            "paired_delay_grid": "The same row/seed has the same payload and nested IID distractor prefixes across delay cells.",
            "split_limitation": "Split identities/RNG streams differ; the finite raw grammar is not text-disjoint, and delay-zero recall has only four distinct inputs.",
        }

    def cursor(
        self, split: str = "train", *, seed: int | None = None, allow_sealed: bool = False
    ) -> MemoryCursor:
        split = _access(split, allow_sealed)
        return MemoryCursor(
            self.task_sha256, split, DEFAULT_SPLIT_SEEDS[split] if seed is None else int(seed)
        )

    def _rng(self, cursor: MemoryCursor, domain: str, index: int) -> random.Random:
        return random.Random(
            int(_digest([self.task_sha256, cursor.split, cursor.seed, domain, index]), 16)
        )

    def _delays(self, delays: int | Sequence[int]) -> tuple[int, ...]:
        values = (delays,) if isinstance(delays, int) else tuple(delays)
        if not values or len(set(values)) != len(values):
            raise ValueError("delay grid must be nonempty without duplicates")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= self.max_delay
            for value in values
        ):
            raise ValueError(f"delays must be integers in [0,{self.max_delay}]")
        return values

    def sample_batch(
        self,
        batch_size: int = 4,
        delays=DEFAULT_DELAYS,
        *,
        split: str | None = None,
        seed: int | None = None,
        cursor: MemoryCursor | None = None,
        device="cpu",
        allow_sealed: bool = False,
    ):
        split = _access(split or (cursor.split if cursor is not None else "train"), allow_sealed)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if split != "train" and batch_size % len(PAYLOAD_BYTES):
            raise ValueError(
                "feedback/audit batch_size must be a multiple of four for exact balance"
            )
        if cursor is not None and seed is not None:
            raise ValueError("provide memory cursor or seed, not both")
        if cursor is None:
            cursor = self.cursor(split, seed=seed, allow_sealed=allow_sealed)
        if cursor.task_sha256 != self.task_sha256 or cursor.split != split:
            raise ValueError("memory cursor belongs to a different task or split")
        if split != "train" and cursor.position % len(PAYLOAD_BYTES):
            raise ValueError("balanced feedback/audit cursor must begin on a four-row boundary")
        delays = self._delays(delays)
        # Shuffle each complete delay curriculum cycle. Every requested delay
        # appears once per cycle; the whole batch shares one actual shape.
        delay_order = list(delays)
        self._rng(cursor, "delay-cycle", cursor.batch_position // len(delays)).shuffle(delay_order)
        delay = delay_order[cursor.batch_position % len(delays)]
        x_rows, y_rows, mask_rows, metadata = [], [], [], []
        for local in range(batch_size):
            index = cursor.position + local
            order = list(range(len(PAYLOAD_BYTES)))
            self._rng(cursor, "payload-block", index // len(PAYLOAD_BYTES)).shuffle(order)
            payload_class = order[index % len(PAYLOAD_BYTES)]
            payload = PAYLOAD_BYTES[payload_class]
            rng = self._rng(cursor, "distractor-row", index)
            distractors = bytes(ord("a") + rng.randrange(26) for _ in range(delay))
            old_payload = None
            if self.kind == "delayed_recall":
                row = bytes((ord("^"), payload)) + distractors + b"?"
                payload_position, prefix_length = 1, 2
            else:
                shift_offset = self._rng(cursor, "revision-shift", 0).randrange(3)
                shift = 1 + ((index // len(PAYLOAD_BYTES) + shift_offset) % 3)
                old_payload = PAYLOAD_BYTES[(payload_class + shift) % len(PAYLOAD_BYTES)]
                gap_rng = self._rng(cursor, "revision-gap-row", index)
                gap = bytes(ord("a") + gap_rng.randrange(26) for _ in range(self.revision_gap))
                row = (
                    bytes((ord("^"), old_payload))
                    + gap
                    + bytes((ord("!"), payload))
                    + distractors
                    + b"?"
                )
                payload_position, prefix_length = self.revision_gap + 3, self.revision_gap + 4
            targets = list(row[1:]) + [payload]
            mask = [False] * (len(row) - 1) + [True]
            x_rows.append(list(row))
            y_rows.append(targets)
            mask_rows.append(mask)
            alternate_class = (payload_class + 1) % len(PAYLOAD_BYTES)
            if PAYLOAD_BYTES[alternate_class] == old_payload:
                alternate_class = (payload_class + 2) % len(PAYLOAD_BYTES)
            metadata.append(
                {
                    "row_index": index,
                    "payload_byte": payload,
                    "old_payload_byte": old_payload,
                    "alternate_payload_byte": PAYLOAD_BYTES[alternate_class],
                    "payload_position": payload_position,
                    "source_prefix_length": prefix_length,
                    "query_position": len(row) - 1,
                    "delay": delay,
                    "distractor_sha256": hashlib.sha256(distractors).hexdigest(),
                    "raw_input_sha256": hashlib.sha256(row).hexdigest(),
                }
            )
        x = torch.tensor(x_rows, dtype=torch.long, device=device)
        y = torch.tensor(y_rows, dtype=torch.long, device=device)
        loss_mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)
        before = cursor.snapshot()
        cursor.position += batch_size
        cursor.batch_position += 1
        meta = {
            "schema": "spf-raw-byte-memory-batch-v1",
            "kind": self.kind,
            "task_sha256": self.task_sha256,
            "split": split,
            "seed": cursor.seed,
            "delay": delay,
            "delay_grid": list(delays),
            "revision_gap": self.revision_gap,
            "source_prefix_length": metadata[0]["source_prefix_length"],
            "query_position": metadata[0]["query_position"],
            "rows": metadata,
            "batch_size": batch_size,
            "sequence_length": x.shape[1],
            "query_targets": batch_size,
            "input_bytes": x.numel(),
            "padding_bytes": 0,
            "model_inputs": "x raw-byte IDs only",
            "cursor_before": before,
            "cursor_after": cursor.snapshot(),
            "sealed_for_selection": split == "sealed_audit",
        }
        return x, y, loss_mask, meta


def as_memory_batch(x, y, loss_mask, metadata) -> dict:
    """Adapt a task tuple to a frozen source/carrier microscope batch."""
    if x.shape != y.shape or loss_mask.shape != x.shape or loss_mask.dtype != torch.bool:
        raise ValueError("memory batch shape/mask contract changed")
    if not bool((loss_mask.sum(-1) == 1).all()) or len(metadata["rows"]) != x.shape[0]:
        raise ValueError("one query and evidence row are required per input")
    alternate = x.clone()
    for index, row in enumerate(metadata["rows"]):
        query = row["query_position"]
        if (
            not bool(loss_mask[index, query])
            or int(x[index, query]) != ord("?")
            or int(y[index, query]) != row["payload_byte"]
            or int(x[index, row["payload_position"]]) != row["payload_byte"]
        ):
            raise ValueError("memory source/query evidence does not match supplied raw bytes")
        alternate[index, row["payload_position"]] = row["alternate_payload_byte"]
    return {"ids": x, "targets": y[loss_mask], "alternate_ids": alternate, "metadata": metadata}


@torch.no_grad()
def query_observation(logits, y, loss_mask) -> dict:
    """A raw-256-byte query score plus separately scoped four-class metrics."""
    if (
        logits.shape != (*y.shape, 256)
        or loss_mask.shape != y.shape
        or loss_mask.dtype != torch.bool
    ):
        raise ValueError("memory logits must be [B,T,256] with a boolean query mask")
    if not bool((loss_mask.sum(-1) == 1).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("memory observation requires one finite query output per row")
    selected, target = logits[loss_mask], y[loss_mask]
    if not bool(torch.isin(target, target.new_tensor(PAYLOAD_BYTES)).all()):
        raise ValueError("query targets must be raw A-D bytes")
    bits = F.cross_entropy(selected, target, reduction="none") / math.log(2)
    payload_logits = selected[:, PAYLOAD_BYTES]
    payload_target = target - PAYLOAD_BYTES[0]
    conditional_bits = F.cross_entropy(payload_logits, payload_target, reduction="none") / math.log(
        2
    )
    predictions = selected.argmax(-1)
    conditional_prediction = payload_logits.argmax(-1) + PAYLOAD_BYTES[0]
    return {
        "query_bits_per_byte": float(bits.mean()),
        "per_row_query_bits": bits.cpu().tolist(),
        "query_accuracy_raw256": float((predictions == target).float().mean()),
        "predicted_raw_bytes": predictions.cpu().tolist(),
        "target_raw_bytes": target.cpu().tolist(),
        "conditional_payload_bits": float(conditional_bits.mean()),
        "conditional_payload_accuracy": float((conditional_prediction == target).float().mean()),
        "payload_probability_mass": selected.softmax(-1)[:, PAYLOAD_BYTES].sum(-1).cpu().tolist(),
        "chance_payload_accuracy": 0.25,
        "uniform_payload_only_bits": 2.0,
        "query_targets": target.numel(),
        "metric_scope": "raw256 loss/accuracy primary; conditional A-D metrics use known target alphabet for evidence only",
    }


@torch.no_grad()
def score_frozen_grid(
    model,
    task: MemoryTask,
    *,
    delays=DEFAULT_DELAYS,
    split: str = "feedback",
    seed: int | None = None,
    batch_size: int = 4,
    num_batches: int = 8,
    device="cpu",
    allow_sealed: bool = False,
) -> dict:
    """Shared deterministic query-only grid; no fitting or target-fed inputs."""
    split = _access(split, allow_sealed)
    delays = task._delays(delays)
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    # Preserve all module train/eval flags, including selectively frozen blocks.
    modes = {module: module.training for module in model.modules()}
    model.eval()
    rows, summaries = [], []
    try:
        for delay in delays:
            cursor = task.cursor(split, seed=seed, allow_sealed=allow_sealed)
            cell_rows = []
            for batch in range(num_batches):
                x, y, mask, meta = task.sample_batch(
                    batch_size,
                    (delay,),
                    split=split,
                    cursor=cursor,
                    device=device,
                    allow_sealed=allow_sealed,
                )
                observed = query_observation(model(x), y, mask)
                if task.kind == "latest_revision":
                    old = torch.tensor(
                        [row["old_payload_byte"] for row in meta["rows"]], device=y.device
                    )
                    prediction = torch.tensor(observed["predicted_raw_bytes"], device=y.device)
                    observed["old_value_response_fraction"] = float(
                        (prediction == old).float().mean()
                    )
                row = {"delay": delay, "batch": batch, "metadata": meta, "observation": observed}
                cell_rows.append(row)
                rows.append(row)
            summary = {
                "delay": delay,
                "query_targets": sum(row["observation"]["query_targets"] for row in cell_rows),
            }
            for metric in (
                "query_bits_per_byte",
                "query_accuracy_raw256",
                "conditional_payload_bits",
                "conditional_payload_accuracy",
                "old_value_response_fraction",
            ):
                if metric in cell_rows[0]["observation"]:
                    summary[metric] = (
                        sum(
                            row["observation"][metric] * row["observation"]["query_targets"]
                            for row in cell_rows
                        )
                        / summary["query_targets"]
                    )
            summaries.append(summary)
    finally:
        for module, mode in modes.items():
            module.training = mode
    return {
        "schema": "spf-frozen-memory-grid-v1",
        "task": task.manifest(),
        "split": split,
        "seed": DEFAULT_SPLIT_SEEDS[split] if seed is None else int(seed),
        "rows": rows,
        "by_delay": summaries,
        "model_inputs": "raw x only",
        "sealed_for_selection": split == "sealed_audit",
        "terminal_status": "not-assessed",
    }
