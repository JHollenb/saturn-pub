"""Bounded numeric source resolver for the previously identified L20/H5 reader.

No prompt text, reference span, token identity, donor contrast or output label is
accepted. Attention is computed by the native model before this resolver runs.
"""
import math

READER_LAYER = 20
READER_HEAD = 5
CAPTURE_LAYERS = (20, 21, 26)
PAIR = ((20, 0), (20, 1))


def nominate(weights, cached_rows):
    if isinstance(cached_rows, bool) or not isinstance(cached_rows, int) or cached_rows < 0:
        raise ValueError("cached row count must be a nonnegative integer")
    if not weights or cached_rows > len(weights):
        raise ValueError("attention and cache geometry disagree")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError("attention weights must be finite nonnegative numbers")
    total = sum(weights)
    if total <= 0:
        raise ValueError("empty native attention mass")
    ordered = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    row = ordered[0]
    return {"reader_layer": READER_LAYER, "reader_head": READER_HEAD,
        "predicted_row": row, "source_row": row if row < cached_rows else None,
        "status": "addressable-cached-source" if row < cached_rows else "outside-retained-prefix",
        "source_mass": weights[row], "attention_sum": total,
        "top_two_gap": weights[row] - (weights[ordered[1]] if len(ordered) > 1 else 0),
        "rule": "native-attention-argmax-first-index-on-tie"}


def arms():
    return [{"arm": name, "kind": kind, "cells": cells, "dose": 1} for name, kind, cells in (
        ("native-recipient", "native", PAIR), ("native-donor", "donor-native", PAIR),
        ("auto-sham", "sham", PAIR), ("auto-pair", "transfer", PAIR),
        ("auto-reciprocal", "reciprocal", PAIR), ("auto-wrong-payload", "wrong-payload", PAIR),
        ("auto-wrong-target", "wrong-target", PAIR),
        ("auto-full-late", "transfer", tuple((l, k) for l in range(19, 28) for k in range(2))),
        ("fixed-distance-pair", "fixed-distance", PAIR))]
