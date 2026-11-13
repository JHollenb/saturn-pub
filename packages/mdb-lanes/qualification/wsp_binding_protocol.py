"""Controlled operand/column contrasts and saved-only numerical reductions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROLES = {"a.tens": 46, "a.units": 47, "b.tens": 50, "b.units": 51}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def contrasts(seed, count, prefix, exclude_a=()):
    import numpy as np

    rng, rows, forbidden = np.random.default_rng(seed), [], set(exclude_a)
    attempts = 0
    while len(rows) < count*8:
        attempts += 1
        if attempts > 50000:
            raise ValueError("contrast population exhausted under declared split")
        a = int(rng.integers(1, 7))*100 + int(rng.choice([0, 1, 2, 4, 5, 6, 8, 9]))*10 + int(rng.choice([0, 1, 2, 4, 5, 6, 8]))
        b = int(rng.integers(1, 3))*100 + int(rng.choice([0, 1, 2, 4, 5, 6, 8, 9]))*10 + int(rng.integers(0, 3))
        block = []
        for role in ROLES:
            for digit in (7, 3):
                values = {"a": a, "b": b}
                operand, column = role.split(".")
                scale = 10 if column == "tens" else 1
                values[operand] += (digit-values[operand]//scale % 10)*scale
                aa, bb = values["a"], values["b"]
                matched = role == "a.units" and digit == 7
                original, expected = str(aa+bb), str(aa+bb-4 if matched else aa+bb)
                block.append({"a": aa, "b": bb, "role": role, "digit": digit,
                    "block": len(rows)//8, "matched": matched, "original": original, "expected": expected,
                    "prompt": prefix+str(aa)+"+"+str(bb)+"="})
        if any(r["a"] in forbidden or len(r["original"]) != 3 or r["original"][:2] != r["expected"][:2] for r in block):
            continue
        rows.extend(block)
        forbidden.update(r["a"] for r in block)
    return rows


def reductions(logits, rows, anchor=None):
    import numpy as np

    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError("finite full-vocabulary rows required")
    x = x-x.max(-1, keepdims=True)
    lp = x-np.log(np.exp(x).sum(-1, keepdims=True))
    ids = np.array([15+int(r["expected"][-1]) for r in rows])
    matched = np.array([r["matched"] for r in rows])
    if not matched.any() or matched.all():
        raise ValueError("both target and control rows required")
    target = lp[np.arange(len(rows)), ids]
    kl = np.zeros(len(rows))
    if anchor is not None:
        y = np.asarray(anchor, dtype=np.float64)
        if y.shape != x.shape or not np.isfinite(y).all():
            raise ValueError("aligned finite native anchor required")
        y = y-y.max(-1, keepdims=True)
        base = y-np.log(np.exp(y).sum(-1, keepdims=True))
        kl = (np.exp(base)*(base-lp)).sum(-1)
    return {"objective": float(target[matched].mean()-kl[~matched].mean()),
        "mean_target_logp": float(target[matched].mean()), "control_mean_kl": float(kl[~matched].mean()),
        "target_logp": target.tolist(), "kl": kl.tolist(), "winner_ids": lp.argmax(-1).tolist(),
        "target_final_digit_correct": int((lp.argmax(-1)[matched] == ids[matched]).sum()),
        "control_final_digit_correct": int((lp.argmax(-1)[~matched] == ids[~matched]).sum())}
