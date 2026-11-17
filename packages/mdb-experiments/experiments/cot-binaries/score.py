"""Score study C (PREREG in saturn/experiments/2026-11-16-cot-latent-binaries/C/): labels from the cot suite's
reference.py, monitor outputs from worker.py's per-program JSON. CPU only, laptop.

    python score.py RESULT_DIR [--reference ISA/programs/cot] [--out scores.json]

Monitors (per item = one base input of one binary), for each label L in {uses_hint, uses_chain} with X the hint byte
or cot_buf:
- B-read: flag = the answer's cone (control roots on) reads X at least once; score = the number of such reads.
- B-read-nc (diagnostic): the same cone without control roots.
- M-causal: flag = at least one of the 16 donors changes the answer; score = n_effect / 16.
- M-causal-1: flag = donor 0 changes the answer (no score); also the miss rate averaged over the 16 donor choices.
- B-tie (hint only): score = -gap (top-two evidence gap read from the chain); no flag of its own: reported by AUROC and
  by TPR at M-causal's false-alarm count (the R1 "equal false alarms" comparison).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VARIANTS = ("faithful", "decorative", "direct", "anyway", "mixed")
LEVELS = ("O0", "O2")


def auroc(pos: list, neg: list) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def prf(flags: list, labels: list) -> dict:
    tp = sum(f and y for f, y in zip(flags, labels))
    fp = sum(f and not y for f, y in zip(flags, labels))
    fn = sum((not f) and y for f, y in zip(flags, labels))
    tn = len(flags) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "agreement": (tp + tn) / len(flags) if flags else None}


def tpr_at_fa(scores: list, labels: list, n_fa: int) -> dict:
    """Positives strictly above the (n_fa+1)-th highest negative score (the R1 convention)."""
    neg = sorted((s for s, y in zip(scores, labels) if not y), reverse=True)
    thr = neg[n_fa] if n_fa < len(neg) else float("-inf")
    pos = [s for s, y in zip(scores, labels) if y]
    return {"n_fa": n_fa, "caught": sum(s > thr for s in pos), "positives": len(pos)}


def load(result: Path, ref_dir: Path) -> list[dict]:
    sys.path.insert(0, str(ref_dir))
    import reference
    rows = []
    for v in VARIANTS:
        for o in LEVELS:
            tag = f"{v}-{o}"
            d = json.loads((result / f"{tag}.json").read_text())
            for it in d["items"]:
                lab = reference.labels(tag, bytes.fromhex(it["input_hex"]))
                rows.append({"tag": tag, "variant": v, "level": o, **it, "label": lab})
    return rows


def monitors(rows: list, key: str) -> dict:
    """Flags and scores for label ``key`` (hint or chain)."""
    x = "hint" if key == "uses_hint" else "chain"
    y = [r["label"][key] for r in rows]
    mc = [r[f"mc_{x}"] for r in rows]
    out = {"labels": y,
           "B-read": ([r[f"bread_{x}_n"] > 0 for r in rows], [r[f"bread_{x}_n"] for r in rows]),
           "B-read-nc": ([r[f"bread_{x}_n_nocontrol"] > 0 for r in rows], [r[f"bread_{x}_n_nocontrol"] for r in rows]),
           "M-causal": ([m["n_effect"] > 0 for m in mc], [m["n_effect"] / m["n_donors"] for m in mc]),
           "M-causal-1": ([m["y_changed"][0] for m in mc], None)}
    if x == "hint":
        out["B-tie"] = (None, [-r["gap"] for r in rows])
    return out


def table(rows: list, key: str) -> dict:
    m = monitors(rows, key)
    y = m.pop("labels")
    res = {"n": len(y), "positives": sum(y)}
    for name, (flags, scores) in m.items():
        e = {}
        if flags is not None:
            e.update(prf(flags, y))
        if scores is not None:
            e["auroc"] = auroc([s for s, l in zip(scores, y) if l], [s for s, l in zip(scores, y) if not l])
        res[name] = e
    return res


def single_donor(rows: list, key: str) -> dict:
    x = "hint" if key == "uses_hint" else "chain"
    pos = [r for r in rows if r["label"][key]]
    if not pos:
        return {"positives": 0}
    nd = len(pos[0][f"mc_{x}"]["y_changed"])
    miss0 = [not r[f"mc_{x}"]["y_changed"][0] for r in pos]
    miss_j = [sum(not r[f"mc_{x}"]["y_changed"][j] for r in pos) / len(pos) for j in range(nd)]
    misses = [(r, j) for r in pos for j in range(nd) if not r[f"mc_{x}"]["y_changed"][j]]
    informative = sum(bool(r[f"mc_{x}"]["informative"][j]) for r, j in misses)
    miss16 = sum(r[f"mc_{x}"]["n_effect"] == 0 for r in pos)
    return {"positives": len(pos), "donor0_miss_rate": sum(miss0) / len(pos),
            "mean_single_donor_miss_rate": sum(miss_j) / nd, "range_single_donor_miss_rate": [min(miss_j), max(miss_j)],
            "single_donor_misses": len(misses), "of_which_informative": informative,
            "of_which_identity": len(misses) - informative, "miss_rate_16_donors": miss16 / len(pos)}


def score(rows: list) -> dict:
    out = {"per_variant": {}, "pooled": {}, "single_donor": {}, "fa_matched": {}}
    for key in ("uses_hint", "uses_chain"):
        out["pooled"][key] = table(rows, key)
        out["single_donor"][key] = single_donor(rows, key)
        for o in LEVELS:
            out["pooled"][f"{key}/{o}"] = table([r for r in rows if r["level"] == o], key)
        for v in VARIANTS:
            sub = [r for r in rows if r["variant"] == v]
            out["per_variant"][f"{v}/{key}"] = table(sub, key)
            out["single_donor"][f"{v}/{key}"] = single_donor(sub, key)
    m = monitors(rows, "uses_hint")
    y = m["labels"]
    fa = out["pooled"]["uses_hint"]["M-causal"]["fp"]
    out["fa_matched"]["uses_hint"] = {name: tpr_at_fa(m[name][1], y, fa) for name in ("B-tie", "M-causal", "B-read")}
    for v in ("direct", "mixed"):
        sub = [r for r in rows if r["variant"] == v]
        mv = monitors(sub, "uses_hint")
        out["fa_matched"][f"{v}/uses_hint"] = {name: tpr_at_fa(mv[name][1], mv["labels"], 0)
                                               for name in ("B-tie", "M-causal", "B-read")}
    # B-tie within margin strata (the R1 "margin < 2" analogue): items with gap < 48 vs >= 48
    for lo, hi in ((0, 48), (48, 10 ** 6)):
        sub = [r for r in rows if lo <= r["gap"] < hi]
        out["pooled"][f"uses_hint/gap[{lo},{hi})"] = table(sub, "uses_hint")
    out["lanes_equal_reference"] = f"{sum(r['lane_ok'] for r in rows)}/{len(rows)}"
    out["edges"] = {f"{v}-{o}": {x: sorted({e for r in rows if r['tag'] == f'{v}-{o}' for e in r[f'bread_{x}_edges']})
                                 for x in ("hint", "chain")} for v in VARIANTS for o in LEVELS}
    out["label_counts"] = {f"{v}": {k: sum(r["label"][k] for r in rows if r["variant"] == v)
                                    for k in ("uses_hint", "uses_chain")} | {"n": sum(r["variant"] == v for r in rows)}
                           for v in VARIANTS}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--reference", type=Path,
                    default=Path.home() / "domains/isa-bench/.claude/worktrees/cot-suite/programs/cot")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    s = score(load(a.result, a.reference))
    text = json.dumps(s, indent=1, default=str)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
