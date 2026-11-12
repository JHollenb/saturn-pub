"""Rounding bounds for every reported value-coalition margin coefficient (artifact-only, numpy).

Each arm's hot - cold margin comes from BF16 logits. A stored BF16 logit is within half a BF16
spacing of the value its FP32 accumulation produced, so every reported coefficient (singleton
effect, complement loss, paired effect, two-cell interaction, KV-group interaction, full effect,
nonadditivity) carries a worst-case readout-rounding bound equal to the sum over its logits of
|coefficient| x half-spacing. Coefficients whose magnitude does not exceed that bound are flagged
UNRESOLVED. This bound covers the final readout rounding only; BF16 rounding upstream of the
readout is not bounded here and needs an FP32 rerun.

Optional FP64 readout: with ``--readout-rows`` (or ``--checkpoint``) the script recomputes the
hot/cold logits of every arm in FP64 from the stored post-norm final hidden state (BF16, at each
arm's readout cut) and the three tied head rows, removing the readout rounding entirely, and
prints each coefficient from those logits beside the BF16 value. Only three 1536-wide rows are
read; no model is loaded.

  python value_coalition_bounds.py /path/to/mdb-value-coalition-20261112/main-02 \
      [--readout-rows rows.json] [--checkpoint /path/to/model.safetensors]

With --checkpoint the file's sha256 is checked against the run's parameter artifacts and the rows
are taken from it (or, together with --readout-rows, the rows file is compared to it); with
--readout-rows alone the rows are trusted on their recorded hash and the output says NOT VERIFIED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdb.analysis_metrics import Bounded, combine, infer_step, rounding_bound  # noqa: E402

TOKENS = {"hot": 4017, "cold": 9255, "boiling": 49837}
LATE = [(layer, kv) for layer in range(19, 28) for kv in range(2)]
WIDTH = 1536


def bf16_to_f64(bits):
    return (np.asarray(bits, dtype="<u2").astype("<u4") << 16).view("<f4").astype(np.float64)


def round_bf16(value):
    bits = np.array([value], dtype="<f4").view("<u4")
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    return float((bits.astype("<u4") << 16).view("<f4")[0])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def native_hidden(store, cut_id):
    """Post-norm final hidden (BF16 bits) recorded in a readout cut, hash-checked."""
    cut = json.loads((store / "cuts" / (cut_id + ".json")).read_text())
    page_path = store / "pages" / (cut["slots"]["hidden"] + ".json")
    if hashlib.sha256(page_path.read_bytes()).hexdigest() != page_path.stem:
        raise ValueError("native page digest mismatch")
    page = json.loads(page_path.read_text())
    blob = store / "blobs" / (page["tensor_blob"] + ".safetensors")
    raw = blob.read_bytes()
    if hashlib.sha256(raw).hexdigest() != blob.stem:
        raise ValueError("native tensor blob digest mismatch")
    n = int.from_bytes(raw[:8], "little")
    item = json.loads(raw[8:8 + n])[page["encoded"]["name"]]
    if item["dtype"] != "BF16" or item["shape"][-1] != WIDTH:
        raise ValueError("unexpected hidden contract")
    lo, hi = item["data_offsets"]
    return np.frombuffer(raw[8 + n + lo:8 + n + hi], dtype="<u2").reshape(-1)


def rows_from_checkpoint(path, expected_sha):
    if sha256(path) != expected_sha:
        raise ValueError("checkpoint sha256 differs from the run's parameter artifacts")
    with Path(path).open("rb") as stream:
        n = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(n))
        entry = header["model.embed_tokens.weight"]  # Qwen2.5-1.5B ties the LM head to this
        if entry["dtype"] != "BF16" or entry["shape"][1] != WIDTH:
            raise ValueError("unexpected embedding contract")
        rows = {}
        for token in TOKENS.values():
            stream.seek(8 + n + entry["data_offsets"][0] + token * WIDTH * 2)
            rows[token] = np.frombuffer(stream.read(WIDTH * 2), dtype="<u2")
    return rows, {"source": str(path), "sha256": expected_sha, "verified_here": True}


def rows_from_file(path, expected_sha):
    data = json.loads(Path(path).read_text())
    if data["sha256"] != expected_sha:
        raise ValueError("row file was not extracted from the run's checkpoint")
    rows = {int(k): np.frombuffer(bytes.fromhex(v), dtype="<u2") for k, v in data["rows"].items()}
    return rows, {"source": str(path), "sha256": data["sha256"],
                  "verified_here": False, "note": "checkpoint hash recorded by the extractor"}


def select_rows(readout_rows, checkpoint, expected_sha):
    """Head rows plus a provenance record saying whether they were verified against a checkpoint.

    Checkpoint only: rows are read from the file after its sha256 matched. Rows file only: the
    recorded hash is trusted and the result is labelled NOT verified. Both: the checkpoint file is
    hash-checked and every extracted row must equal the checkpoint's row byte for byte.
    """
    if checkpoint and not readout_rows:
        rows, provenance = rows_from_checkpoint(checkpoint, expected_sha)
        provenance["verification"] = "VERIFIED: checkpoint sha256 matched the run and rows were read from it"
        return rows, provenance
    rows, provenance = rows_from_file(readout_rows, expected_sha)
    if not checkpoint:
        provenance["verification"] = ("NOT VERIFIED here: rows file trusted on its recorded sha256; "
                                      "pass --checkpoint to check it against the checkpoint file")
        return rows, provenance
    reference, _ = rows_from_checkpoint(checkpoint, expected_sha)
    if set(rows) != set(reference) or any(not np.array_equal(rows[t], reference[t]) for t in reference):
        raise ValueError("extracted head rows differ from the checkpoint file's rows")
    provenance.update(verified_here=True, verification=(
        "VERIFIED: checkpoint sha256 matched the run and the rows file equals the checkpoint's rows"))
    return rows, provenance


def coefficients(margin):
    """Every coefficient in trends.json, as linear combinations of per-arm margins."""
    m0, full = margin["native-recipient"], margin["donor-late"]
    out = {"full_effect": full - m0}
    singles = []
    for layer, kv in LATE:
        single = margin[f"only-L{layer}-KV{kv}"] - m0
        singles.append(single)
        out[f"singleton L{layer}/KV{kv}"] = single
        out[f"complement_loss L{layer}/KV{kv}"] = full - margin[f"without-L{layer}-KV{kv}"]
    for layer in range(19, 28):
        both = margin[f"only-L{layer}-both"]
        out[f"paired_effect L{layer}"] = both - m0
        out[f"interaction L{layer}"] = (both - margin[f"only-L{layer}-KV0"]
                                        - margin[f"only-L{layer}-KV1"] + m0)
    out["KV_group_interaction"] = full - margin["only-KV0"] - margin["only-KV1"] + m0
    out["sum_singleton_effects"] = combine((1, s) for s in singles)
    out["nonadditivity (full - sum singletons)"] = out["full_effect"] - out["sum_singleton_effects"]
    return out


def reported(trends):
    rep = {"full_effect": trends["full_effect"], "KV_group_interaction": trends["KV_group_interaction"],
           "sum_singleton_effects": trends["sum_singleton_effects"],
           "nonadditivity (full - sum singletons)": trends["full_effect"] - trends["sum_singleton_effects"]}
    for c in trends["cells"]:
        rep[f"singleton L{c['layer']}/KV{c['kv_head']}"] = c["singleton_effect"]
        rep[f"complement_loss L{c['layer']}/KV{c['kv_head']}"] = c["complement_loss"]
    for r in trends["layer_interactions"]:
        rep[f"paired_effect L{r['layer']}"] = r["paired_effect"]
        rep[f"interaction L{r['layer']}"] = r["interaction"]
    return rep


def main(run, readout_rows=None, checkpoint=None):
    run = Path(run)
    arms = {a["arm"]: a for a in json.loads((run / "analysis/arms.json").read_text())}
    trends = json.loads((run / "analysis/trends.json").read_text())
    logits = np.array([arms[a]["words"][w]["logit"] for a in arms for w in ("hot", "cold")])
    step = infer_step(logits)
    print(f"stored hot/cold logits: {len(logits)} values in [{logits.min()}, {logits.max()}]; "
          f"largest power-of-two grid dividing all = {step}")
    half = {}
    for name, arm in arms.items():
        for word in ("hot", "cold"):
            value = arm["words"][word]["logit"]
            half[(name, word)] = Bounded(value, float(rounding_bound([value], dtype="bfloat16")[0]),
                                         key=(name, word))
    margin = {name: half[(name, "hot")] - half[(name, "cold")] for name in arms}
    for name, arm in arms.items():
        if margin[name].value != arm["hot_minus_cold"]:
            raise ValueError("margin recomputation differs from arms.json: " + name)
    coeffs = coefficients(margin)
    expected = reported(trends)
    for name, value in expected.items():
        if abs(coeffs[name].value - value) > 1e-12:
            raise ValueError("coefficient recomputation differs from trends.json: " + name)
    fp64 = None
    if readout_rows or checkpoint:
        sha = json.loads((run / "results/parameter-artifacts.json").read_text())[
            "provenance"]["weight_files_sha256"]["model.safetensors"]
        rows, provenance = select_rows(readout_rows, checkpoint, sha)
        rows = {t: bf16_to_f64(r) for t, r in rows.items()}
        store = run / "results/project/store"
        exact, worst = 0, 0.0
        fp64_margin = {}
        for name in arms:
            arm = json.loads((run / "results/arms" / (name + ".json")).read_text())
            hidden = bf16_to_f64(native_hidden(store, arm["trajectory"][0]["readout_cut"]))
            value = {w: float(hidden @ rows[TOKENS[w]]) for w in ("hot", "cold")}
            for w in ("hot", "cold"):
                stored = arms[name]["words"][w]["logit"]
                worst = max(worst, abs(value[w] - stored) / half[(name, w)].bound)
                exact += int(round_bf16(value[w]) == stored)
            fp64_margin[name] = Bounded(value["hot"] - value["cold"], 0.0)
        fp64 = coefficients(fp64_margin)
        print(f"FP64 readout from stored post-norm hidden x tied head rows; rows from "
              f"{provenance['source']} (checkpoint sha256 {provenance['sha256'][:12]}..., "
              f"{provenance['verification']}); {exact}/{2 * len(arms)} stored logits "
              f"equal the BF16 rounding of the FP64 readout "
              f"(worst |fp64 - stored| / half-spacing = {worst:.3f})")
    print()
    header = f"{'coefficient':42s} {'BF16 value':>11s} {'bound':>7s}  {'status':10s}"
    if fp64:
        header += f" {'FP64 readout':>13s} {'shift':>8s}"
    print(header)
    unresolved = 0
    for name, value in coeffs.items():
        status = "resolved" if value.resolved else "UNRESOLVED"
        unresolved += not value.resolved
        line = f"{name:42s} {value.value:11.4f} {value.bound:7.4f}  {status:10s}"
        if fp64:
            line += f" {fp64[name].value:13.4f} {fp64[name].value - value.value:8.4f}"
        print(line)
    print(f"\n{unresolved}/{len(coeffs)} coefficients do not exceed their BF16 readout-rounding bound.")
    print("Bounds cover readout rounding only; BF16 rounding upstream of the readout needs an FP32 rerun.")
    return {"coefficients": {k: v.as_dict() for k, v in coeffs.items()},
            "fp64_readout": None if fp64 is None else {k: v.value for k, v in fp64.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--readout-rows", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    main(**vars(parser.parse_args()))
