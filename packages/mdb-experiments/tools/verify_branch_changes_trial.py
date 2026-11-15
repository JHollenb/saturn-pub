"""Independently recompute finite terms from the saved native endpoints.

No model or mdb analysis imports. Numerical comparisons check report consistency,
never whether an intervention is scientifically interesting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load(path):
    try:
        data = load_file(str(path))
        return data["tensor"] if set(data) == {"tensor"} else data
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    inputs = json.loads((root / "inputs.json").read_text())
    a, b = inputs["answer_ids"]
    baseline = load(root / "baseline-logits.pt").double().reshape(-1)
    errors = []

    def check(actual, expected, label):
        error = abs(float(actual) - float(expected))
        errors.append({"field": label, "abs_error": error})
        # Float64 evaluation-order allowance, not a native-error or effect gate.
        if error > 1e-10 * max(1, abs(float(expected))):
            raise ValueError(f"saved report disagrees with endpoints: {label}: {error}")

    for arm in ("sham", "v", "k", "kv"):
        logits = load(root / arm / "logits.pt").double().reshape(-1)
        outcome = json.loads((root / arm / "outcome.json").read_text())
        check(outcome["margin"], logits[a] - logits[b], f"{arm}.margin")
        check(outcome["delta"], logits[a] - logits[b] - baseline[a] + baseline[b], f"{arm}.delta")
        if arm == "sham" and not torch.equal(logits, baseline):
            raise ValueError("sham does not reproduce baseline logits")
        for layer in (20, 21):
            report = json.loads((root / arm / f"L{layer}.json").read_text())
            ends = load(root / arm / f"L{layer}-endpoints.pt")
            x, y = ends["base"], ends["edited"]
            h = x["p"].shape[1]
            p0, p1 = [v["p"].double().reshape(h, -1) for v in (x, y)]
            cache0, cache1 = [ends["values"][v].double()[0] for v in ("base", "edited")]
            kv = cache0.shape[0]
            index = torch.arange(h) // (h // kv)
            v0, v1 = cache0[index], cache1[index]
            dp, dv = p1 - p0, v1 - v0
            attention = {"payload": torch.einsum("hj,hjd->hd", p0, dv),
                "routing": torch.einsum("hj,hjd->hd", dp, v0),
                "interaction": torch.einsum("hj,hjd->hd", dp, dv)}
            g0, g1 = [v["g"].double().flatten() for v in (x, y)]
            u0, u1 = [v["u"].double().flatten() for v in (x, y)]
            s0, s1 = [torch.nn.functional.silu(g) for g in (g0, g1)]
            mlp = {"gate": u0 * (s1 - s0), "up": s0 * (u1 - u0),
                   "interaction": (s1 - s0) * (u1 - u0)}
            for section, terms, port in (("attention", attention, "A"), ("mlp", mlp, "f")):
                for name, value in terms.items():
                    check(report[section]["terms"][name]["norm"]["value"], value.norm(),
                          f"{arm}.L{layer}.{section}.{name}")
                native = y[port].double().flatten() - x[port].double().flatten()
                check(report[section]["native_discrepancy"]["value"],
                      (native - sum(terms.values()).flatten()).norm(),
                      f"{arm}.L{layer}.{section}.native_discrepancy")
                ranked = report[section]["sources" if section == "attention" else "neurons"]
                check(ranked["total"]["value"],
                    sum(r["total"]["value"] for r in ranked["rows"]) +
                    sum(v["value"] for v in ranked["tail"].values()),
                    f"{arm}.L{layer}.{section}.top_plus_tail")
    summary = json.loads((root / "summary.json").read_text())
    fresh = json.loads((root / "fresh/whatif.json").read_text())
    repair = load(root / "fresh/repair-logits.pt").double().flatten()
    check(fresh["measured"]["edited"]["value"], repair[a] - repair[b], "fresh.independent_repair")
    if not summary["arms"]["fresh_selected_neuron_restore"]["independent_native_matches_whatif"]:
        raise ValueError("independent repair differs from whatif")
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    args.out.write_text(json.dumps({"schema": "mdb-branch-changes-verification-v1",
        "mechanics_status": "verified", "checked_fields": len(errors),
        "max_abs_report_disagreement": max(e["abs_error"] for e in errors),
        "checks": errors, "files": files,
        "scope": "independent finite-term and native-margin checks; no semantic certification"},
        indent=2, sort_keys=True))
    print(json.dumps({"checked_fields": len(errors),
        "max_abs_report_disagreement": max(e["abs_error"] for e in errors)}))


if __name__ == "__main__":
    main()
