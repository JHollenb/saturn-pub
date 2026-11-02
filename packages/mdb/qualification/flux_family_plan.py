"""Frozen geometry and prefix-only resolver for one six-panel MDB assay."""
from __future__ import annotations

import math

FEATURES = ("R_symmetric_relative_rms_delta", "A_symmetric_relative_rms_delta",
            "M_symmetric_relative_rms_delta", "delta_R_A_cosine", "recipient_conditioning_working_cosine")
RECIPIENT_FEATURES = ("R_changed_rms", "A_changed_rms", "M_changed_rms", "changed_R_A_cosine",
                      "recipient_conditioning_working_cosine")
ACTIONS = {"R": "100", "A": "010", "RA": "110"}


def make_plan(prior):
    suffix = " on a plain white background. Two separate objects, simple studio product photograph."
    prompts = {
        "base": "A red cube to the left of a blue sphere" + suffix,
        "relation": "A red cube to the right of a blue sphere" + suffix,
        "color": "A blue cube to the left of a red sphere" + suffix,
        "shape": "A red sphere to the left of a blue cube" + suffix,
        "novel.base": "A green cone to the left of a yellow cylinder" + suffix,
        "novel.relation": "A green cone to the right of a yellow cylinder" + suffix,
        "triple.base": "A red cube to the left of a blue sphere. A small yellow cylinder stands behind both objects on a plain white background. Three separate objects, simple studio product photograph.",
        "triple.relation": "A red cube to the right of a blue sphere. A small yellow cylinder stands behind both objects on a plain white background. Three separate objects, simple studio product photograph.",
    }
    histories = [{"id": "anchor." + name, "seed": 37, "conditioning": "base" if name == "hybrid" else name,
                  "working_text": "relation" if name == "hybrid" else name, "split": "development"}
                 for name in ("base", "hybrid", "relation", "color", "shape")]
    histories += [{"id": cohort + "." + name, "seed": seed, "conditioning": cohort + "." + name,
                   "working_text": cohort + "." + name, "split": "heldout"}
                  for cohort, seed in (("novel", 53), ("triple", 59)) for name in ("base", "relation")]
    pairs = [{"id": name, "source": source, "recipient": recipient, "split": "development", "contrast": contrast}
             for name, source, recipient, contrast in (
                 ("hybrid", "anchor.base", "anchor.hybrid", "relation"),
                 ("natural", "anchor.base", "anchor.relation", "relation"),
                 ("reverse", "anchor.relation", "anchor.base", "relation"),
                 ("color", "anchor.base", "anchor.color", "color-assignment"),
                 ("shape", "anchor.base", "anchor.shape", "shape-assignment"))]
    pairs += [{"id": name, "source": name + ".base", "recipient": name + ".relation",
               "split": "heldout", "contrast": "relation"} for name in ("novel", "triple")]
    rows = [{"id": "ref." + h["id"], "kind": "reference", "history": h["id"], "split": h["split"], "panels": [2, 3, 4]}
            for h in histories]
    rows += [{"id": "sham." + pair, "kind": "copy", "pair": pair,
              "split": "heldout" if pair == "novel" else "development", "panels": [2, 4]}
             for pair in ("hybrid", "natural", "novel")]
    rows += [{"id": pair["id"] + "." + action, "kind": "merge", "pair": pair["id"],
              "mask": mask, "split": pair["split"], "panels": [2 if pair["id"] in {"hybrid", "natural", "reverse"} else 3 if pair["split"] == "development" else 4, 6]}
             for pair in pairs for action, mask in ACTIONS.items()]
    recipes = {"K": {"k_text": "base"}, "V": {"v_text": "base"},
               "KV": {"k_text": "base", "v_text": "base"}, "Q.changedR": {"q": "base"}}
    rows += [{"id": "kv." + name, "kind": "attention", "pair": "hybrid", "recipe": recipe,
              "restore_residual": name != "Q.changedR", "split": "development", "panels": [1]}
             for name, recipe in recipes.items()]
    rows += [{"id": "support." + axis + "." + side, "kind": "support", "pair": "hybrid",
              "axis": axis, "side": side, "split": "development", "panels": [5]}
             for axis in ("heads", "text", "image") for side in ("top", "alternative")]
    rows += [{"id": "break.image", "kind": "break", "pair": "hybrid", "split": "development", "panels": [5]}]
    rows += [{"id": "rescue." + when + "." + support, "kind": "rescue", "pair": "hybrid",
              "when": when, "support": support, "split": "development", "panels": [5]}
             for when, support in (("early", "top"), ("late", "top"), ("early", "alternative"))]
    return {"schema": "mdb-flux-six-panel-plan-v1", "model": prior["model"], "model_path": prior["model_path"],
        "expected_checkpoint_sha256": prior["expected_checkpoint_sha256"], "prompts": prompts,
        "histories": histories, "pairs": pairs, "rows": rows,
        "selector": {"logical_step": 0, "phase": "single", "index": 4},
        "rescue_points": {"early": {"logical_step": 0, "phase": "single", "index": 6},
                          "late": {"logical_step": 1, "phase": "joint", "index": 0}},
        "support_policy": "rank native prefix contrast energy; top quarter and bottom quarter; stable index ties; no outcomes select supports",
        "historical_plan_role": "Pinned checkpoint and prior route-replication evidence; family draft versions are not model outcomes",
        "resolver": {"features": list(FEATURES), "algorithm": "z-scored nearest development prefix; exact ties lexicographic pair id",
                     "recipient_only_features": list(RECIPIENT_FEATURES),
                     "comparison": "paired prefix versus recipient-only state features; both include stored-conditioning/current-text alignment; no heldout outcomes",
                     "abstention": "distance > 2 * maximum development leave-one-out nearest distance; all heldout branches still execute",
                     "targets": list(ACTIONS), "scope": "small empirical resolver, not independent tensor/RGB forecast"},
        "hypotheses": {"merge": "R and A individually preserve recipient assignment; RA restores source assignment; pRA < min(pR,pA)",
            "KV": "Exploratory guess: V-only final projection < K-only; retain both outcomes regardless",
            "Q": "With changed residual, base queries alone preserve recipient ordering",
            "supports": "Exploratory rank: top prefix-contrast supports move toward source more than equal-size alternative supports",
            "rescue": "Reverting top image-row attention outputs to recipient values within the full text-KV edit moves away from clean KV future; early selected-row rescue is closer to clean than broken or alternative-row rescue; late rescue may be weaker",
            "heldout": "Resolver is fit only to development outcomes; all heldout predictions sealed before every heldout reference/edited suffix"},
        "max_images": len(rows), "height": 512, "width": 512, "steps": 4, "cpu_threads": 4,
        "max_adapter_steps": 10000, "max_wall_seconds": 3000,
        "limits": {"max_archive_bytes": 2560 << 20, "max_expanded_bytes": 4096 << 20,
                   "max_files": 20000, "max_member_bytes": 128 << 20, "chunk_bytes": 8 << 20,
                   "max_log_bytes": 3584 << 20},
        "scope": "One resident Klein CUDA BF16 B1 lease;47 dependent native RGB futures; observed source ports and typed image interventions; no independently forecast edited RGB; all six panels and all counterexamples retained",
        "terminal_status": "unassessed"}


def fit_resolver(samples, *, features=FEATURES):
    usable = [row for row in samples if all(row["targets"].get(action) is not None for action in ACTIONS)]
    if len(usable) < 2:
        return {"status": "insufficient-development-contrast", "samples": usable, "features": list(features)}
    usable.sort(key=lambda row: row["pair_id"])
    vectors = [[row["signature"][key] for key in features] for row in usable]
    means = [sum(col) / len(col) for col in zip(*vectors)]
    scales = [math.sqrt(sum((v - mean) ** 2 for v in col) / len(col)) or 1.
              for col, mean in zip(zip(*vectors), means)]
    standardized = [[(v - mean) / scale for v, mean, scale in zip(row, means, scales)] for row in vectors]
    distance = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    loo = [min(distance(row, other) for j, other in enumerate(standardized) if i != j)
           for i, row in enumerate(standardized)]
    return {"status": "frozen", "features": list(features), "means": means, "scales": scales,
            "standardized": standardized, "samples": usable, "abstention_distance": 2 * max(loo), "leave_one_out_distances": loo}


def resolve(model, signature):
    if model["status"] != "frozen":
        return {"status": "abstain", "reason": model["status"], "predictions": None}
    row = [(signature[key] - mean) / scale for key, mean, scale in zip(model["features"], model["means"], model["scales"])]
    distances = [math.sqrt(sum((a - b) ** 2 for a, b in zip(row, sample))) for sample in model["standardized"]]
    index = min(range(len(distances)), key=lambda i: (distances[i], model["samples"][i]["pair_id"]))
    sample = model["samples"][index]
    targets = sample["targets"]
    winner = min(ACTIONS, key=lambda action: (targets[action], action))
    return {"status": "abstain" if distances[index] > model["abstention_distance"] else "predict",
            "reason": "outside-development-neighborhood" if distances[index] > model["abstention_distance"] else "nearest-prefix",
            "distance": distances[index], "nearest_pair": sample["pair_id"], "candidate_predictions": targets,
            "predictions": targets if distances[index] <= model["abstention_distance"] else None,
            "candidate_record": sample["pair_id"] + "." + winner,
            "candidate_family": "single4.merge." + winner,
            "baseline_record": "single4.merge.RA", "baseline": "pRA < min(pR,pA)"}
