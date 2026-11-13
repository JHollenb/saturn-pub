"""Convert the saved FLUX.2 Klein read-graph experiment into mdb symbol-read records. No model run.

    python flux_read_graph_symbols.py --out NEW_DIR [--experiment DIR]

Inputs (read only): ``runs/full-a/result/{tokens.json, capture.<prompt>.<seed>.npz}`` (``tok_w``:
[step, block, head, 512 text rows] image<-row write mass, mean over image receivers),
``head_sets3.json``, ``runs/headfree/{plan.json, headfree_table.json}``,
``runs/headx/{plan.json, headx_table.json}``, ``runs/headcut-cat/headcut_table.json`` (cat and
car rows) and ``runs/headcut-{cat,car}/plan.json`` (other-job random-60 references), ``runs/*/submission.json`` (job ids),
``AUDIT.md`` (picture judgements).

Output (a NEW directory, required): ``lifecycles.json`` (one canonical SymbolLifecycle per test
word: a supplied binding over the word's text rows and one observed use per own60 head),
``reads.json`` (``SymbolRead`` per use, causal status ``unknown``; ``ReadSet`` per (word, head
set) with set-level cut/keep results), ``summary.json`` (counts per status and per contrast)
and an offline view. Files are referenced by locator (root ``flux-read-graph`` + relative path)
and digest (sha256 + size), never copied.

The verdict rule is written once, in ``VERDICT_RULE`` below; ``meta.verdict_rule`` stores it
with its id and sha256, and every result names that id and hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mdb.attention_ranking import contrast_ranking
from mdb.symbol_reads import (
    ALTERNATIVES,
    CLAIM_SCOPES,
    CLAIMS,
    SIZE_MATCH_MIN_RATIO,
    CausalResult,
    PassiveSurface,
    Reader,
    ReadSet,
    ReadSource,
    declare_root,
    file_pointer,
    file_sha256,
    model_identity,
    passive_binding,
    passive_definition,
    passive_lifecycle,
    passive_use,
    read_for_use,
    save_reads,
    verdict_rule,
)

DEFAULT_EXPERIMENT = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
ROOT_NAME = "flux-read-graph"
TEST = {"cat": ["red", "black", "ball", "snow"], "car": ["night", "rain"]}
SEEDS = (4242, 9001)
JOINT, BLOCKS, HEADS, STEPS = 5, 25, 24, 4
SCORE = ("gone = (d(arm) - d(iso)) / (d(none) - d(iso)), d = 1 - MAD(arm, deleted) / "
         "MAD(native, deleted); 0 = word intact, 1 = word as gone as with every read cut")
CUT_CLAIM = ("cutting this set's reads of the word's own rows (receivers: image rows and other "
             "text rows) removes or weakens the word")
KEEP_CLAIM = "with only this set reading the word's own rows, the word is carried"

# Independent contrasts (AUDIT X2, L "Independent contrasts"): red and black share their head
# pools and 29/60 own heads, so they are one contrast read out twice.
CONTRASTS = {"cat:red": "colour-words", "cat:black": "colour-words", "cat:ball": "ball",
             "car:night": "night", "cat:snow": "snow", "car:rain": "rain"}
INDEPENDENT_CUT_CONTRASTS = ("colour-words", "ball", "night")
CONTRAST_NOTES = {
    "colour-words": "red and black (cat prompt): same group pools, 29/60 shared own60 heads; one contrast",
    "ball": "ball (cat prompt)",
    "night": "night (car prompt, second ranking)",
    "snow": "no cut removes ground snow (AUDIT X2, L3); keep arms only, no random keep baseline",
    "rain": "no cut removes rain (AUDIT L3); keep arms only, no random keep baseline",
}
# Pixel score unusable; picture calls are kept and decide, numbers are omitted.
PIXEL_UNUSABLE = {"snow": "AUDIT.md L3/X2: snow pixel score follows sky/background, ground snow never removed",
                  "rain": "AUDIT.md L3: rain pixel score ranges -.36 to 1.50 while streaks never change"}
# Words no cut removes cannot witness "these units do not damage any word" (other-item control).
NO_OTHER_ITEM_WITNESS = ("snow", "rain")

# Audited picture grids, transcribed from AUDIT.md (seed 4242, seed 9001). "-" = the audit's minus.
# Follow-up: label-free head cut, "Picture grid" (headfree, job-a0bca63168b4).
W6 = ("red", "black", "ball", "snow", "night", "rain")
HEADFREE_PICTURES = {
    "own60": ("N- mauve|P brown", "P|P", "Y- 64%|Y- 69%", "Y|Y", "P dusk 46%|Y- 22%", "Y|Y"),
    "own120": ("N- mauve|P brown", "N- grey tabby|N- grey tabby", "Y- 64%|Y- 69%", "Y|Y", "P 55%|P- 31%", "Y|Y"),
    "own240": ("N grey|N grey", "N|N", "Y- 63%|Y- 45%", "Y|Y", "N mostly day 77%|P 50%", "Y|Y"),
    "rand120": ("Y|Y", "Y|Y", "Y 112%|Y 102%", "Y|Y", "Y-|Y", "Y|Y"),
    "rand240": ("Y|Y", "Y|Y", "Y 90%|Y- 66%", "Y|Y", "P- twilight 31%|Y 12%", "Y|Y"),
    "mod": ("N slate|N grey", "N|N", "Y|Y", "Y|Y", "P 43%|Y- 19%", "Y|Y"),
    "noun": ("Y|Y", "Y|Y", "Y- 61%|Y- 57%", "Y|Y", "Y|Y", "Y|Y"),
    "modI": ("P mauve|P mauve", "Y- 15%|Y- 18%", "Y|Y", "Y|Y", "Y|Y", "Y|Y"),
    "nounI": ("Y|Y", "Y|Y", "Y- 69%|Y- 79%", "Y|Y", "Y|Y", "Y|Y"),
    "keep-own60": ("Y|Y", "Y|Y", "Y|Y", "Y thinner|Y thinner", "P-N 58%|P 45%", "Y- sparser|Y- sparser"),
    "keep-own120": ("Y|Y", "Y|Y", "Y|Y", "Y|Y", "P 48%|P- 32%", "Y|Y"),
    "keep-own240": ("Y|Y", "Y|Y", "Y|Y", "Y|Y", "Y-/P 33%|Y- 17%", "Y|Y"),
    "keep-modI": ("Y|Y", "Y|Y", "N|N", "Y- smooth|Y- smooth", "N 79%|N- 63%", "N|N"),
    "keep-nounI": ("N grey|N grey", "N grey|N grey", "Y- 79%|N", "Y- smooth|Y- smooth", "N|N", "N faint traces|N faint traces"),
    "m:own60": ("P dark red|Y", "P 16%|Y- 19%", "Y- 73%|Y- 72%", "Y|Y", "P 44%|Y- 20%", "Y|Y"),
}
# Follow-up: cross-category head cut, "Picture grid" (headx, job-2cecfaa04a49). No rain word.
W5 = ("red", "black", "ball", "snow", "night")
HEADX_PICTURES = {
    "color": ("N slate|N bronze", "P dark grey|P dark grey", "Y full|Y full", "Y|Y", "P dusk|Y-"),
    "object": ("Y|Y", "Y|Y", "Y- small 57%|Y- small 61%", "Y|Y", "Y|Y"),
    "scene": ("Y|Y", "Y|Y", "Y- medium 72%|Y- medium 63%", "Y|Y", "Y|Y"),
    "light": ("Y- duller|Y- duller", "Y|Y", "Y|Y", "Y|Y", "P dusk|Y-"),
    "attribute": ("N mauve|P", "N grey|N grey", "Y|Y", "Y|Y", "Y-|Y-"),
    "heavy60": ("Y|Y", "Y|Y", "Y|Y", "Y|Y", "Y-|Y"),
    "roleM": ("Y|Y", "Y very slightly lighter|Y very slightly lighter", "Y|Y", "Y|Y", "Y|Y"),
    "ownX": ("Y|Y", "Y|Y", "Y|Y", "Y|Y", "Y|Y"),
    "keep-color": ("Y|Y", "Y|Y", "N|N", "Y-|Y", "N|N"),
    "keep-object": ("N|N", "N grey|N grey", "Y|Y", "P|Y-", "N|N"),
    "keep-scene": ("N|N", "N grey|N grey", "Y|Y", "Y-|Y", "N|N"),
    "keep-light": ("P mottled|Y-", "Y|Y", "N|N", "P icy|N wet", "N|N"),
    "keep-attribute": ("Y|Y", "Y|Y", "Y- small|N", "Y|Y", "N|N"),
}
HEADX_CATEGORIES = ("color", "object", "scene", "light", "attribute")
HEADX_X = ("colorX", "objectX", "sceneX", "lightX", "attributeX")

# ----------------------------------------------------------------------------- the verdict rule
# Named parameters (the only thresholds the rule uses).
LEVELS = {"Y": 0, "Y-": 1, "P": 2, "N": 3}
CUT_EFFECT_MIN_LEVEL = 1        # cut: effect present iff every seed >= 1; absent iff every seed is 0
KEEP_CARRIED_MAX_LEVEL = 1      # keep: carried iff every seed <= 1; lost iff every seed >= 2
REFERENCE_CUT_CLEAN_MAX_ABS = .15   # other-job pixel reference (cut): |gone| <= .15 reads "clean"
REFERENCE_KEEP_LOST_MIN = .7        # other-job pixel reference (keep): gone >= .7 reads "word lost"

_RANDOM = (["set_size", "joint_single_counts"],
           "uniform random draw with the claim set's joint/single counts; not mass-matched, may overlap it")
_POOL = (["other_pool", "same_ranking_procedure"],
         "the other pool from the same Addendum 5 rankings (test prompt left out); not a random draw")
_HEAVY = (["set_size", "joint_single_counts"],
          "60 heaviest readers outside the own set, joint/single counts matched; holds less word mass (AUDIT H2)")
# (run, intervention, set) -> the ONE in-run control matched for the claim; it decides.
COUNTED_CONTROLS = {
    ("headfree", "cut", "own120"): ("rand120", *_RANDOM),
    ("headfree", "cut", "own240"): ("rand240", *_RANDOM),
    ("headfree", "cut", "mod"): ("noun", *_POOL), ("headfree", "cut", "noun"): ("mod", *_POOL),
    ("headfree", "cut", "modI"): ("nounI", *_POOL), ("headfree", "cut", "nounI"): ("modI", *_POOL),
    **{("headx", "cut", c): ("heavy60", *_HEAVY) for c in HEADX_CATEGORIES},
}
# In-run arms that are NOT matched for the claim. They never make a result supported; when no
# counted control exists and a screen shows the effect on any seed, the result is unresolved.
SCREENS = {
    ("headfree", "cut", "own60"): [("rand120", ["joint_single_counts"], "larger random draw (120 vs 60); not size-matched"),
                                   ("rand240", ["joint_single_counts"], "larger random draw (240 vs 60); not size-matched")],
    ("headfree", "keep", "modI"): [("keep-nounI", ["other_pool"], "other-pool keep; not a size-matched random keep")],
    ("headfree", "keep", "nounI"): [("keep-modI", ["other_pool"], "other-pool keep; not a size-matched random keep")],
}
# Other-job references (Addendum 4 random-60 draws, matched to role60's joint/single counts,
# judged by a two-seed pixel mean, different worker sha). Recorded, never counted (AUDIT L1).
OTHER_JOB = {("headfree", "cut", "own60"): "rand60a", ("headfree", "keep", "own60"): "keep-rand60a",
             **{("headx", "keep", c): "keep-rand60a" for c in HEADX_CATEGORIES}}
X1_OVERRIDE = "AUDIT.md X1: with the shared heads removed nothing is left (every <cat>X arm pixel <= .13)"

RULE = {
    "summary": "picture calls decide (AUDIT.md); scores are context; one in-run matched control decides",
    "parameters": {"levels": LEVELS, "cut_effect_min_level": CUT_EFFECT_MIN_LEVEL,
                   "keep_carried_max_level": KEEP_CARRIED_MAX_LEVEL,
                   "reference_cut_clean_max_abs_gone": REFERENCE_CUT_CLEAN_MAX_ABS,
                   "reference_keep_lost_min_gone": REFERENCE_KEEP_LOST_MIN,
                   "size_match_min_ratio": SIZE_MATCH_MIN_RATIO},
    "steps": [
        "1. Each seed's picture call maps to a level (parameters.levels; Y-/P mixtures take the first token).",
        "2. Set state. cut: present if every seed >= cut_effect_min_level, absent if every seed is 0. "
        "keep: present (carried) if every seed <= keep_carried_max_level, absent (lost) if every seed >= 2. "
        "Otherwise mixed -> unresolved. Absent -> not_supported.",
        "3. No picture call: a headx <cat>X cut (not snow) takes the AUDIT X1 sentence -> not_supported, "
        "basis 'audit_statement'; anything else -> unresolved, basis 'none'.",
        "4. Present, with a counted control (counted_controls): the control is judged by the SAME thresholds "
        "(symmetric: a Y- control counts as a weak effect). Control absent -> supported. Control present and at "
        "least as strong on every seed -> not_supported. Otherwise -> unresolved.",
        "5. Present, no counted control: if an in-run screen (screens) shows the effect on any seed "
        "(cut: level >= 1; keep: level <= 1) -> unresolved; else -> control_missing.",
        "6. Other-job references (other_job_references) are recorded under reference_controls with a pixel "
        "judgement (reference_* thresholds) and never change a verdict.",
        "7. snow and rain: pixel score unusable; picture calls decide; effect, floor and control numbers omitted "
        "(basis 'picture_only').",
        "8. Alternatives (necessary: generic_damage, mass, set_size; sufficient: mass, set_size). A clean "
        "counted control rules out set_size if its size >= size_match_min_ratio x the set's, and mass only if it "
        "is mass-matched or holds at least the set's mass on every seed (attention_panels: mass_relation 'more'). "
        "Alternatives are ruled out only when the set's effect is present. An other-item control (the same units "
        "cut on another informative word's rows in the same run, picture Y/Y) rules out generic_damage. "
        "claim_scope on supported: needed_set / sufficient_set unless nothing remains (identity).",
    ],
    "counted_controls": [{"run": r, "intervention": k, "set": s, "control": c, "matched_on": m, "how": h}
                         for (r, k, s), (c, m, h) in COUNTED_CONTROLS.items()],
    "screens": [{"run": r, "intervention": k, "set": s, "arms": [a for a, _, _ in v]} for (r, k, s), v in SCREENS.items()],
    "other_job_references": [{"run": r, "intervention": k, "set": s, "arm": a} for (r, k, s), a in OTHER_JOB.items()],
    "x1_override": {"arms": list(HEADX_X), "excluded_words": ["snow"], "verdict": "not_supported",
                    "basis": "audit_statement", "source": X1_OVERRIDE},
    "pixel_unusable_words": PIXEL_UNUSABLE,
}
VERDICT_RULE = verdict_rule("flux-read-graph-picture-rule-v2", RULE)


def level(call):
    token = call.split()[0]
    if token.startswith("Y-"):
        return LEVELS["Y-"]
    return LEVELS[token[0]]


def pictures(grid, words, arm, word):
    row = grid.get(arm)
    if row is None or word not in words:
        return None
    calls = tuple(row[words.index(word)].split("|"))
    return {"calls": list(calls), "levels": [level(c) for c in calls]}


def judge_set(kind, levels):
    """present / absent / mixed for a set OR a control (one rule for both: symmetric)."""
    if kind == "cut":
        if min(levels) >= CUT_EFFECT_MIN_LEVEL:
            return "present"
        return "absent" if max(levels) == 0 else "mixed"
    if max(levels) <= KEEP_CARRIED_MAX_LEVEL:
        return "present"
    return "absent" if min(levels) >= 2 else "mixed"


def judge_control(kind, set_levels, control_levels):
    state = judge_set(kind, control_levels)
    if state == "absent":
        return "clean"
    stronger = (all(c >= s for s, c in zip(set_levels, control_levels)) if kind == "cut"
                else all(c <= s for s, c in zip(set_levels, control_levels)))
    return "same" if state == "present" and stronger else "mixed"


def shows_effect(kind, levels):
    return any(x >= CUT_EFFECT_MIN_LEVEL for x in levels) if kind == "cut" else \
        any(x <= KEEP_CARRIED_MAX_LEVEL for x in levels)


def judge_reference_pixel(kind, value):
    if value is None:
        return None
    if kind == "cut":
        return "clean" if abs(value) <= REFERENCE_CUT_CLEAN_MAX_ABS else "mixed"
    return "clean" if value >= REFERENCE_KEEP_LOST_MIN else "mixed"


def decide(kind, word, pics, control, screens, override=None):
    """(verdict, basis, reason). Mutates ``control['judgement']``. The whole rule is ``RULE``."""
    if pics is None:
        if override:
            return "not_supported", "audit_statement", override
        return "unresolved", "none", "no audited picture call for this arm"
    basis = "picture_only" if word in PIXEL_UNUSABLE else "picture"
    if control is not None and control.get("levels"):
        control["judgement"] = judge_control(kind, pics["levels"], control["levels"])
    state = judge_set(kind, pics["levels"])
    if state == "mixed":
        return "unresolved", basis, "picture: seeds disagree or the effect is partial in one direction"
    if state == "absent":
        return "not_supported", basis, "picture: " + ("word stays on both seeds" if kind == "cut"
                                                       else "word lost on both seeds")
    if control is not None:
        judged = control.get("judgement")
        if judged == "clean":
            return "supported", basis, f"picture shows the effect; matched in-run control {control['arm']} does not"
        if judged == "same":
            return "not_supported", basis, f"matched in-run control {control['arm']} shows the same effect"
        return "unresolved", basis, (f"matched in-run control {control['arm']} partly shows the effect "
                                     "(symmetric rule: a Y- control is a weak effect)")
    shown = [s["arm"] for s in screens if s.get("levels") and shows_effect(kind, s["levels"])]
    if shown:
        return "unresolved", basis, ("no matched in-run control; in-run screen " + ", ".join(shown)
                                     + " shows the effect on at least one seed")
    return "control_missing", basis, ("picture shows the effect; no in-run control matched for this claim "
                                      "(any other arm is under reference_controls and does not count)")


def job(experiment, run):
    return json.loads((experiment / "runs" / run / "submission.json").read_text())["job_id"]


def word_profiles(tokens, capture):
    """head_sets3.py profiles: per word position, [block, head] mass summed over steps and rows."""
    tw = capture["tok_w"].astype(np.float64)
    out = []
    for position, wd in enumerate(tokens["words"]):
        if wd["rows"]:
            m = tw[..., wd["rows"]].sum((0, -1))
            out.append((position, wd["word"].strip(",.").lower(), wd["rows"], m,
                        tw[..., wd["rows"]].sum(-1)))
    return out


def convert(experiment, out):
    """Convert ``experiment`` into the NEW directory ``out`` (required; refuses an existing one)."""
    experiment, out = Path(experiment), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    root = declare_root(ROOT_NAME, experiment)
    full = experiment / "runs" / "full-a" / "result"
    plan_full = json.loads((full / "plan.json").read_text())
    tokens = json.loads((full / "tokens.json").read_text())
    hs3_path = experiment / "head_sets3.json"
    hs3 = json.loads(hs3_path.read_text())
    plans = {run: experiment / "runs" / run / "plan.json" for run in ("headfree", "headx", "headcut-cat", "headcut-car")}
    plan_json = {run: json.loads(p.read_text()) for run, p in plans.items()}
    headx_plan = plan_json["headx"]
    tables = {"headfree": {f"{r['prompt']}:{r['word']}": r for r in
                           json.loads((experiment / "runs/headfree/headfree_table.json").read_text())},
              "headx": {f"{r['prompt']}:{r['word']}": r for r in
                        json.loads((experiment / "runs/headx/headx_table.json").read_text())},
              # Addendum 4 table: one file (under headcut-cat) holds both prompts' rows.
              "headcut": {f"{r['prompt']}:{r['word']}": r for r in
                          json.loads((experiment / "runs/headcut-cat/headcut_table.json").read_text())}}
    plan_ptr = {run: file_pointer(p, root=root, role=f"{run} plan") for run, p in plans.items()}
    sha = {run: plan_ptr[run]["digest"]["sha256"] for run in plans}
    pointers = {
        "tokens": file_pointer(full / "tokens.json", root=root, role="word -> text rows"),
        "head_sets3": file_pointer(hs3_path, root=root, role="label-free and group head sets"),
        "headfree_table": file_pointer(experiment / "runs/headfree/headfree_table.json", root=root, role="gone scores"),
        "headx_table": file_pointer(experiment / "runs/headx/headx_table.json", root=root, role="gone scores"),
        "headcut_table": file_pointer(experiment / "runs/headcut-cat/headcut_table.json", root=root,
                                      role="Addendum 4 random-60 references (cat and car rows)"),
        "headfree_metrics": file_pointer(experiment / "runs/headfree/result/metrics.json", root=root, role="pixel MAD"),
        "audit": file_pointer(experiment / "AUDIT.md", root=root, role="picture judgements and verdict sources"),
        "findings": file_pointer(experiment / "FINDINGS.md", root=root, role="write-up"),
        **{f"{run}_plan": p for run, p in plan_ptr.items()},
    }
    if (experiment / "runs/headx/result/metrics.json").exists():
        pointers["headx_metrics"] = file_pointer(experiment / "runs/headx/result/metrics.json", root=root,
                                                 role="pixel MAD")
    jobs = {run: job(experiment, run) for run in ("full-a", "headfree", "headx", "headcut-cat", "headcut-car")}
    model = plan_full["model"]
    identity = model_identity(model, resolve_by=(
        "sha256 of the checkpoint files at the plan's model_path on the worker; no checkpoint hash was "
        "recorded in the plans or readable offline, so the model name stands in for the identity"))
    execution = {"model": model, "model_path": plan_full["model_path"], "height": plan_full["height"],
                 "width": plan_full["width"], "steps": plan_full["steps"]}

    plan_sets = {f"{q['id']}:{w}": s for q in plan_json["headfree"]["prompts"] if "head_sets" in q
                 for w, s in q["head_sets"].items()}
    checks = {"headfree_plan_sets_equal_head_sets3": all(
        plan_sets[k] == hs3["sets"][k] for k in hs3["sets"]) and set(plan_sets) == set(hs3["sets"])}
    context = {"experiment": experiment, "jobs": jobs, "sha": sha, "plan_ptr": plan_ptr, "pointers": pointers,
               "tables": tables, "hs3": hs3, "plan_json": plan_json}
    lifecycles, reads, sets = [], [], []
    verdicts = Counter()
    for pid, words in TEST.items():
        caps = {s: full / f"capture.{pid}.{s}.npz" for s in SEEDS}
        cap_sha = {s: file_sha256(p) for s, p in caps.items()}
        profiles = {}
        for s, path in caps.items():
            with np.load(path) as capture:
                profiles[s] = word_profiles(tokens[pid], capture)
        ids = tokens[pid]["input_ids"]
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        input_slot = {"kind": "token-ids", "length": len(ids),
                      "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        surface = PassiveSurface(
            model_identity=model, execution=execution,
            execution_point={"run": "full-a", "job": jobs["full-a"], "prompt": pid,
                             "selection_seed": SEEDS[0], "capture_sha256": cap_sha[SEEDS[0]]},
            surface={"state_schema": "flux-read-graph-capture-v1",
                     "kind": "saved capture; passive observation; no StateCut; not resumable",
                     "slots": [{"name": "text_rows", "role": "conditioning-text-row"},
                               {"name": "input_ids", "role": "prompt-token-ids"}]},
            cut_ref=f"passive-capture:sha256:{cap_sha[SEEDS[0]]}",
            session_id=f"flux-read-graph/full-a/{pid}",
            slots={"text_rows": {"kind": "tensor", "shape": [512], "dtype": "row"}, "input_ids": input_slot})
        from saturn_pub.values import digest as public_digest
        x_prompt = next(i for i, q in enumerate(headx_plan["prompts"]) if q["id"] == pid and "head_sets" in q)
        context.update(pid=pid, x_prompt=x_prompt)
        for word in words:
            key = f"{pid}:{word}"
            item = [i for i, p in enumerate(profiles[SEEDS[0]]) if p[1] == word]
            assert len(item) == 1, key
            item = item[0]
            _, _, rows, mass, per_step = profiles[SEEDS[0]][item]
            mass_b = profiles[SEEDS[1]][item][3]
            share = {SEEDS[0]: mass / mass.sum(), SEEDS[1]: mass_b / mass_b.sum()}
            rest = [i for i, p in enumerate(profiles[SEEDS[0]]) if p[1] != word]
            ranking = contrast_ranking(np.stack([p[3] for p in profiles[SEEDS[0]]]), item, rest_items=rest)
            rank_of = {int(f): r for r, f in enumerate(ranking["order"])}
            own = {n: sorted(tuple(int(x) for x in np.unravel_index(f, (BLOCKS, HEADS)))
                             for f in ranking["order"][:n]) for n in (60, 120, 240)}
            checks[f"{key}:own_sets_recomputed"] = all(
                own[n] == sorted(tuple(u) for u in hs3["sets"][key][f"own{n}"]) for n in (60, 120, 240))
            source = ReadSource(key, tuple(rows), "supplied", {
                "by": "experimenter: PREREG test word (head_sets3.py TEST)", "prompt": tokens[pid]["prompt"],
                "rows_from": pointers["tokens"]})
            binding = passive_binding(surface, f"flux.{pid}.{word}",
                tuple(surface.address("text_rows", (r,), "conditioning-text-row") for r in rows),
                context_slots=(("input_ids", public_digest(input_slot)),),
                evidence=("sha256:" + cap_sha[SEEDS[0]], "sha256:" + pointers["tokens"]["digest"]["sha256"]),
                claim="Supplied test word over its own text rows (tokens.json); readers observed by "
                      "image<-row write mass; observation is not a causal claim.", status="supplied")
            source = ReadSource(source.name, source.rows, source.assignment, source.provenance,
                                binding.to_dict()["fingerprint"])

            def held(members):
                return [round(float(sum(share[s][b, h] for b, h in members)), 4) for s in SEEDS]

            context.update(key=key, word=word, held=held)
            # Head sets for this word: headfree (head_sets3) and headx (plan), skipping empty sets.
            word_sets = []
            for name, members in hs3["sets"][key].items():
                word_sets.append(("headfree", name, [tuple(u) for u in members], file_pointer(
                    hs3_path, root=root, json_pointer=f"/sets/{key}/{name}",
                    sha256=pointers["head_sets3"]["digest"]["sha256"])))
            if word in headx_plan["prompts"][x_prompt]["head_sets"]:
                for name, members in headx_plan["prompts"][x_prompt]["head_sets"][word].items():
                    if members:
                        word_sets.append(("headx", name, [tuple(u) for u in members], file_pointer(
                            plans["headx"], root=root, json_pointer=f"/prompts/{x_prompt}/head_sets/{word}/{name}",
                            sha256=sha["headx"])))
            built = {}
            for run, name, members, pointer in word_sets:
                read_set = ReadSet(f"{run}:{name}", source, ReadSet.membership(members, pointer),
                                   selection_of(run, name),
                                   metadata={"run": run, "size": len(members),
                                             "joint_members": sum(b < JOINT for b, _ in members),
                                             "word_mass_held": held(members), "word_mass_seeds": list(SEEDS),
                                             "contrast_id": CONTRASTS[key]})
                built[(run, name)] = (read_set, set(members))
            # Reader uses for own60, in ranking order.
            uses, use_rows = [], []
            for b, h in sorted(own[60], key=lambda u: rank_of[u[0] * HEADS + u[1]]):
                label = f"joint.{b}" if b < JOINT else f"single.{b - JOINT}"
                use = passive_use(surface, binding, use_id=f"read-{pid}-{word}-{label}-h{h}",
                    consumer=f"flux2.{label}.attn.head{h}",
                    signed_observations=(
                        {"observable": "image_from_word_rows_write_mass", "seed": SEEDS[0],
                         "value": float(mass[b, h]), "n_steps": STEPS},
                        {"observable": "image_from_word_rows_write_mass", "seed": SEEDS[1],
                         "value": float(mass_b[b, h]), "n_steps": STEPS},
                        {"observable": "word_mass_share", "seed": SEEDS[0], "value": float(share[SEEDS[0]][b, h])}),
                    evidence_refs=tuple("sha256:" + cap_sha[s] for s in SEEDS),
                    metadata={"block": b, "head": h, "block_label": label,
                              "contrast_rank": rank_of[b * HEADS + h],
                              "contrast_score": float(ranking["scores"][b * HEADS + h]),
                              "per_step_mass_seed4242": [float(v) for v in per_step[:, b, h]],
                              "interpretation": "observed read; individual causal effect untested"})
                uses.append(use)
                use_rows.append((b, h, label))
            definition = passive_definition(surface, binding, occurrence_id=f"definition-{pid}-{word}",
                clock=None, evidence_refs=("sha256:" + pointers["tokens"]["digest"]["sha256"],),
                metadata={"rows": list(rows), "assignment": "supplied"}, status="supplied")
            lifecycle = passive_lifecycle(binding, definitions=(definition,), uses=uses,
                claim_boundary="Supplied word rows and observed image<-row write mass from a saved capture. "
                               "Set-level interventions live on ReadSets in reads.json; no head is shown "
                               "individually necessary.",
                metadata={"model": identity,
                          "captures": [file_pointer(caps[s], root=root, array="tok_w", sha256=cap_sha[s])
                                       for s in SEEDS]})
            lifecycles.append(lifecycle.to_dict())
            word_reads = []
            for use, (b, h, label) in zip(uses, use_rows, strict=True):
                member_of = [rs.set_id for rs, m in built.values() if (b, h) in m]
                word_reads.append(read_for_use(use, lifecycle, source=source,
                    reader=Reader(b, h, steps=tuple(range(STEPS)), block_label=label,
                                  receivers="image rows (mean over 256 image receivers)"),
                    mass=float(mass[b, h]),
                    mass_kind="image<-row write mass, summed over 4 steps and the word's rows (seed 4242)",
                    surface="passive-observation", backend="flux2-klein-4b", sets=member_of,
                    evidence=(file_pointer(caps[SEEDS[0]], root=root, array="tok_w",
                                           selector=[None, b, h, list(rows)], sha256=cap_sha[SEEDS[0]]),),
                    metadata={"share": float(share[SEEDS[0]][b, h]), "contrast_rank": rank_of[b * HEADS + h]},
                    model=identity))
            reads.extend(word_reads)
            for (run, name), (read_set, members) in built.items():
                read_set = ReadSet(read_set.name, read_set.source, read_set.members, read_set.selection,
                                   member_reads=tuple(r.fingerprint for r in word_reads
                                                      if (r.reader.block, r.reader.head) in members),
                                   metadata=read_set.metadata)
                for result in set_results(run, name, sorted(members), context):
                    read_set = read_set.with_result(result)
                    verdicts[(CONTRASTS[key], result.intervention, result.verdict)] += 1
                sets.append(read_set)
    (out / "lifecycles.json").write_text(json.dumps(lifecycles, indent=1, sort_keys=True) + "\n")
    contrasts = sorted(set(CONTRASTS.values()))
    scope = {
        "prompts": len(TEST), "seeds": len(SEEDS),
        "seeds_note": "seed-9001 mass is within .06 of 4242 for every set: a held-out image seed, not a held-out selection",
        "independent_contrasts": len(INDEPENDENT_CUT_CONTRASTS),
        "independent_contrast_ids": list(INDEPENDENT_CUT_CONTRASTS),
        "contrasts": {c: {"sources": sorted(k for k, v in CONTRASTS.items() if v == c),
                          "independent_cut_contrast": c in INDEPENDENT_CUT_CONTRASTS, "note": CONTRAST_NOTES[c]}
                      for c in contrasts},
        "source_to_contrast": CONTRASTS,
        "statement": ("2 prompts, 2 seeds, about 3 independent cut contrasts (colour words, ball, night); "
                      "set-level results only; every result names its contrast_id, so counts of results "
                      "are not counts of independent evidence"),
    }
    meta = {"roots": {root["name"]: root["path"]}, "inputs": pointers, "jobs": jobs, "plan_sha256": sha,
            "checks": checks, "model": identity, "scope": scope, "verdict_rule": VERDICT_RULE}
    saved = save_reads(out / "reads.json", reads, sets, meta=meta)
    by_status = Counter()
    by_contrast = {c: Counter() for c in contrasts}
    for (contrast, _, verdict), n in verdicts.items():
        by_status[verdict] += n
        by_contrast[contrast][verdict] += n
    summary = {"bindings": len(lifecycles), "uses": sum(len(l["uses"]) for l in lifecycles),
               "reads": len(reads), "read_sets": len(sets),
               "set_results": sum(verdicts.values()),
               "set_results_by_verdict": dict(sorted(by_status.items())),
               "set_results_by_contrast_verdict": {c: dict(sorted(v.items())) for c, v in by_contrast.items()},
               "scope": scope, "verdict_rule": {"id": VERDICT_RULE["id"], "sha256": VERDICT_RULE["sha256"]},
               "model": identity,
               "reads_with_causal_status_unknown": sum(r.causal.status == "unknown" for r in reads),
               "checks": checks, "reads_file": saved}
    by_kind = Counter()
    for (_, kind, verdict), n in verdicts.items():
        by_kind[f"{kind}:{verdict}"] += n
    summary["set_results_by_intervention_verdict"] = dict(sorted(by_kind.items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    from mdb.symbol_attention_view import write_view
    summary["view"] = write_view(out)
    return summary


def selection_of(run, name):
    if run == "headfree":
        if name.startswith("own"):
            return {"kind": "ranked", "ranking": "share*log(share/mean share of the prompt's other words)",
                    "label_free": True, "capture_seed": SEEDS[0], "top": int(name[3:]),
                    "tool": "attention_ranking.contrast_ranking (recomputed); head_sets3.py original"}
        if name.startswith("rand"):
            return {"kind": "random", "draw": "uniform; joint/single counts matched to own" + name[4:],
                    "rng_seed": 20261112, "mass_matched": False, "disjoint_from_own": False}
        return {"kind": "group", "definition": {"mod": "colour or attribute top 60", "noun": "object or scene top 60",
                "modI": "colour and attribute top 60", "nounI": "object and scene top 60"}[name],
                "ranking": "Addendum 5 category rankings, test prompt left out", "labels": "hand-assigned"}
    if name in HEADX_CATEGORIES:
        return {"kind": "category", "ranking": "category top 60, test prompt left out", "labels": "hand-assigned"}
    if name in HEADX_X:
        return {"kind": "category-unique", "definition": name[:-1] + " set minus every other category set"}
    return {"kind": "control", "definition": {
        "ownX": "own category set minus every other set",
        "heavy60": "60 heaviest readers outside the own set, joint/single counts matched (mass control)",
        "roleM": "top of own ranking with cumulative mass >= heavy60"}[name]}


def members_of(run, name, word, ctx):
    if run == "headfree":
        members = ctx["hs3"]["sets"].get(f"{ctx['pid']}:{word}", {}).get(name)
    else:
        members = ctx["plan_json"]["headx"]["prompts"][ctx["x_prompt"]]["head_sets"].get(word, {}).get(name)
    return None if not members else sorted(tuple(u) for u in members)


def set_results(run, name, members, ctx):
    key, word, held, pid = ctx["key"], ctx["word"], ctx["held"], ctx["pid"]
    tables, jobs = ctx["tables"], ctx["jobs"]
    hx = tables["headx"]
    if run == "headfree":
        arms = [("cut", "no-renormalisation", name, name)]
        if name in ("own60", "own120", "own240", "modI", "nounI"):
            arms.append(("keep", "no-renormalisation", "keep-" + name, name))
        if name == "own60":
            arms.append(("cut", "renormalising", "m:own60", name))
        table, grid, words = tables["headfree"][key], HEADFREE_PICTURES, W6
        per_seed = {arm: [s[arm] for s in table["seeds"]] for arm in table["mean"]}
        mean = table["mean"]
        sham = hx.get(key)
        floor = None if sham is None else max(abs(v) for v in sham["per_seed"]["sham-iso"])
        floor_source = (f"borrowed: headx {jobs['headx']} z:sham-iso, same word, max |gone| over seeds "
                        "(headfree ran no sham; AUDIT.md L5)" if sham else "none: no sham arm for this word in any run")
        floor_incomplete = True
        section = "AUDIT.md Follow-up: label-free head cut"
    else:
        arms = [("cut", "no-renormalisation", name, name)]
        if name in HEADX_CATEGORIES:
            arms.append(("keep", "no-renormalisation", "keep-" + name, name))
        table, grid, words = hx[key], HEADX_PICTURES, W5
        per_seed, mean = table["per_seed"], table["mean"]
        floor = max(abs(v) for v in per_seed["sham-iso"])
        floor_source = "in-run: z:sham-iso (FP32 path in all 25 blocks, nothing removed), max |gone| over seeds"
        floor_incomplete = False
        section = "AUDIT.md Follow-up: cross-category head cut"
    readable = word not in PIXEL_UNUSABLE
    run_id = jobs[run]
    set_mass = held(members)
    out = []
    for kind, cut_type, arm, set_name in arms:
        claim = CLAIMS[kind]
        pics = pictures(grid, words, arm if kind == "keep" or arm.startswith("m:") else set_name, word)
        control, screens, references = None, [], []
        if cut_type == "no-renormalisation":
            spec = COUNTED_CONTROLS.get((run, kind, set_name))
            if spec is not None:
                control = in_run_record(spec[0], kind, run, word, set_name, members, set_mass, grid, words,
                                        per_seed, mean, ctx, matched_on=spec[1], how=spec[2])
            for other, matched_on, how in SCREENS.get((run, kind, set_name), ()):
                record = in_run_record(other, kind, run, word, set_name, members, set_mass, grid, words,
                                       per_seed, mean, ctx, matched_on=matched_on, how=how)
                record.update(role="screen", verdict_effect="downgrade_only",
                              judgement=None if not record.get("levels") else
                              ("mixed" if shows_effect(kind, record["levels"]) else "clean"))
                screens.append(record)
            if (run, kind, set_name) in OTHER_JOB:
                references.append(other_job_record(OTHER_JOB[run, kind, set_name], kind, pid, word, set_mass, ctx))
        override = X1_OVERRIDE if (run == "headx" and set_name in HEADX_X and kind == "cut"
                                   and word != "snow") else None
        verdict, basis, reason = decide(kind, word, pics, control, screens, override)
        # Alternatives are ruled out only for an effect that is present.
        present = pics is not None and judge_set(kind, pics["levels"]) == "present"
        # Other-item controls: the same units cut on another informative word's rows (same run).
        if kind == "cut" and cut_type == "no-renormalisation" and present:
            references += other_item_records(run, set_name, members, ctx, grid, words)
        ruled = set()
        if present and control is not None and control.get("judgement") == "clean":
            rules = []
            if control["size"] >= SIZE_MATCH_MIN_RATIO * control["set_size"]:
                rules.append("set_size")
            if "mass" in control["matched_on"] or all(c >= s for c, s in zip(control["mass_held"], set_mass)):
                rules.append("mass")
            control["rules_out"] = [r for r in rules if r in ALTERNATIVES[claim]]
            ruled |= set(control["rules_out"])
        for r in references:
            ruled |= set(r.get("rules_out") or ())
        remaining = [a for a in sorted(ALTERNATIVES[claim]) if a not in ruled]
        claim_scope = None if verdict != "supported" else ("identity" if not remaining else CLAIM_SCOPES[claim])
        if control is not None and verdict == "control_missing":
            raise AssertionError("a counted control cannot give control_missing")
        references = screens + references
        if not readable:
            for record in (control, *references):
                if record is not None:
                    for k in ("effect", "effect_per_seed"):
                        record.pop(k, None)
                    if record.get("judged_by") == "score":
                        record["judgement"] = None
                        record["note"] = "pixel score unusable for this word; no picture call for this arm"
        caveat = {}
        if set_name in ("own120", "own240"):
            caveat = {"caveat": "AUDIT.md L1: heads added beyond own60 hold ~0 of the word's image-read "
                                "mass; the set result is not attributable to the word's own heads"}
        result = CausalResult(
            intervention=kind, cut_type=cut_type, verdict=verdict, plan_sha256=ctx["sha"][run], run_id=run_id,
            scope="set", basis=basis,
            effect=round(mean[arm], 4) if readable else None, effect_units="gone (pixel)" if readable else None,
            effect_per_seed=tuple(round(v, 4) for v in per_seed[arm]) if readable else None,
            floor=None if floor is None or not readable else round(floor, 4),
            floor_source=floor_source if readable else "omitted: pixel score unusable for this word",
            floor_incomplete=floor_incomplete if readable else True,
            control=control, reference_controls=tuple(references), picture=pics,
            claim_text=CUT_CLAIM if kind == "cut" else KEEP_CLAIM, claim_scope=claim_scope,
            ruled_out=tuple(sorted(ruled)), contrast_id=CONTRASTS[key],
            rule_id=VERDICT_RULE["id"], rule_sha256=VERDICT_RULE["sha256"], plan=ctx["plan_ptr"][run],
            reason=reason if readable else reason + "; " + PIXEL_UNUSABLE[word],
            evidence=(ctx["pointers"][f"{run}_table"], ctx["pointers"]["audit"]),
            metadata={"arm": "z:" + arm if not arm.startswith("m:") else arm, "seeds": list(SEEDS),
                      "set_size": len(members), "set_mass_held": set_mass, "score": SCORE,
                      "audit_section": section, **caveat})
        result.verify_plan(ctx["experiment"] / "runs" / run / "plan.json")
        out.append(result)
    return out


def in_run_record(arm_set, kind, run, word, set_name, members, set_mass, grid, words, per_seed, mean, ctx, *,
                  matched_on, how):
    """An in-run arm on this word (control or screen), judged by picture."""
    arm = arm_set if kind == "cut" or arm_set.startswith("keep-") else "keep-" + arm_set
    core = arm.removeprefix("keep-")
    other = members_of(run, core, word, ctx) or []
    pics = pictures(grid, words, arm, word)
    return {"arm": "z:" + arm, "in_run": True, "run_id": ctx["jobs"][run], "matched_on": list(matched_on),
            "how_matched": how, "judged_by": "picture", "size": len(other), "set_size": len(members),
            "mass_held": ctx["held"](other) if other else None, "set_mass_held": set_mass,
            "mass_matched": "mass" in matched_on, "mass_relation": mass_relation(ctx["held"](other), set_mass)
            if other else "unknown",
            "effect": round(mean[arm], 4), "effect_per_seed": [round(v, 4) for v in per_seed[arm]],
            "picture": None if pics is None else pics["calls"], "levels": None if pics is None else pics["levels"]}


def mass_relation(control, target):
    if all(c >= t for c, t in zip(control, target)):
        return "more"
    if all(c < t for c, t in zip(control, target)):
        return "less"
    return "mixed"


def other_job_record(arm, kind, pid, word, set_mass, ctx):
    source_run = "headcut-cat" if pid == "cat" else "headcut-car"
    row = ctx["tables"]["headcut"].get(f"{pid}:{word}")
    q = next(q for q in ctx["plan_json"][source_run]["prompts"] if q.get("id") == pid and "head_sets" in q)
    core = arm.removeprefix("keep-")
    members = sorted(tuple(u) for u in q["head_sets"].get(word, {}).get(core, []))
    value = None if row is None else round(row[arm], 4)
    return {"arm": "z:" + arm, "role": "other_job_reference", "in_run": False, "run_id": ctx["jobs"][source_run],
            "plan_sha256": ctx["sha"][source_run], "matched_on": ["joint_single_counts_of_role60"],
            "how_matched": "Addendum 4 random 60-head draw matched to role60's joint/single counts (another job, "
                           "another worker sha); AUDIT.md Follow-up: head-set cut",
            "judged_by": "score", "effect": value, "effect_kind": "mean of 2 seeds, pixel gone (no picture call)",
            "judgement": judge_reference_pixel(kind, value), "verdict_effect": "none",
            "size": len(members), "mass_held": ctx["held"](members) if members else None, "set_mass_held": set_mass}


def other_item_records(run, set_name, members, ctx, grid, words):
    """Same units cut on another informative word's rows, same run: rules out generic damage if clean."""
    out = []
    for other in TEST[ctx["pid"]]:
        if other == ctx["word"] or other in NO_OTHER_ITEM_WITNESS:
            continue
        if members_of(run, set_name, other, ctx) != members:
            continue
        pics = pictures(grid, words, set_name, other)
        if pics is None:
            continue
        clean = judge_set("cut", pics["levels"]) == "absent"
        out.append({"arm": f"z:{set_name}@rows-{other}", "role": "other_item", "in_run": True,
                    "run_id": ctx["jobs"][run], "matched_on": ["same_units"],
                    "how_matched": f"the same {len(members)} heads cut on {other}'s rows in the same run",
                    "judged_by": "picture", "picture": pics["calls"], "levels": pics["levels"],
                    "judgement": "clean" if clean else ("same" if judge_set("cut", pics["levels"]) == "present"
                                                        else "mixed"),
                    "verdict_effect": "none", "rules_out": ["generic_damage"] if clean else []})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--out", type=Path, required=True, help="NEW output directory (must not exist)")
    args = parser.parse_args(argv)
    summary = convert(args.experiment, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("reads_file", "scope")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
