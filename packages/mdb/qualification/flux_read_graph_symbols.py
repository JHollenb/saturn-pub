"""Convert the saved FLUX.2 Klein read-graph experiment into mdb symbol-read records. No model run.

    python flux_read_graph_symbols.py [--experiment DIR] [--out NEW_DIR]

Inputs (read only): ``runs/full-a/result/{tokens.json, capture.<prompt>.<seed>.npz}`` (``tok_w``:
[step, block, head, 512 text rows] image<-row write mass, mean over image receivers),
``head_sets3.json``, ``runs/headfree/{plan.json, headfree_table.json}``,
``runs/headx/{plan.json, headx_table.json}``, ``runs/headcut-cat/headcut_table.json`` (cross-run
random-60 controls), ``runs/*/submission.json`` (job ids), ``AUDIT.md`` (picture judgements).

Output (a NEW directory): ``lifecycles.json`` (one canonical SymbolLifecycle per test word: a
supplied binding over the word's text rows and one observed use per own60 head),
``reads.json`` (``SymbolRead`` per use, causal status ``unknown``; ``ReadSet`` per (word, head
set) with set-level cut/keep results), ``summary.json`` and an offline view. Captures and tables
are referenced by path + sha256, never copied.

Verdict rule (mechanical, from the audited picture grids; the picture decides, AUDIT.md):
picture call per seed -> level Y=0, Y-=1 (weakened), P=2 (partial), N=3 (gone).
cut: effect present if both seeds >= 1, absent if both 0. keep: carried if both <= 1, lost if
both >= 2. Mixed -> unresolved. Absent -> not_supported. Present -> compare the matched control:
none -> control_missing; control clean (cut: control <= 1 on both seeds and below the set on
each seed; keep: control loses the word on both seeds; cross-run pixel control: |gone| <= .15
for cut, gone >= .7 for keep) -> supported; control equal or stronger -> not_supported; else
unresolved. snow and rain are always unresolved: their pixel score is unusable (AUDIT L3, X2)
and no numbers are recorded for them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mdb.attention_ranking import contrast_ranking
from mdb.symbol_reads import (
    CausalResult,
    PassiveSurface,
    Reader,
    ReadSet,
    ReadSource,
    file_pointer,
    file_sha256,
    passive_binding,
    passive_definition,
    passive_lifecycle,
    passive_use,
    read_for_use,
    save_reads,
)

DEFAULT_EXPERIMENT = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
DEFAULT_OUT = Path("/private/tmp/claude-501/-Users-jakeholl-domains/"
                   "445364a4-10f1-4c06-89fe-7e5206baa3c3/scratchpad/flux-symbols")
TEST = {"cat": ["red", "black", "ball", "snow"], "car": ["night", "rain"]}
UNREADABLE = {"snow": "AUDIT.md L3/X2: snow pixel score follows sky/background, ground snow never removed",
              "rain": "AUDIT.md L3: rain pixel score ranges -.36 to 1.50 while streaks never change"}
SEEDS = (4242, 9001)
JOINT, BLOCKS, HEADS, STEPS = 5, 25, 24, 4
SCORE = ("gone = (d(arm) - d(iso)) / (d(none) - d(iso)), d = 1 - MAD(arm, deleted) / "
         "MAD(native, deleted); 0 = word intact, 1 = word as gone as with every read cut")
CUT_CLAIM = ("cutting this set's reads of the word's own rows (receivers: image rows and other "
             "text rows) removes or weakens the word")
KEEP_CLAIM = "with only this set reading the word's own rows, the word is carried"

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
HEADX_X = ("colorX", "objectX", "sceneX", "lightX", "attributeX")
# AUDIT.md X1: "Every <cat>X arm is pixel <= .13, apart from snow"; no picture call recorded.
X1_OVERRIDE = "AUDIT.md X1: with the shared heads removed nothing is left (every <cat>X arm pixel <= .13)"

HEADFREE_CONTROLS = {("cut", "own120"): "rand120", ("cut", "own240"): "rand240",
                     ("cut", "mod"): "noun", ("cut", "noun"): "mod",
                     ("cut", "modI"): "nounI", ("cut", "nounI"): "modI",
                     ("keep", "modI"): "nounI", ("keep", "nounI"): "modI"}
HEADFREE_CROSS = {("cut", "own60"): "rand60a", ("keep", "own60"): "keep-rand60a"}
HEADX_CATEGORIES = ("color", "object", "scene", "light", "attribute")


def level(call):
    token = call.split()[0]
    if token.startswith("Y-"):
        return 1
    return {"Y": 0, "P": 2, "N": 3}[token[0]]


def pictures(grid, words, arm, word):
    row = grid.get(arm)
    if row is None or word not in words:
        return None
    calls = tuple(row[words.index(word)].split("|"))
    return {"calls": list(calls), "levels": [level(c) for c in calls]}


def judge_set(kind, levels):
    if kind == "cut":
        return "present" if min(levels) >= 1 else "absent" if max(levels) == 0 else "mixed"
    return "present" if max(levels) <= 1 else "absent" if min(levels) >= 2 else "mixed"


def judge_control(kind, set_levels, control):
    if control is None:
        return None
    if control.get("levels") is not None:
        c, a = control["levels"], set_levels
        if kind == "cut":
            if max(c) <= 1 and all(x > y for x, y in zip(a, c)):
                return "clean"
            return "same" if all(y >= x for x, y in zip(a, c)) else "mixed"
        if min(c) >= 2:
            return "clean"
        return "same" if max(c) <= 1 else "mixed"
    value = control.get("effect")
    if value is None:
        return "mixed"
    if kind == "cut":
        return "clean" if abs(value) <= .15 else "mixed"
    return "clean" if value >= .7 else "mixed"


def decide(kind, word, pics, control, override=None):
    if word in UNREADABLE:
        return "unresolved", UNREADABLE[word] + "; picture calls kept as text, no numbers"
    if pics is None:
        if override:
            return override
        return "unresolved", "no audited picture call for this arm"
    state = judge_set(kind, pics["levels"])
    if state == "mixed":
        return "unresolved", "picture: seeds disagree or the effect is partial in one direction"
    if state == "absent":
        return "not_supported", "picture: " + ("word stays on both seeds" if kind == "cut"
                                               else "word lost on both seeds")
    judged = judge_control(kind, pics["levels"], control)
    if judged is None:
        return "control_missing", "picture shows the effect; no matched control arm was run"
    control["judgement"] = judged
    if judged == "clean":
        return "supported", "picture shows the effect; matched control does not"
    if judged == "same":
        return "not_supported", "matched control shows the same effect"
    return "unresolved", "matched control partly shows the effect"


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


def convert(experiment=DEFAULT_EXPERIMENT, out=DEFAULT_OUT):
    experiment, out = Path(experiment), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    full = experiment / "runs" / "full-a" / "result"
    plan_full = json.loads((full / "plan.json").read_text())
    tokens = json.loads((full / "tokens.json").read_text())
    hs3_path = experiment / "head_sets3.json"
    hs3 = json.loads(hs3_path.read_text())
    headfree_plan_path = experiment / "runs" / "headfree" / "plan.json"
    headx_plan_path = experiment / "runs" / "headx" / "plan.json"
    headfree_plan = json.loads(headfree_plan_path.read_text())
    headx_plan = json.loads(headx_plan_path.read_text())
    headfree_table = {f"{r['prompt']}:{r['word']}": r for r in
                      json.loads((experiment / "runs/headfree/headfree_table.json").read_text())}
    headx_table = {f"{r['prompt']}:{r['word']}": r for r in
                   json.loads((experiment / "runs/headx/headx_table.json").read_text())}
    headcut_table = {f"{r['prompt']}:{r['word']}": r for r in
                     json.loads((experiment / "runs/headcut-cat/headcut_table.json").read_text())}
    sha = {name: file_sha256(path) for name, path in {
        "headfree_plan": headfree_plan_path, "headx_plan": headx_plan_path,
        "headcut_cat_plan": experiment / "runs/headcut-cat/plan.json",
        "headcut_car_plan": experiment / "runs/headcut-car/plan.json"}.items()}
    pointers = {
        "tokens": file_pointer(full / "tokens.json", role="word -> text rows"),
        "head_sets3": file_pointer(hs3_path, role="label-free and group head sets"),
        "headfree_table": file_pointer(experiment / "runs/headfree/headfree_table.json", role="gone scores"),
        "headx_table": file_pointer(experiment / "runs/headx/headx_table.json", role="gone scores"),
        "headcut_table": file_pointer(experiment / "runs/headcut-cat/headcut_table.json",
                                      role="Addendum 4 random-60 controls (cat and car)"),
        "headfree_metrics": file_pointer(experiment / "runs/headfree/result/metrics.json", role="pixel MAD"),
        "audit": file_pointer(experiment / "AUDIT.md", role="picture judgements and verdict sources"),
        "findings": file_pointer(experiment / "FINDINGS.md", role="write-up"),
    }
    if (experiment / "runs/headx/result/metrics.json").exists():
        pointers["headx_metrics"] = file_pointer(experiment / "runs/headx/result/metrics.json", role="pixel MAD")
    jobs = {run: job(experiment, run) for run in ("full-a", "headfree", "headx", "headcut-cat", "headcut-car")}
    model = plan_full["model"]
    execution = {"model": model, "model_path": plan_full["model_path"], "height": plan_full["height"],
                 "width": plan_full["width"], "steps": plan_full["steps"]}

    plan_sets = {f"{q['id']}:{w}": s for q in headfree_plan["prompts"] if "head_sets" in q
                 for w, s in q["head_sets"].items()}
    checks = {"headfree_plan_sets_equal_head_sets3": all(
        plan_sets[k] == hs3["sets"][k] for k in hs3["sets"]) and set(plan_sets) == set(hs3["sets"])}
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
                      "sha256": __import__("hashlib").sha256(json.dumps(ids).encode()).hexdigest()}
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
        for word in words:
            key = f"{pid}:{word}"
            item = [i for i, p in enumerate(profiles[SEEDS[0]]) if p[1] == word]
            assert len(item) == 1, key
            item = item[0]
            _, _, rows, mass, per_step = profiles[SEEDS[0]][item]
            mass_b = profiles[SEEDS[1]][item][3]
            share = mass / mass.sum()
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
                evidence=("sha256:" + cap_sha[SEEDS[0]], "sha256:" + pointers["tokens"]["sha256"]),
                claim="Supplied test word over its own text rows (tokens.json); readers observed by "
                      "image<-row write mass; observation is not a causal claim.", status="supplied")
            source = ReadSource(source.name, source.rows, source.assignment, source.provenance,
                                binding.to_dict()["fingerprint"])
            # Head sets for this word: headfree (head_sets3) and headx (plan), skipping empty sets.
            word_sets = []
            for name, members in hs3["sets"][key].items():
                word_sets.append(("headfree", name, [tuple(u) for u in members], file_pointer(
                    hs3_path, json_pointer=f"/sets/{key}/{name}", sha256=pointers["head_sets3"]["sha256"])))
            if word in headx_plan["prompts"][x_prompt]["head_sets"]:
                for name, members in headx_plan["prompts"][x_prompt]["head_sets"][word].items():
                    if members:
                        word_sets.append(("headx", name, [tuple(u) for u in members], file_pointer(
                            headx_plan_path, json_pointer=f"/prompts/{x_prompt}/head_sets/{word}/{name}",
                            sha256=sha["headx_plan"])))
            built = {}
            for run, name, members, pointer in word_sets:
                held = float(sum(share[b, h] for b, h in members))
                selection = selection_of(run, name)
                read_set = ReadSet(f"{run}:{name}", source, ReadSet.membership(members, pointer), selection,
                                   metadata={"run": run, "size": len(members),
                                             "joint_members": sum(b < JOINT for b, _ in members),
                                             "held_word_mass_share_seed4242": round(held, 6)})
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
                        {"observable": "word_mass_share", "seed": SEEDS[0], "value": float(share[b, h])}),
                    evidence_refs=tuple("sha256:" + cap_sha[s] for s in SEEDS),
                    metadata={"block": b, "head": h, "block_label": label,
                              "contrast_rank": rank_of[b * HEADS + h],
                              "contrast_score": float(ranking["scores"][b * HEADS + h]),
                              "per_step_mass_seed4242": [float(v) for v in per_step[:, b, h]],
                              "interpretation": "observed read; individual causal effect untested"})
                uses.append(use)
                use_rows.append((b, h, label))
            definition = passive_definition(surface, binding, occurrence_id=f"definition-{pid}-{word}",
                clock=None, evidence_refs=("sha256:" + pointers["tokens"]["sha256"],),
                metadata={"rows": list(rows), "assignment": "supplied"}, status="supplied")
            lifecycle = passive_lifecycle(binding, definitions=(definition,), uses=uses,
                claim_boundary="Supplied word rows and observed image<-row write mass from a saved capture. "
                               "Set-level interventions live on ReadSets in reads.json; no head is shown "
                               "individually necessary.",
                metadata={"captures": [file_pointer(caps[s], array="tok_w", sha256=cap_sha[s]) for s in SEEDS]})
            lifecycles.append(lifecycle.to_dict())
            word_reads = []
            for use, (b, h, label) in zip(uses, use_rows, strict=True):
                member_of = [rs.set_id for rs, m in built.values() if (b, h) in m]
                word_reads.append(read_for_use(use, lifecycle, source=source,
                    reader=Reader(b, h, steps=tuple(range(STEPS)), block_label=label,
                                  receivers="image rows (mean over 256 image receivers)"),
                    mass=float(mass[b, h]), mass_kind="image<-row write mass, summed over 4 steps and the word's rows (seed 4242)",
                    surface="passive-observation", backend="flux2-klein-4b", sets=member_of,
                    evidence=(file_pointer(caps[SEEDS[0]], array="tok_w", selector=[None, b, h, list(rows)],
                                           sha256=cap_sha[SEEDS[0]]),),
                    metadata={"share": float(share[b, h]), "contrast_rank": rank_of[b * HEADS + h]}))
            reads.extend(word_reads)
            for (run, name), (read_set, members) in built.items():
                read_set = ReadSet(read_set.name, read_set.source, read_set.members, read_set.selection,
                                   member_reads=tuple(r.fingerprint for r in word_reads
                                                      if (r.reader.block, r.reader.head) in members),
                                   metadata=read_set.metadata)
                for result in set_results(run, name, word, key, pid, experiment, jobs, sha, pointers,
                                          headfree_table, headx_table, headcut_table):
                    read_set = read_set.with_result(result)
                    verdicts[(result.intervention, result.verdict)] += 1
                sets.append(read_set)
    (out / "lifecycles.json").write_text(json.dumps(lifecycles, indent=1, sort_keys=True) + "\n")
    meta = {"experiment": str(experiment), "inputs": pointers,
            "jobs": jobs, "plan_sha256": sha, "checks": checks,
            "scope": "2 prompts (cat, car), 2 seeds, about 3 independent cut contrasts (colour words, "
                     "ball, night); set-level results only; snow and rain unresolved",
            "verdict_rule": __doc__.split("Verdict rule")[1].strip()[:3000]}
    saved = save_reads(out / "reads.json", reads, sets, meta=meta)
    summary = {"bindings": len(lifecycles), "uses": sum(len(l["uses"]) for l in lifecycles),
               "reads": len(reads), "read_sets": len(sets),
               "set_results": sum(verdicts.values()),
               "set_results_by_verdict": dict(Counter(v for (_, v) in verdicts.elements())),
               "set_results_by_intervention_verdict": {f"{k}:{v}": n for (k, v), n in sorted(verdicts.items())},
               "reads_with_causal_status_unknown": sum(r.causal.status == "unknown" for r in reads),
               "checks": checks, "reads_file": saved}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    from mdb.symbol_attention_view import write_view
    summary["view"] = write_view(out)
    return summary


def selection_of(run, name):
    if run == "headfree":
        if name.startswith("own"):
            return {"kind": "nominated", "ranking": "share*log(share/mean share of the prompt's other words)",
                    "label_free": True, "capture_seed": SEEDS[0], "top": int(name[3:]),
                    "tool": "attention_ranking.contrast_ranking (recomputed); head_sets3.py original"}
        if name.startswith("rand"):
            return {"kind": "control", "draw": "uniform; joint/single counts matched to own" + name[4:],
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


def set_results(run, name, word, key, pid, experiment, jobs, sha, pointers, hf, hx, hc):
    if run == "headfree":
        arms = [("cut", "no-renormalisation", name, name)]
        if name in ("own60", "own120", "own240", "modI", "nounI"):
            arms.append(("keep", "no-renormalisation", "keep-" + name, name))
        if name == "own60":
            arms.append(("cut", "renormalising", "m:own60", name))
        table, grid, words = hf[key], HEADFREE_PICTURES, W6
        per_seed = {arm: [s[arm] for s in table["seeds"]] for arm in table["mean"]}
        mean = table["mean"]
        sham = hx.get(key)
        floor = None if sham is None else max(abs(v) for v in sham["per_seed"]["sham-iso"])
        floor_source = ("cross-run: headx z:sham-iso, same word (headfree ran no sham; AUDIT.md L5)"
                        if sham else "none: no sham arm for this word in any run")
        section = "AUDIT.md Follow-up: label-free head cut"
    else:
        arms = [("cut", "no-renormalisation", name, name)]
        if name in HEADX_CATEGORIES:
            arms.append(("keep", "no-renormalisation", "keep-" + name, name))
        table, grid, words = hx[key], HEADX_PICTURES, W5
        per_seed, mean = table["per_seed"], table["mean"]
        floor = max(abs(v) for v in per_seed["sham-iso"])
        floor_source = "in-run: z:sham-iso (FP32 path, nothing removed)"
        section = "AUDIT.md Follow-up: cross-category head cut"
    run_id, plan_sha = jobs[run], sha[f"{run}_plan"]
    out = []
    for kind, cut_type, arm, set_name in arms:
        pics = pictures(grid, words, arm if kind == "keep" or arm.startswith("m:") else set_name, word)
        control = None if cut_type == "renormalising" else control_for(
            run, kind, set_name, word, key, pid, grid, words, per_seed, mean, hc, jobs, sha)
        override = None
        if run == "headx" and set_name in HEADX_X and kind == "cut" and word != "snow":
            override = ("not_supported", X1_OVERRIDE)
        verdict, reason = decide(kind, word, pics, control, override)
        readable = word not in UNREADABLE
        if control is not None and not readable:
            control = {k: v for k, v in control.items() if k not in ("effect", "effect_per_seed")}
        out.append(CausalResult(
            intervention=kind, cut_type=cut_type, verdict=verdict, plan_sha256=plan_sha, run_id=run_id,
            effect=round(mean[arm], 4) if readable else None, effect_units="gone (pixel)" if readable else None,
            effect_per_seed=tuple(round(v, 4) for v in per_seed[arm]) if readable else None,
            floor=None if floor is None or not readable else round(floor, 4),
            floor_source=floor_source if readable else "omitted: pixel score unusable for this word",
            control=control if verdict != "control_missing" else None, reason=reason,
            claim=CUT_CLAIM if kind == "cut" else KEEP_CLAIM,
            evidence=(pointers[f"{run}_table"], pointers["audit"]),
            metadata={"arm": "z:" + arm if not arm.startswith("m:") else arm, "seeds": list(SEEDS),
                      "picture": pics, "score": SCORE, "audit_section": section,
                      **({"caveat": "AUDIT.md L1: heads added beyond own60 hold ~0 of the word's image-read "
                                    "mass; the set result is not attributable to the word's own heads"}
                         if set_name in ("own120", "own240") else {})}))
    return out


def control_for(run, kind, set_name, word, key, pid, grid, words, per_seed, mean, hc, jobs, sha):
    if run == "headfree" and (kind, set_name) in HEADFREE_CONTROLS:
        other = HEADFREE_CONTROLS[kind, set_name]
        arm = other if kind == "cut" else "keep-" + other
    elif run == "headx" and kind == "cut" and set_name in HEADX_CATEGORIES:
        arm = "heavy60"
    elif (run == "headfree" and (kind, set_name) in HEADFREE_CROSS) or (
            run == "headx" and kind == "keep" and set_name in HEADX_CATEGORIES):
        cross = HEADFREE_CROSS.get((kind, set_name), "keep-rand60a")
        row = hc.get(key)
        if row is None:
            return None
        source_run = "headcut-cat" if pid == "cat" else "headcut-car"
        return {"arm": "z:" + cross, "in_run": False, "run_id": jobs[source_run],
                "plan_sha256": sha[source_run.replace("-", "_") + "_plan"],
                "effect": round(row[cross], 4), "effect_kind": "mean of 2 seeds, pixel gone (no picture call)",
                "levels": None, "size": 60, "source": "Addendum 4 random 60-head draw (AUDIT.md Follow-up: head-set cut)"}
    else:
        return None
    pics = pictures(grid, words, arm, word)
    return {"arm": "z:" + arm, "in_run": True, "run_id": jobs[run], "effect": round(mean[arm], 4),
            "effect_per_seed": [round(v, 4) for v in per_seed[arm]],
            "picture": None if pics is None else pics["calls"], "levels": None if pics is None else pics["levels"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    summary = convert(args.experiment, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "reads_file"}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
