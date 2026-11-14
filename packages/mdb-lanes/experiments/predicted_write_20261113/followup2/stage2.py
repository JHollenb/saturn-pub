"""Stage the PREREG-ADDENDUM-2 jobs (no model is loaded here).

  PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
  export PYTHONPATH=<worktree>/packages/mdb-lanes/src:<worktree>/packages/mdb/src
  $PY stage2.py plans --sets plans2/sets2.json --out plans2         # sealed plan "gm" + word-aligned leases
  $PY stage2.py panel RUN --lease 0|1                              # unchanged flux_read_panel_worker.py, retain latents
  $PY stage2.py vswap RUN                                          # vswap2_worker.py (+ pw_worker.py unchanged)
  $PY ../stage.py submit RUN [--attempt N]                         # the original submit (declared pin, see ADDENDUM-2)

Reuses ``../stage.py`` (``_stage``, ``items_map``) and ``../plans.py`` (``word_panel``) unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
import plans as PL  # noqa: E402
import stage as ST  # noqa: E402

FS = ST.FS
GM_CUT = ("gmass60", "massmatch-lowpw-single", "massmatch-lowpw-single-own", "gmassmatch-lowtextpw",
          "rand60c", "rand60d", "pw60", "own60")
LEASE_WORDS = (("red", "black", "ball"), ("snow", "night", "rain"))
UNRELATED = {
    # same number of real tokens as the scene prompt (checked with the Qwen3 tokenizer: 32 and 27;
    # the worker records n_real for both and flags any difference)
    "cat": "a photorealistic wooden table holding a green vase of tulips in a bright kitchen, morning light",
    "car": "a tall glass of orange juice standing on a white kitchen counter in the morning",
}
SCENES = [
    {"word": "snow", "prompt_id": "cat", "unrelated": UNRELATED["cat"], "nonscene": "red"},
    {"word": "rain", "prompt_id": "car", "unrelated": UNRELATED["car"], "nonscene": "blue"},
]


def build_plan(sets_path, out):
    from mdb_lanes.attention_panels import Panel
    from mdb_lanes.flux_read_panels import merge_panels, work_items
    out = Path(out)
    raw = json.loads(Path(sets_path).read_text())
    sets = {w: {k: PL.flat(v) for k, v in s.items()} for w, s in raw.items()}
    note = "experiment predicted_write_20261113, PREREG-ADDENDUM-2; sets from followup2/select2.py; seed 4242 only"

    def drop_native(panel):
        d = json.loads(panel.canonical())
        d["arms"] = [a for a in d["arms"] if a["arm"] != "native"]
        return Panel.from_json(d)

    panel = merge_panels([drop_native(PL.word_panel(w, sets[w], GM_CUT, ())) for w in ST.WORDS], notes=[note])
    seal = panel.seal()
    items = ST.items_map()
    ids = panel.by_id()
    per_word = {w: sum(1 for a in panel.arms if a.item == w) for w in ST.WORDS}
    cap = max(sum(per_word[w] for w in g) for g in LEASE_WORDS)
    leases = panel.leases(cap, seal=seal)
    got = [sorted({ids[a].item for _, a in l["branches"]}) for l in leases]
    if got != [sorted(g) for g in LEASE_WORDS] or any(l["replicated"] for l in leases):
        raise ValueError(f"leases are not word-aligned without replicas: {got}")
    (out / "gm.plan.json").write_text(panel.canonical() + "\n")
    ST.save(out / "gm.seal.json", seal)
    ST.save(out / "gm.leases.json", {"plan_sha256": panel.sha256(), "cap": cap, "leases": leases})
    summary = {"plan_sha256": panel.sha256(), "seal_sha256": seal["seal_sha256"], "arms": len(panel.arms),
               "branches": panel.branch_count, "per_word": per_word, "lease_cap": cap,
               "leases": [(len(l["branches"]), work_items(panel, l, items)) for l in leases]}
    ST.save(out / "gm-summary.json", summary)
    return summary


def stage_panel(run, plans, lease_index, vram_mb, ram_mb, est_wall_s, timeout_s):
    from mdb_lanes.attention_panels import Panel
    from mdb_lanes.flux_read_panels import RUN_SCHEMA, check_lease, lease_roots
    plans = Path(plans)
    panel = Panel.from_json(json.loads((plans / "gm.plan.json").read_text()))
    seal = json.loads((plans / "gm.seal.json").read_text())
    leases = json.loads((plans / "gm.leases.json").read_text())
    if panel.sha256() != seal["plan_sha256"] or leases["plan_sha256"] != seal["plan_sha256"]:
        raise ValueError("plan, seal and leases disagree")
    lease = check_lease(panel, leases["leases"][lease_index])
    items = ST.items_map()
    work = len(lease["branches"]) + len(lease_roots(panel, lease, items))
    tag = f"gm-{lease_index:02d}"
    passes = [dict(tag=tag, plan_file="plans/gm.plan.json", plan_sha256=seal["plan_sha256"],
                   seal_sha256=seal["seal_sha256"], lease=lease, precision="native", zero_method="delta",
                   reference_dir=None)]
    config = dict(schema=RUN_SCHEMA, job=tag, model="flux2-klein-4b-distilled", model_path=FS.MODEL_PATH,
                  output_scratch=FS.SCRATCH, height=256, width=256, steps=4, prompts=FS.PROMPTS, items=items,
                  contexts={str(ST.SEED): {"seed": ST.SEED}}, retain=["latents"], max_wall_seconds=600, mode="panel",
                  passes=passes, hash_checkpoint=False, experiment="predicted_write_20261113/addendum-2")
    extras = [(plans / "gm.plan.json", "plans/gm.plan.json")]
    return ST._stage(run, ST.QUAL / "flux_read_panel_worker.py", config, extras,
                     ("attention_port", "attention_panels", "flux_read_panels"), work, vram_mb, ram_mb, est_wall_s,
                     timeout_s, schema="mdb-flux-read-panel-request-v1", backend="diffusers", dtype="bfloat16",
                     policy="one resident Klein 4B lease; prompts encoded first, encoder released; reference roots "
                            "then sealed-lease branches in process; latents retained; no retries",
                     reason="Klein 4B BF16 B1 256px 4-step resident lane; read-panel jobs measured 7991 MB peak "
                            "allocated, load transient 8022-8367 MB tree VRAM; x1.3",
                     model="flux2-klein-4b-distilled", allow_tokens=[FS.MODEL_PATH, FS.SCRATCH, "flux-read-panel"])


def stage_vswap(run, vram_mb, ram_mb, est_wall_s, timeout_s):
    config = dict(schema="mdb-predicted-write-vswap2-v1", model="flux2-klein-4b-distilled", model_path=FS.MODEL_PATH,
                  output_scratch=FS.SCRATCH, height=256, width=256, steps=4, seed=ST.SEED, prompts=FS.PROMPTS,
                  scenes=SCENES, max_wall_seconds=540)
    work = sum(3 + 2 * (2 + (1 if s.get("nonscene") else 0)) + 1 for s in SCENES)
    extras = [(ST.QUAL / "flux_read_panel_worker.py", "flux_read_panel_worker.py"),
              (EXP / "pw_worker.py", "pw_worker.py")]
    return ST._stage(run, HERE / "vswap2_worker.py", config, extras, ST.PW_MODULES, work, vram_mb, ram_mb,
                     est_wall_s, timeout_s, schema="mdb-predicted-write-vswap2-request-v1", backend="diffusers",
                     dtype="bfloat16",
                     policy="one resident Klein 4B lease; prompts encoded first, encoder released; roots, source "
                            "runs and value-swap arms in process; no retries",
                     reason="Klein 4B BF16 B1 256px 4-step; capture job measured 8385 MB peak; x1.3",
                     model="flux2-klein-4b-distilled",
                     allow_tokens=[FS.MODEL_PATH, FS.SCRATCH, "predicted-write-vswap2"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plans")
    p.add_argument("--sets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("panel")
    p.add_argument("run", type=Path)
    p.add_argument("--plans", type=Path, default=HERE / "plans2")
    p.add_argument("--lease", type=int, required=True)
    p.add_argument("--vram-mb", type=int, default=11000)
    p.add_argument("--ram-mb", type=int, default=20000)
    p.add_argument("--est-wall-s", type=int, default=420)
    p.add_argument("--timeout-s", type=int, default=900)
    p = sub.add_parser("vswap")
    p.add_argument("run", type=Path)
    p.add_argument("--vram-mb", type=int, default=11000)
    p.add_argument("--ram-mb", type=int, default=20000)
    p.add_argument("--est-wall-s", type=int, default=300)
    p.add_argument("--timeout-s", type=int, default=900)
    a = parser.parse_args()
    if a.cmd == "plans":
        print(json.dumps(build_plan(a.sets, a.out), indent=1))
    elif a.cmd == "panel":
        print(json.dumps(stage_panel(a.run.resolve(), a.plans, a.lease, a.vram_mb, a.ram_mb, a.est_wall_s,
                                     a.timeout_s)))
    else:
        print(json.dumps(stage_vswap(a.run.resolve(), a.vram_mb, a.ram_mb, a.est_wall_s, a.timeout_s)))
