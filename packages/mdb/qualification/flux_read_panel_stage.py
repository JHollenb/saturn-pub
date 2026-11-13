"""Build, seal, stage, submit and report FLUX.2 Klein read panels (mdb.attention_panels + attention_port).

Every step is offline except ``submit`` (mrun) — no model is loaded here.

  PY=/Users/jakeholl/domains/saturn-pub/packages/mdb/.venv/bin/python
  SRC=<worktree>/packages/mdb/src
  PYTHONPATH=$SRC $PY flux_read_panel_stage.py plans PLANS                      # build + seal panels, split leases
  PYTHONPATH=$SRC $PY flux_read_panel_stage.py stage RUN --plans PLANS --job smoke
  PYTHONPATH=$SRC $PY flux_read_panel_stage.py stage RUN --plans PLANS --job full:3
  PYTHONPATH=$SRC $PY flux_read_panel_stage.py stage RUN --plans PLANS --job regenerate --recipes DIR \
      --arms smoke-legacy@4242:z:own60:ball full-00@4242:m:own60:red
  PYTHONPATH=$SRC $PY flux_read_panel_stage.py submit RUN
  PYTHONPATH=$SRC $PY flux_read_panel_stage.py report --plans PLANS --results JOBDIR... --out OUT

The payload wheel is built from whichever ``mdb`` the interpreter imports (editable install), so
PYTHONPATH must point at the checkout under test; ``stage`` prints ``mdb_source``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_read_panel_worker.py"
FLUX = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
HEAD_SETS = FLUX / "head_sets3.json"
OLD_JOB = "job-a0bca63168b4"
OLD_RESULT_BEAST = f"/mnt/hdd8tb/mdb/research/{OLD_JOB}/read-graph-result"
MODEL_PATH = "/mnt/big/llm-models/FLUX.2-klein-4B"
SCRATCH = "/mnt/hdd8tb/mdb/research"
ENVIRONMENT = {"uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache",
               "torch_home": "/mnt/hdd8tb/mdb/torch", "extra_model_roots": "/mnt/big/llm-models", "python": "3.14.6"}
EXTRA_ENV = {"TMPDIR": "/mnt/hdd8tb/mdb/tmp", "CUDA_CACHE_PATH": "/mnt/hdd8tb/mdb/cuda-cache",
             "TORCHINDUCTOR_CACHE_DIR": "/mnt/hdd8tb/mdb/torchinductor-cache",
             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
PROMPTS = {"cat": "a photorealistic black cat sitting beside a red ball in fresh snow at dawn, soft blue light",
           "car": "a small blue car parked under a large oak tree in heavy rain at night"}
WORDS = {"cat": ("red", "black", "ball", "snow"), "car": ("night", "rain")}
OTHER = {"red": ("black",), "black": ("ball",), "ball": ("snow",), "snow": ("red",),
         "night": ("rain",), "rain": ("night",)}
CUT = ("own60", "own120", "own240", "mod", "noun", "modI", "nounI")
KEEP = ("own60", "own120", "own240", "modI", "nounI")
EXTERNAL = {"rand120": "own120", "rand240": "own240"}
BLOCKS, HEADS, JOINT = 25, 24, 5
CONTEXTS = ("4242", "9001")
WORK_LIMIT = 256          # mdb.scheduling_contract.MAX_LOCAL_WORK_ITEMS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exist_ok=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if exist_ok else "x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def items_map():
    items = {w: {"prompt": pid, "word": w} for pid, ws in WORDS.items() for w in ws}
    items.update({pid: {"prompt": pid, "word": None} for pid in PROMPTS})
    return items


# ======================================================================================= plans

def universe():
    return tuple((b, h) for b in range(BLOCKS) for h in range(HEADS))


def word_masses():
    """Each word's own image<-row write-mass share per (block, head): full-a capture, seed 4242."""
    import numpy as np
    from mdb.flux_read_panels import unit_mass
    tokens = json.loads((FLUX / "runs/full-a/result/tokens.json").read_text())
    out = {}
    for pid, words in WORDS.items():
        tw = np.load(FLUX / f"runs/full-a/result/capture.{pid}.4242.npz")["tok_w"].astype(np.float64)
        if tw.shape[1:3] != (BLOCKS, HEADS):
            raise ValueError("capture geometry differs")
        for w in words:
            rows = [r for x in tokens[pid]["words"] if x["word"] == w for r in x["rows"]]
            out[w] = unit_mass(tw, rows, HEADS)
    return out


def legacy_word_panel(word, contexts, sets):
    """The 2026-11-11 headfree arm list for one word, as executed (tests/test_attention_panels.py)."""
    from mdb.attention_panels import WAIVABLE, ReadPanelSpec, read_panel
    return read_panel(ReadPanelSpec(
        items=(word,), universe=universe(), sets={word: sets}, contexts=contexts,
        cut_sets=CUT, keep_sets=KEEP, external_controls=EXTERNAL, cut_types=("zero",),
        cut_type_overrides={"own60": ("zero", "renorm")}, random=None, other_items={}, reference_types=("zero",),
        floors=False, waive={k: "reproduce the 2026-11-11 headfree run (job-a0bca63168b4) as executed"
                             for k in WAIVABLE},
        port={"precision": "legacy", "zero_method": "legacy FP32 recompute of every head in a touched block"}))


def full_word_panel(word, contexts, sets, mass):
    from mdb.attention_panels import RandomControls, ReadPanelSpec, read_panel
    strata = tuple("joint" if b < JOINT else "single" for b, _ in universe())
    return read_panel(ReadPanelSpec(
        items=(word,), universe=universe(), sets={word: sets}, contexts=contexts,
        cut_sets=CUT, keep_sets=KEEP, external_controls=EXTERNAL, nested=("own60", "own240"),
        other_items={word: OTHER[word]},
        random=RandomControls(n=1, strata=strata, mass=mass, mass_tolerance=0.02, disjoint=True, seed=0),
        port={"precision": "native", "zero_method": "delta"}))


def build_plans():
    from mdb.flux_read_panels import merge_panels, prompt_level_arms
    named = json.loads(HEAD_SETS.read_text())["sets"]
    masses = word_masses()
    sets = {w: named[f"{pid}:{w}"] for pid, ws in WORDS.items() for w in ws}
    all_words = [w for ws in WORDS.values() for w in ws]
    legacy_note = ("as executed by read_graph_worker.py headcut (head_sets3.json); prompt-level `sham` = "
                   "native:<pid>, `img<-alltext` = m:img<-alltext:<pid>")
    full_note = ("per-word read_panel calls merged because RandomControls.mass is one vector for every item "
                 "(attention_panels.py RandomControls / _draw_random); random mass = each word's own image<-row "
                 "write-mass share, full-a capture seed 4242")
    plans = {
        "smoke-legacy": merge_panels([legacy_word_panel("ball", ("4242",), sets["ball"])],
                                     extra_arms=prompt_level_arms("cat", cut_types=("renorm",), floors=False),
                                     notes=[legacy_note]),
        "smoke-new": merge_panels([full_word_panel("ball", ("4242",), sets["ball"], masses["ball"])],
                                  extra_arms=prompt_level_arms("cat"), notes=[full_note]),
        "legacy": merge_panels([legacy_word_panel(w, CONTEXTS, sets[w]) for w in all_words],
                               extra_arms=[a for pid in PROMPTS for a in prompt_level_arms(pid, cut_types=("renorm",),
                                                                                          floors=False)],
                               notes=[legacy_note]),
        "full": merge_panels([full_word_panel(w, CONTEXTS, sets[w], masses[w]) for w in all_words],
                             extra_arms=[a for pid in PROMPTS for a in prompt_level_arms(pid)], notes=[full_note]),
    }
    return plans


def write_plans(root):
    from mdb.flux_read_panels import leases_within, work_items
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    items = items_map()
    summary = {}
    for name, panel in build_plans().items():
        cap, leases = leases_within(panel, items, limit=WORK_LIMIT)
        (root / f"{name}.plan.json").write_text(panel.canonical() + "\n")   # canonical form: sha256 = seal
        save(root / f"{name}.seal.json", panel.seal(), exist_ok=True)
        rows = [{"lease": l["lease"], "lease_sha256": l["lease_sha256"], "branches": len(l["branches"]),
                 "work_items": work_items(panel, l, items),
                 "contexts": sorted({c for c, _ in l["branches"]}),
                 "items": sorted({panel.by_id()[a].item for _, a in l["branches"]})} for l in leases]
        save(root / f"{name}.leases.json", {"plan_sha256": panel.sha256(), "cap": cap, "leases": leases}, exist_ok=True)
        kinds = {}
        for a in panel.arms:
            kinds[a.role] = kinds.get(a.role, 0) + 1
        summary[name] = {"plan_sha256": panel.sha256(), "arms": len(panel.arms), "contexts": panel.contexts,
                         "branches": panel.branch_count, "roles": kinds, "lease_cap": cap, "leases": rows,
                         "identity": {k: {kk: vv.get("separable") if isinstance(vv, dict) else vv
                                          for kk, vv in v.items()} for k, v in panel.identity.items()},
                         "random_status": {k: v.get("status") for k, v in panel.controls.items()}}
    save(root / "plans-summary.json", summary, exist_ok=True)
    return summary


def load_plan_files(root, name):
    from mdb.attention_panels import Panel
    root = Path(root)
    plan = json.loads((root / f"{name}.plan.json").read_text())
    seal = json.loads((root / f"{name}.seal.json").read_text())
    leases = json.loads((root / f"{name}.leases.json").read_text())
    panel = Panel.from_json(plan)
    if panel.sha256() != seal["plan_sha256"] or leases["plan_sha256"] != seal["plan_sha256"]:
        raise ValueError(f"{name}: plan, seal and leases disagree")
    return plan, seal, leases, panel


# ===================================================================================== staging

def bind():
    for key, name in {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
                      "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
                      "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}.items():
        os.environ[name] = ENVIRONMENT[key]
    os.environ.setdefault("MRUN_URL", "http://127.0.0.1:19025")
    from mdb.execution_environment import intent
    resolved = intent()
    if any(resolved[key] != value for key, value in ENVIRONMENT.items()):
        raise ValueError("execution environment intent differs")
    return resolved


def command(extra=()):
    from mdb.payload import worker_command
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for name, value in EXTRA_ENV.items():
        args.insert(1, name + "=" + value)
    return args + list(extra)


def passes_for(job, plans):
    """Run passes for a job name: smoke | legacy | full:N."""
    out = []
    if job == "smoke":
        specs = [("smoke-legacy", 0, "legacy", True), ("smoke-new", 0, "native", False)]
    elif job == "legacy":
        specs = [("legacy", 0, "legacy", True)]
    elif job.startswith("full:"):
        specs = [("full", int(job.split(":", 1)[1]), "native", False)]
    else:
        raise ValueError("job must be smoke, legacy, full:N or regenerate")
    for name, index, precision, ref in specs:
        plan, seal, leases, panel = load_plan_files(plans, name)
        lease = leases["leases"][index]
        out.append(dict(tag=name if not job.startswith("full:") else f"full-{index:02d}",
                        plan_file=f"plans/{name}.plan.json", plan_source=str(Path(plans) / f"{name}.plan.json"),
                        plan_sha256=seal["plan_sha256"], seal_sha256=seal["seal_sha256"], lease=lease,
                        precision=precision, zero_method="delta",
                        reference_dir=OLD_RESULT_BEAST if ref else None))
    return out


def job_work(passes):
    from mdb.attention_panels import Panel
    from mdb.flux_read_panels import lease_roots
    items = items_map()
    roots, branches = set(), 0
    for p in passes:
        panel = Panel.from_json(json.loads(Path(p["plan_source"]).read_text()))
        roots |= set(lease_roots(panel, p["lease"], items))
        branches += len(p["lease"]["branches"])
    return branches, len(roots)


def stage(run, job, plans, vram_mb, ram_mb, est_wall_s, timeout_s, recipes=None, arms=()):
    import mdb
    environment = bind()
    from mdb.flux_read_panels import RUN_SCHEMA
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    base = dict(schema=RUN_SCHEMA, job=job, model="flux2-klein-4b-distilled", model_path=MODEL_PATH,
                output_scratch=SCRATCH, height=256, width=256, steps=4, prompts=PROMPTS, items=items_map(),
                contexts={c: {"seed": int(c)} for c in CONTEXTS}, retain=[], max_wall_seconds=540,
                head_sets_sha256=sha(HEAD_SETS), reference_job=OLD_JOB)
    extras, extra_args = [], []
    if job == "regenerate":
        if not recipes or not arms:
            raise ValueError("regenerate needs --recipes and --arms")
        from mdb.flux_read_panels import validate_recipe
        rdir = run / "recipes"
        rdir.mkdir()
        # --arms TAG@CONTEXT:ARM_ID (TAG = pass tag, e.g. smoke-legacy, full-03): the same (context, arm)
        # exists in a legacy and a native pass, so the pass tag picks the recipe.
        index = {}
        for path in Path(recipes).rglob("*.json"):
            try:
                r = validate_recipe(json.loads(path.read_text()))
            except Exception:
                continue
            index[(r["arm"]["pass_tag"], r["arm"]["context"], r["arm"]["arm"])] = (path, r)
        chosen = []
        for item in arms:
            where, _, arm = item.partition(":")
            tag, _, context = where.partition("@")
            path, r = index[(tag, context, arm)]
            target = rdir / f"{tag}__{context}__{path.stem}.json"
            shutil.copyfile(path, target)
            extras.append((target, f"recipes/{target.name}"))
            chosen.append(f"{context}:{arm}")
        if len(set(chosen)) != len(chosen):
            raise ValueError("the worker indexes recipes by (context, arm): pick distinct ones")
        config = dict(base, mode="regenerate", hash_checkpoint=True, regenerate=list(arms))
        branches, roots = len(arms), 0
        extra_args = ["--regenerate", *chosen, "--recipes", "recipes"]
    else:
        passes = passes_for(job, plans)
        branches, roots = job_work(passes)
        if branches + roots > WORK_LIMIT:
            raise ValueError(f"{job}: {branches} branches + {roots} roots exceed {WORK_LIMIT} work items")
        for p in passes:
            if (Path(p["plan_source"]), p["plan_file"]) not in extras:
                extras.append((Path(p["plan_source"]), p["plan_file"]))
        passes = [{k: v for k, v in p.items() if k != "plan_source"} for p in passes]
        config = dict(base, mode="panel", passes=passes, hash_checkpoint=job == "smoke")
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKER, preflight.name],
                     "path_token_allowlist": [MODEL_PATH, SCRATCH, "flux-read-panel"]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    work = branches + roots
    src = Path(mdb.__file__).resolve().parent
    source_hashes = {WORKER: sha(HERE / WORKER), **{name: sha(path) for path, name in extras},
                     **{f"mdb/{m}.py": sha(src / f"{m}.py") for m in ("attention_port", "attention_panels",
                                                                       "flux_read_panels")}}
    request = dict(schema="mdb-flux-read-panel-request-v1", config=config, model=config["model"], task_family="forward",
                   backend="diffusers", device="cuda", dtype="bfloat16", batch_size=1, height=256, width=256,
                   needs={"cuda": True}, execution_environment=environment,
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(), source_hashes=source_hashes,
                   scheduling_contract=one_lease_scheduling_contract(
                       model_loads=1, logical_operations=work, logical_branches=work, logical_checkpoint_replays=0,
                       max_local_work_items=WORK_LIMIT,
                       checkpoint_policy="one resident Klein 4B lease; prompts encoded first, encoder released; "
                                         "reference roots then sealed-lease branches in process; no retries"),
                   resource_exception=dict(ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, disk_gb=2,
                                           est_wall_s=est_wall_s, timeout_s=timeout_s,
                                           reason="Klein 4B BF16 B1 256px 4-step resident lane; encoder released "
                                                  "before denoising. job-345766dba0e8 measured 7991 MB peak "
                                                  "allocated; job-a0bca63168b4 8738 MB."))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    cmd = command(extra_args)
    save(run / "staged-command.json", {"command": cmd, "request_sha256": request["request_sha256"],
                                        "worker_sha256": sha(run / "payload" / WORKER), "mdb_source": mdb.__file__,
                                        "work_items": work, "branches": branches, "roots": roots})
    return dict(run=str(run), job=job, request_sha256=request["request_sha256"], mdb_source=mdb.__file__,
                worker_sha256=sha(run / "payload" / WORKER), work_items=work, branches=branches, roots=roots,
                config_bytes=len(raw))


def submit(run, priority):
    bind()
    from mdb.job_support import guarded_mrun_launch
    run = Path(run)
    request = json.loads((run / "request.json").read_text())
    staged = json.loads((run / "staged-command.json").read_text())
    if (run / "submission.json").exists():
        raise ValueError("already submitted")
    for name, digest in request["source_hashes"].items():
        if name.startswith("mdb/"):
            continue                       # checked by the worker bootstrap against the installed wheel
        if sha(run / "payload" / name) != digest:
            raise ValueError("staged payload changed: " + name)
    resource = request["resource_exception"]
    job = guarded_mrun_launch(staged["command"], request=request, experiment="mdb-flux-read-panel-" +
                              request["config"]["job"].replace(":", "-"),
                              task="forward", model=request["model"], payload=run / "payload", prefer_host="beast",
                              gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
                              disk_gb=resource["disk_gb"], cpu_threads=resource["cpu_threads"],
                              est_wall_s=resource["est_wall_s"], timeout_s=resource["timeout_s"], priority=priority,
                              retry_on_kill=False, preflight=True,
                              note="mdb FLUX read panel (attention_port + attention_panels): " + request["config"]["job"])
    save(run / "submission.json", {"job_id": str(job)})
    return dict(job_id=str(job), run=str(run))


# ====================================================================================== report

def _rows(results, pattern):
    out = []
    for d in results:
        for path in sorted(Path(d).rglob(pattern)):
            out += json.loads(path.read_text())
    return out


def report(plans, results, out):
    """(a) legacy pixel identity, (b) gone tables, (c) regeneration, (d) verdicts. Writes JSON + text."""
    from mdb.attention_panels import arm_effects, verdicts
    from mdb.flux_read_panels import format_gone, gone_table, merge_results
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    # ---------------------------------------------------------------- (a) legacy reproduction
    legacy_rows = _rows(results, "metrics.legacy.json") + _rows(results, "metrics.smoke-legacy.json")
    root_rows = _rows(results, "roots.json")
    leg = {"arms": len(legacy_rows), "identical": 0, "not_identical": [], "missing_reference": []}
    for r in legacy_rows:
        lr = r.get("legacy_reference") or {}
        if not lr.get("found"):
            leg["missing_reference"].append(f"{r['context']}:{r['arm']}")
        elif lr["identical"]:
            leg["identical"] += 1
        else:
            leg["not_identical"].append({"arm": f"{r['context']}:{r['arm']}", "px_mad": lr["px_mad"], "px_max": lr["px_max"]})
    roots_leg = [r for r in root_rows if r.get("legacy_reference")]
    leg["roots"] = {"compared": sum(1 for r in roots_leg if r["legacy_reference"].get("found")),
                    "identical": sum(1 for r in roots_leg if r["legacy_reference"].get("identical")),
                    "not_identical": [f"{r['context']}:{r['root']}" for r in roots_leg
                                      if r["legacy_reference"].get("found") and not r["legacy_reference"]["identical"]]}
    save(out / "a-legacy-identity.json", leg, exist_ok=True)
    lines += ["(a) legacy precision vs saved job-a0bca63168b4 PNGs",
              f"  arms {leg['arms']}, pixel-identical {leg['identical']}, not identical {len(leg['not_identical'])}, "
              f"reference missing {len(leg['missing_reference'])}",
              f"  roots compared {leg['roots']['compared']}, identical {leg['roots']['identical']}"]
    for x in leg["not_identical"]:
        lines.append(f"  NOT identical: {x}")
    # ------------------------------------------------------------------- (b) gone tables
    old = {r["word"]: r for r in json.loads((FLUX / "runs/headfree/headfree_table.json").read_text())}
    cols = [f"z:{c}:{{w}}" for c in ("own60", "own120", "own240", "rand120", "rand240", "mod", "noun", "modI", "nounI")]
    cols += [f"z:keep-{c}:{{w}}" for c in KEEP] + ["m:own60:{w}"]
    extra_cols = ([f"m:{c}:{{w}}" for c in CUT if c != "own60"] + [f"m:keep-{c}:{{w}}" for c in KEEP] +
                  ["z:own240-minus-own60:{w}", "m:own240-minus-own60:{w}"] +
                  [f"z:rand0-{c}:{{w}}" for c in CUT] + [f"z:keep-rand0-{c}:{{w}}" for c in KEEP] +
                  [f"z:{c}@rows-{{o}}:{{w}}" for c in CUT] + [f"z:keep-{c}@rows-{{o}}:{{w}}" for c in KEEP])
    words = [w for ws in WORDS.values() for w in ws]
    tables = {}
    for tag, pattern in (("legacy", "metrics.legacy.json"), ("full", "metrics.full-*.json")):
        rows = _rows(results, pattern)
        if not rows:
            continue
        px_vs_del = {(r["context"], r["arm"]): r["vs_del"]["px_mad"] for r in rows if r.get("vs_del")}
        mad_nd = {(r["context"], r["arm"][4:]): r["vs_native"]["px_mad"] for r in rows if r["kind"] == "deletion"}
        floors = None
        if tag == "full":
            _, _, _, panel = load_plan_files(plans, "full")
            merged = merge_results(panel, [json.loads(p.read_text()) for d in results
                                           for p in sorted(Path(d).rglob("results.full-*.px_mad.json"))])
            floors = {(e["context"], e["arm"]): e["floor"] for e in arm_effects(panel, merged)}
        tcols = cols + (extra_cols if tag == "full" else [])
        table = []
        for w in words:
            wc = [c.replace("{o}", OTHER[w][0]) for c in tcols]
            table += gone_table(px_vs_del, mad_nd, [w], wc, contexts=list(CONTEXTS), floors=floors)
        tables[tag] = table
        save(out / f"b-gone-{tag}.json", table, exist_ok=True)
    lines.append("\n(b) gone = (d(arm) - d(iso)) / (d(none) - d(iso)), d = 1 - MAD(arm, del) / MAD(native, del)")
    for w in words:
        o = old.get(w)
        lines.append(f"\n  {w}:")
        hdr = ["own60", "own120", "own240", "rand120", "rand240", "mod", "noun", "modI", "nounI",
               "keep-own60", "keep-own120", "keep-own240", "keep-modI", "keep-nounI", "m:own60"]
        lines.append("    " + f"{'source':<22}" + " ".join(f"{h[-10:]:>10}" for h in hdr))
        for ctx_i, ctx in enumerate(CONTEXTS):
            if o:
                lines.append("    " + f"{'old ' + ctx:<22}" + " ".join(f"{o['seeds'][ctx_i][h]:10.2f}" for h in hdr))
            for tag in ("legacy", "full"):
                t = next((x for x in tables.get(tag, []) if x["word"] == w), None)
                p = next((x for x in (t or {}).get("contexts", []) if x["context"] == ctx), None)
                if p is None:
                    continue
                keys = [f"z:{h}:{{w}}" if not h.startswith("m:") else f"{h}:{{w}}" for h in hdr]
                lines.append("    " + f"{tag + ' ' + ctx:<22}" + " ".join(
                    f"{p['gone'][k]:10.2f}" if k in p["gone"] else f"{'-':>10}" for k in keys))
                if tag == "full":
                    lines.append("    " + f"{'  floor bound ' + ctx:<22}" + " ".join(
                        f"{p['floor_gone_bound'][k]:10.3f}" if k in p.get("floor_gone_bound", {}) else f"{'-':>10}"
                        for k in keys))
    # ------------------------------------------------------------------- (c) regeneration
    regen = _rows(results, "regenerate.json")
    if regen:
        save(out / "c-regenerate.json", regen, exist_ok=True)
        lines.append("\n(c) regeneration from recipes")
        for r in regen:
            lines.append(f"  {r['context']}:{r['arm']} match={r['match']} same_lane={r.get('same_lane')} " +
                         " ".join(f"{k.replace('_sha256', '')}={v['equal']}" for k, v in r["hashes"].items()))
    # ------------------------------------------------------------------------ (d) verdicts
    full_results = [json.loads(p.read_text()) for d in results for p in sorted(Path(d).rglob("results.full-*.px_mad.json"))]
    if full_results:
        _, _, _, panel = load_plan_files(plans, "full")
        merged = merge_results(panel, full_results)
        save(out / "d-results-px.json", merged, exist_ok=True)
        records = verdicts(panel, merged)
        save(out / "d-verdicts-all.json", records, exist_ok=True)
        wanted = [r for r in records if r["item"] in ("red", "black", "ball")
                  and r["unit_set"] in ("own60", "modI", "nounI", "keep-own60", "keep-modI", "keep-nounI")]
        save(out / "d-verdicts-selected.json", wanted, exist_ok=True)
        lines.append("\n(d) verdicts (px MAD vs iso; picture judgement slot left empty: picture=None)")
        for r in wanted:
            lines.append(f"  {r['item']:<6} {r['unit_set']:<11} {r['claim']:<10} status={r['status']:<15} "
                         f"agree={r['cut_types_agree']} missing={r['missing_cut_types']} picture={r['picture']}")
            for t, ct in r["cut_types"].items():
                for c in ct["contexts"]:
                    ctrl = "; ".join(f"{x['control'].split(':')[1]}({x['kind']},sep={x['separates_identity']}) "
                                     f"e={x['control_effect']:.4f} m={x['margin']:+.4f} >floor={x['beyond_floor']}"
                                     for x in c.get("controls", []))
                    lines.append(f"      {t:<6} {c['context']} {c['status']:<15} effect={c.get('effect', float('nan')):.4f} "
                                 f"floor={(c.get('floor') if c.get('floor') is not None else float('nan')):.4f} "
                                 f"{('allcut=%.4f ' % c['all_cut_effect']) if 'all_cut_effect' in c else ''}| {ctrl}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plans")
    p.add_argument("root", type=Path)
    p = sub.add_parser("stage")
    p.add_argument("run", type=Path)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--vram-mb", type=int, default=12000)
    p.add_argument("--ram-mb", type=int, default=20000)
    p.add_argument("--est-wall-s", type=int, default=300)
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument("--recipes", type=Path)
    p.add_argument("--arms", nargs="*", default=())
    p = sub.add_parser("submit")
    p.add_argument("run", type=Path)
    p.add_argument("--priority", type=int, default=70)
    p = sub.add_parser("report")
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.cmd == "plans":
        s = write_plans(a.root)
        print(json.dumps({k: {kk: v[kk] for kk in ("plan_sha256", "arms", "branches", "lease_cap", "roles")} |
                          {"leases": [(l["branches"], l["work_items"]) for l in v["leases"]]} for k, v in s.items()},
                         indent=1))
    elif a.cmd == "stage":
        print(json.dumps(stage(a.run.resolve(), a.job, a.plans, a.vram_mb, a.ram_mb, a.est_wall_s, a.timeout_s,
                               a.recipes, a.arms), sort_keys=True))
    elif a.cmd == "submit":
        print(json.dumps(submit(a.run.resolve(), a.priority), sort_keys=True))
    else:
        report(a.plans, a.results, a.out)
