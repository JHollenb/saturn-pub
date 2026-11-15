"""Lane verbs of the attention read panels: ``mdb lanes qwen-panels`` and ``mdb lanes flux-read-panels``.

Both lanes are run on mrun by a qualification launcher and a worker that are evidence (their bytes
are hashed into every staged payload), so they stay where they are, in the lane package's
``qualification/`` tree, and this module does not copy them. A verb has two kinds of subcommand:

* offline, from this package only: ``info`` (the case, the schemas, the launcher and worker file
  names, the documents) and ``check PLAN [--seal SEAL]`` (load a plan, recompute its sha256 and
  verify its seal);
* launch: ``plans``, ``stage``, ``submit`` and ``report`` (FLUX) / ``analyze`` (Qwen) run the
  qualification launcher with the remaining arguments, exactly as ``python <launcher> SUB ...``
  would. The ``qualification`` directory is given explicitly, by ``--qualification DIR`` or the
  environment variable ``MDB_LANES_QUALIFICATION``; it is never searched for (the package rule:
  launchers never discover sibling checkout paths).

``mdb lanes qwen-panels info`` and ``mdb lanes flux-read-panels info`` need neither numpy nor torch.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ENV = "MDB_LANES_QUALIFICATION"

LANES = {
    "qwen-panels": {
        "module": "qwen_panels",
        "launcher": "qwen_panel_stage.py",
        "worker": "qwen_panel_worker.py",
        "scripts": {"plans": "qwen_panel_stage.py", "stage": "qwen_panel_stage.py",
                    "submit": "qwen_panel_stage.py", "analyze": "qwen_panel_analyze.py"},
        "docs": ["packages/mdb-experiments/docs/QWEN-PANELS.md", "packages/mdb-experiments/docs/ATTENTION-PANELS.md",
                 "packages/mdb-experiments/docs/ATTENTION-PORT.md"],
        "results": "packages/mdb-lanes/qualification/qwen_panels_20261112",
    },
    "flux-read-panels": {
        "module": "flux_read_panels",
        "launcher": "flux_read_panel_stage.py",
        "worker": "flux_read_panel_worker.py",
        "scripts": {"plans": "flux_read_panel_stage.py", "stage": "flux_read_panel_stage.py",
                    "submit": "flux_read_panel_stage.py", "report": "flux_read_panel_stage.py"},
        "docs": ["packages/mdb-experiments/docs/FLUX-READ-PANELS.md", "packages/mdb-experiments/docs/ATTENTION-PANELS.md",
                 "packages/mdb-experiments/docs/ATTENTION-PORT.md"],
        "results": "packages/mdb-lanes/qualification/flux_read_panels_20261112",
    },
}
VERBS = tuple(sorted(LANES))


def usage(verb):
    lane = LANES[verb]
    launch = "|".join(lane["scripts"])
    return (f"usage: mdb lanes {verb} info\n"
            f"       mdb lanes {verb} check PLAN.json [--seal SEAL.json]\n"
            f"       mdb lanes {verb} [--qualification DIR] {{{launch}}} ARGS...\n"
            f"  launch subcommands run qualification/{lane['launcher']} (or the analysis script) from DIR\n"
            f"  (or ${ENV}); in a checkout DIR is packages/mdb-lanes/qualification")


def _module(name):
    import importlib
    return importlib.import_module("." + name, __package__)


def info(verb):
    """The case and the launch files of one lane (plain JSON; numpy and torch are not imported)."""
    lane = LANES[verb]
    out = {"verb": verb, "module": f"{__package__}.{lane['module']}", "launcher": lane["launcher"],
           "worker": lane["worker"], "launch_subcommands": sorted(lane["scripts"]), "docs": lane["docs"],
           "saved_results": lane["results"]}
    m = _module(lane["module"])
    if verb == "qwen-panels":
        out["case"] = {"model": m.MODEL, "model_path": m.MODEL_PATH, "weight_file_sha256": m.WEIGHT_FILE_SHA256,
                       "lanes": list(m.LANES), "rows": {"source": m.SOURCE_ROW, "wrong": m.WRONG_ROW,
                                                        "receiver": m.RECEIVER_ROW},
                       "tracked_tokens": dict(m.TRACKED), "named_heads": {k: list(v) for k, v in m.NAMED_HEADS.items()},
                       "context": m.CONTEXT, "engine": m.ENGINE}
        out["schemas"] = {"results": m.RESULTS_SCHEMA, "recipe": m.RECIPE_SCHEMA}
    else:
        out["schemas"] = {"run": m.RUN_SCHEMA, "recipe": m.RECIPE_SCHEMA}
        out["recipe_fields"] = list(m.RECIPE_FIELDS)
        out["conditioning_kinds"] = list(m.CONDITIONING_KINDS)
    return out


def check(path, seal=None):
    """Load a sealed plan, recompute its sha256, verify its seal when given."""
    from .attention_panels import Panel, verify_seal
    panel = Panel.from_json(json.loads(Path(path).read_text()))
    out = {"plan": str(path), "sha256": panel.sha256(), "family": panel.family, "arms": len(panel.arms),
           "branches": panel.branch_count, "lane": (panel.port or {}).get("lane")}
    if seal is not None:
        verify_seal(panel, json.loads(Path(seal).read_text()))
        out["seal"] = "verified"
    return out


def _qualification(argv):
    """(directory or None, remaining argv) from ``--qualification DIR`` / ``--qualification=DIR``."""
    rest, found = [], None
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--qualification":
            if i + 1 >= len(argv):
                raise SystemExit("--qualification needs a directory")
            found, i = argv[i + 1], i + 2
            continue
        if item.startswith("--qualification="):
            found = item.split("=", 1)[1]
        else:
            rest.append(item)
        i += 1
    return found or os.environ.get(ENV), rest


def launch(verb, sub, args, directory):
    """Run the qualification script of ``sub`` as ``__main__`` with ``args`` (returns its exit code)."""
    if not directory:
        print(f"mdb lanes {verb} {sub}: give the qualification directory with --qualification DIR or "
              f"${ENV} (in a checkout: packages/mdb-lanes/qualification); it is not searched for",
              file=sys.stderr)
        return 2
    script = Path(directory).expanduser().resolve() / LANES[verb]["scripts"][sub]
    if not script.is_file():
        print(f"mdb lanes {verb} {sub}: {script} does not exist", file=sys.stderr)
        return 2
    saved = sys.argv
    sys.argv = [str(script), *([] if sub == "analyze" else [sub]), *args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_:
        code = exit_.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv = saved
    return 0


def main(argv):
    """``argv[0]`` is the verb (``qwen-panels`` or ``flux-read-panels``)."""
    argv = list(argv)
    verb = argv[0] if argv else ""
    if verb not in LANES:
        print(f"panel lanes: unknown verb {verb!r}; verbs: {', '.join(VERBS)}", file=sys.stderr)
        return 2
    directory, rest = _qualification(argv[1:])
    if not rest or rest[0] in {"-h", "--help", "help"}:
        print(usage(verb))
        return 0 if rest else 2
    sub, args = rest[0], rest[1:]
    if sub == "info":
        print(json.dumps(info(verb), indent=1, sort_keys=True))
        return 0
    if sub == "check":
        if not args:
            print(usage(verb), file=sys.stderr)
            return 2
        seal = None
        if "--seal" in args:
            k = args.index("--seal")
            seal = args[k + 1] if k + 1 < len(args) else None
            args = args[:k] + args[k + 2:]
        print(json.dumps(check(args[0], seal), indent=1, sort_keys=True))
        return 0
    if sub in LANES[verb]["scripts"]:
        return launch(verb, sub, args, directory)
    print(f"mdb lanes {verb}: unknown subcommand {sub!r}\n" + usage(verb), file=sys.stderr)
    return 2
