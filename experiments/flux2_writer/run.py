"""Run construction, recipient feedback, and verified fresh-process replay without a scheduler."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=here / "recipe.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/flux2-writer"))
    for name in ("base-model", "base-revision", "model", "revision", "prompt"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print stages without loading models"
    )
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    for key in ("base_model", "base_revision", "model", "revision", "prompt"):
        if getattr(args, key) is not None:
            recipe[key] = getattr(args, key)
    output = args.output.resolve()
    common = [
        "--model",
        recipe["model"],
        "--revision",
        recipe["revision"],
        "--prompt",
        recipe["prompt"],
        "--seed",
        str(recipe["seed"]),
        "--size",
        str(recipe["size"]),
        "--output",
        str(output),
    ]
    if args.local_files_only:
        common.append("--local-files-only")
    build = [
        sys.executable,
        str(repository / "examples/flux2_build_writer.py"),
        *common,
        "--base-model",
        recipe["base_model"],
        "--base-revision",
        recipe["base_revision"],
        "--rank",
        str(recipe["rank"]),
    ]
    feedback = [
        sys.executable,
        str(repository / "examples/flux2_hotfix.py"),
        *common,
        "--package",
        str(output / "writer.npz"),
        "--expected-count",
        str(recipe["expected_count"]),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "recipe": recipe,
                    "build": build,
                    "feedback": feedback,
                    "replay": feedback
                    + ["--replay", "<saved parent>", "--dose", "<selected dose>"],
                },
                indent=2,
            )
        )
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            "output directory is not empty; choose a new --output to preserve evidence"
        )
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        args.recipe,
        repository / "examples/flux2_build_writer.py",
        repository / "examples/flux2_hotfix.py",
        *sorted((repository / "src/saturn_pub").rglob("*.py")),
    ]
    provenance = {
        "recipe": recipe,
        "source_sha256": {
            str(path.relative_to(repository))
            if path.is_relative_to(repository)
            else path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "commands": [build, feedback],
        "runner": "standalone-python",
    }
    (output / "run-manifest.json").write_text(json.dumps(provenance, indent=2))
    started = time.perf_counter()
    subprocess.run(build, check=True)
    subprocess.run(feedback, check=True)
    report = json.loads((output / "report.json").read_text())
    replay = feedback + ["--replay", report["parent"], "--dose", str(report["selected_dose"])]
    provenance["commands"].append(replay)
    (output / "run-manifest.json").write_text(json.dumps(provenance, indent=2))
    subprocess.run(replay, check=True)
    fresh = json.loads((output / "replay-report.json").read_text())
    exact = {
        name: fresh["records"][name] == report["records"][name]
        for name in ("native", "patched", "zero-dose", "uninstalled")
    }
    result = {
        "schema": "saturn-pub-experiment-result-v1",
        "recipe": recipe,
        "construction": json.loads((output / "construction-report.json").read_text()),
        "selected_dose": report["selected_dose"],
        "counts": {name: row["observed_count"] for name, row in report["records"].items()},
        "fresh_process_exact_replay": exact,
        "elapsed_seconds": time.perf_counter() - started,
        "mechanics_status": "verified" if all(exact.values()) else "replay-mismatch",
        "terminal_status": "not-assessed",
        "claim_boundary": "One labelled development context; generalization and collateral unassessed.",
    }
    (output / "experiment-report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not all(exact.values()):  # Named replay-integrity check, not a scientific score gate.
        raise RuntimeError(
            "fresh-process replay differs; individual branch outcomes were preserved"
        )


if __name__ == "__main__":
    main()
