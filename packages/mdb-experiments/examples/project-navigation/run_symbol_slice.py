"""Demonstrate installed CLI slices, then reopen with frameworks blocked."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for project and raw evidence")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    mdb = str(Path(sys.executable).with_name("mdb"))
    project = root / "project"
    live = [
        mdb,
        "public",
        "--tiny",
        "qwen3",
        "--project",
        str(project),
        "--session-id",
        "demo",
        "--json",
    ]

    def run(label, command, *, expected=0):
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        (root / f"{label}.jsonl").write_text(result.stdout)
        (root / f"{label}.stderr").write_text(result.stderr)
        if result.returncode != expected:
            raise RuntimeError(
                f"{label}: {result.returncode}\n{result.stderr}\n{result.stdout[-2000:]}"
            )
        try:
            return [json.loads(result.stdout)]
        except json.JSONDecodeError:
            return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]

    script = root / "start.mdb"
    script.write_text(
        "step 2\ncheckpoint named-carrier\n"
        "project symbol-bind demo.carrier --port hidden --context demo "
        '--context-slot tokens --context-slot hidden --claim "Supplied carrier at this boundary"\n'
        "project slice --symbol demo.carrier\n"
    )
    initial = run("initial", live + ["--script", str(script)])
    bound, sliced = initial[-2:]
    assert bound["support"] == "native-support-verified"
    assert sliced["support"] == "metadata-support-verified"
    assert sliced["symbol_binding"] == bound["binding"]
    assert sliced["complete"]
    cut, slice_id = sliced["cut_id"], sliced["slice_id"]
    finding = run(
        "finding",
        live
        + [
            "--resume",
            "--command",
            f'project finding symbol:demo.carrier --title "Supplied carrier observed" '
            f"--cut {cut} --slice-id {slice_id}",
        ],
    )[-1]
    assert finding["details"]["slice_id"] == slice_id

    script = root / "edit.mdb"
    script.write_text("fork edited\nset hidden 0\ncommit\nproject slice --symbol demo.carrier\n")
    edited = run("edited", live + ["--resume", "--script", str(script)], expected=1)
    assert "input context" in json.dumps(edited[-1])
    sessions = run("sessions", [mdb, "project", str(project), "sessions"])[0]
    future = sessions[0]["selected_cut"]
    downstream = run(
        "downstream",
        [
            mdb,
            "project",
            str(project),
            "slice",
            "--symbol",
            "demo.carrier",
            "--cut",
            cut,
            "--context",
            "demo",
            "--to-cut",
            future,
        ],
    )[0]
    assert downstream["complete"] and downstream["consumers"]
    saved = run("reopened-slice", [mdb, "project", str(project), "slice-show", slice_id])[0]
    assert saved == sliced
    findings = run("reopened-findings", [mdb, "project", str(project), "findings"])[0]
    assert findings[0]["details"]["symbol_binding"] == bound["binding"]
    offline = run(
        "offline-check",
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc
import json
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'diffusers', 'saturn', 'common'}:
            raise ImportError('offline check attempted forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
from mdb import Project
project = Project(sys.argv[1])
sliced = project.slice_record('demo', sys.argv[2])
assert sliced['support'] == 'metadata-support-verified'
assert project.graph().verify().ok
assert project.findings('demo')[0]['details']['slice_id'] == sys.argv[2]
print(json.dumps({'frameworks_blocked': True, 'project_graph_verified': True}))
""",
            str(project),
            slice_id,
        ],
    )[0]
    summary = {
        "specimen": "random tiny Qwen3; CPU FP32; CLI mechanics only",
        "native_binding_support": bound["support"],
        "offline_support": sliced["support"],
        "source_cut": cut,
        "slice_id": slice_id,
        "writers": [row["operation"] for row in sliced["writers"]],
        "consumers_toward_edited_cut": [row["operation"] for row in downstream["consumers"]],
        "binding_refused_after_context_edit": True,
        "saved_slice_survived_fresh_process": saved == sliced,
        **offline,
        "finding_id": finding["annotation_id"],
        "authority": sliced["authority"],
        "causal_effect": sliced["causal_effect"],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
