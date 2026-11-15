"""Run the real installed CLI, retain raw output, and print a compact walkthrough."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
mdb = Path(sys.executable).with_name("mdb")
command = [str(mdb), "public", "--tiny", "qwen3", "--project", str(root / "project"),
           "--session-id", "demo", "--script", str(root / "demo.mdb"), "--json"]
result = subprocess.run(command, capture_output=True, text=True, timeout=90)
(root / "raw-session.jsonl").write_text(result.stdout)
(root / "stderr.log").write_text(result.stderr)
if result.returncode:
    raise RuntimeError(result.stderr + result.stdout[-2000:])
commands = [line for line in (root / "demo.mdb").read_text().splitlines()
            if line and not line.startswith("#")]
outputs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
if len(outputs) != len(commands):
    raise RuntimeError(f"expected {len(commands)} responses, got {len(outputs)}")
steps = list(zip(commands, outputs))
observed = [body["value"] for cmd, body in steps if cmd == "inspect hidden"]
assert observed[0] != observed[1], "intervention did not change hidden"
assert observed[0] == observed[2], "restore did not recover exact hidden descriptor"
analysis = next(body for cmd, body in steps if cmd == "project analyze")
callers = next(body for cmd, body in steps if cmd == "project callers port:hidden")
trace = next(body for cmd, body in steps if cmd == "project trace port:hidden --depth 2")
finding = next(body for cmd, body in steps if cmd.startswith("project finding "))
summary = {
    "model": "random tiny Qwen3, CPU FP32",
    "analysis_groups": len(analysis["result"]["groups"]),
    "writers": sorted({e["source"] for e in callers["edges"]}),
    "trace_nodes": trace["nodes"],
    "original_hidden_sha256": observed[0]["sha256"],
    "edited_hidden_sha256": observed[1]["sha256"],
    "restored_hidden_sha256": observed[2]["sha256"],
    "restore_exact": observed[0] == observed[2],
    "finding_id": finding["annotation_id"],
    "finding_kind": finding["details"]["kind"],
    "terminal_status": finding["details"]["terminal_status"],
}
offline = [str(mdb), "project", str(root / "project"), "findings"]
reopened = subprocess.run(offline, capture_output=True, text=True, check=True, timeout=30)
(root / "reopened-findings.json").write_text(reopened.stdout)
assert json.loads(reopened.stdout)[0]["annotation_id"] == finding["annotation_id"]
summary["finding_survived_new_process"] = True
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
