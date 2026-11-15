"""Execute the installed source-only API against four authenticated metadata cases."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "86b7f56e983c41bb43f7169683c3a85415c38807864d773ccba759864777478f"
MODULE_SHA = "820e971a6128aa7f49cd169734fd2a123e8a60924b25a6d08f6d0a46d33996f3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "prepare_packets.py"
    if sha(source) != SOURCE_SHA:
        raise ValueError("Pinned normalizer changed")
    qualification = ROOT / "installed-qualification-v1/qualification.json"
    installed = json.loads(qualification.read_text())
    if installed["installed"]["sha256"] != MODULE_SHA or installed["peer_uncommitted_sources_included"]:
        raise ValueError("Installed evaluator qualification differs")
    out = ROOT / "execution-v1"
    out.mkdir(exist_ok=False)
    commands = []
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    def run(label, arguments):
        command = [installed["python"], "-I", "-B", str(source), *arguments]
        before = time.monotonic()
        result = subprocess.run(command, cwd=out, env=environment, capture_output=True, text=True, timeout=60)
        for stream in ("stdout", "stderr"):
            (out / (label + "." + stream)).write_text(getattr(result, stream))
        commands.append(dict(label=label, command=command, cwd=str(out), returncode=result.returncode,
                             elapsed_s=time.monotonic() - before))
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(label + ": " + result.stderr)

    inputs = ROOT / "packets-v1"
    run("prepare", ["prepare", "--out", str(inputs)])
    run("replay", ["replay", "--inputs", str(inputs), "--out", str(ROOT / "replay-v1"),
        "--expected-input-manifest", sha(inputs / "inputs-manifest.json"), "--expected-module-sha256", MODULE_SHA])
    report = dict(schema="mdb-source-applicability-execution-v1", normalizer_sha256=SOURCE_SHA,
        qualification_sha256=sha(qualification), inputs_manifest_sha256=sha(inputs / "inputs-manifest.json"),
        replay_receipt_sha256=sha(ROOT / "replay-v1/replay-receipt.json"),
        commands_sha256=sha(out / "commands.json"), original_model_calls=0, arrays_or_weights_read=0,
        scope="Actual installed metadata replay; no floating-point prediction or observed outcome is consulted.")
    (out / "execution.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    print((out / "prepare.stdout").read_text())
    print((out / "replay.stdout").read_text())


if __name__ == "__main__":
    main()
