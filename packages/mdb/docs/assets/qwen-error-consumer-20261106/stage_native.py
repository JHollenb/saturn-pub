"""Root stages the one frozen original component study, using intent-first sizing."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
ARTIFACTS = ROOT.parent
PYTHON = "/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python"


def main():
    frontend = BASE / "saturn-pub/packages/mdb/qualification/stage_qwen_error_consumer.py"
    if hashlib.sha256(frontend.read_bytes()).hexdigest() != "c7574da612f8548e0574e627496e273c6ac8f81e8b04f5984fb368759d21cf28":
        raise ValueError("Qualified native frontend changed")
    world = ARTIFACTS / "mdb-qwen-row-sign-20261106"
    context = ARTIFACTS / "mdb-qwen-sign-shortcut-20261106"
    values = {"world-run": world / "run-02", "world-compact": world / "compact-collection-01/extracted/binding-minimal-result",
        "world-seal": ROOT / "source-metadata-v1/world_seal.json", "context-run": context / "run-02",
        "context-compact": context / "compact-collection-01/extracted/binding-minimal-result",
        "context-seal": context / "frontier-metadata-v1/transaction/seal.json", "selection": ROOT / "selection-v1",
        "parameter-subset": ROOT / "planning-v1/qwen-block19-parameter-artifacts.json"}
    command = [PYTHON, "-I", "-B", str(frontend), str(ROOT / "run-01")]
    for key, path in values.items():
        command.extend(["--" + key, str(path)])
    record = ROOT / "root-staging-v1"
    record.mkdir(exist_ok=False)
    (record / "command.json").write_text(json.dumps(dict(command=command, sizing="intent-first", original_forward_authorized=False), indent=2) + "\n")
    result = subprocess.run(command, cwd=BASE, capture_output=True, text=True, timeout=120)
    (record / "stdout.txt").write_text(result.stdout)
    (record / "stderr.txt").write_text(result.stderr)
    (record / "execution.json").write_text(json.dumps(dict(returncode=result.returncode), indent=2) + "\n")
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
