"""Transport one already-sealed JSON metadata member, never model payloads."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "/mnt/big/mdb/research/job-74efe5506ecf/qwen-row-sign-result/transaction/seal.json"
EXPECTED = "12077f1ee8ca0f2b23c1c440e11a9dea7b5f07058270041594320289055f0121"


def main():
    inventory = ROOT.parent / "mdb-qwen-row-sign-20261106/compact-collection-01/extracted/binding-minimal-result/source-files.json"
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != "d36f790376a85ad995cf20f01cff89fb6e04d282fe1290f677ba348ec98f4c8f":
        raise ValueError("Frozen world inventory changed")
    record = json.loads(inventory.read_text())["files"]["transaction/seal.json"]
    if record != dict(bytes=17663, sha256=EXPECTED):
        raise ValueError("Expected metadata record differs")
    script = "import base64; from pathlib import Path; p=Path(" + repr(SOURCE) + "); assert not p.is_symlink(); print(base64.b64encode(p.read_bytes()).decode())"
    command = ["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"]
    result = subprocess.run(command, input=script, capture_output=True, text=True, timeout=30, check=True)
    raw = base64.b64decode(result.stdout.strip(), validate=True)
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("Transported metadata differs")
    json.loads(raw)
    out = ROOT / "source-metadata-v1"
    out.mkdir(exist_ok=False)
    (out / "world_seal.json").write_bytes(raw)
    report = dict(schema="mdb-existing-world-seal-transport-v1", source=SOURCE, bytes=len(raw), sha256=EXPECTED,
        command=command, remote_script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        stderr=result.stderr, returncode=result.returncode, model_calls=0, arrays_or_weights_read=0,
        scope="Transport only; immutable existing forecast-seal JSON joined to original raw inventory.")
    (out / "transport.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
