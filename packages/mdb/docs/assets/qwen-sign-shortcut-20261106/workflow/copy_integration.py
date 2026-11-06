"""Copy the owned compact result and source closures, never arrays or weights."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
REPO = BASE / "saturn-pub"
DEST = REPO / "packages/mdb/docs/assets/qwen-sign-shortcut-20261106"
VAULT = BASE / "obsidian/images/mdb-qwen-sign-shortcut-20261106"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    VAULT.mkdir(parents=True, exist_ok=False)
    copied = {}

    def copy(source, relative):
        assert source.is_file() and not source.is_symlink(), source
        assert source.suffix not in {".npy", ".pt", ".pth", ".safetensors", ".whl"}, source
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), target
        shutil.copyfile(source, target)
        assert sha(target) == sha(source)
        copied[str(relative)] = dict(source=str(source.relative_to(BASE)), bytes=source.stat().st_size, sha256=sha(source))

    for source in sorted((ROOT / "publication-v1").rglob("*")):
        if source.is_file():
            copy(source, Path("publication") / source.relative_to(ROOT / "publication-v1"))
    for source in sorted((ROOT / "saved-report-verification-v1").rglob("*")):
        if source.is_file():
            copy(source, Path("verification") / source.relative_to(ROOT / "saved-report-verification-v1"))
    for run in ("run-01", "run-02", "audit-01"):
        for name in ("request.json", "staged-command.json", "admission-command.json", "submission.json", "finite-run-review-v1.json", "source-cohort.json", "source-job.json", "source-worker.log", "job-first-observed.json", "job-completed.json"):
            source = ROOT / run / name
            if source.is_file():
                copy(source, Path("execution") / run / name)
        request = json.loads((ROOT / run / "request.json").read_text())
        for name, expected in request["source_hashes"].items():
            source = ROOT / run / "payload" / name
            assert sha(source) == expected
            copy(source, Path("executed-source") / run / name)
    qualifications = ("helper-qualification-v1", "producer-qualification-v1", "audit-qualification-v2", "saved-audit-controller-qualification-v1", "stager-qualification-v1", "stager-qualification-v2", "saved-report-verifier-qualification-v1", "publisher-qualification-v1", "launcher-config-argument-check-root-v1")
    for directory in qualifications:
        for source in sorted((ROOT / directory).iterdir()):
            if source.is_file() and source.suffix in {".json", ".log", ".txt", ".py"}:
                copy(source, Path("qualification") / directory / source.name)
    for name in ("collect_compact.py", "verify_saved_reports.py", "review_launcher_correction.py", "copy_integration.py", "compact-collection-01/collection-receipt.json", "stager-source-archive-v1/archive.json"):
        copy(ROOT / name, Path("workflow") / name)
    for name in ("holistic-review-09.json", "direction-09.md"):
        copy(BASE / "artifacts/mdb-flux-understanding-20261106" / name, Path("advisory") / name)
    for name in ("shortcut-errors.png", "oddness-and-zero-signs.png", "conditional-computation-path.png"):
        source = ROOT / "publication-v1" / name
        shutil.copyfile(source, VAULT / name)
        assert sha(VAULT / name) == sha(source)
    report = dict(schema="mdb-qwen-shortcut-integration-copy-v1", copied=copied,
                  files=len(copied), bytes=sum(row["bytes"] for row in copied.values()),
                  vault_figures={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in VAULT.iterdir()},
                  arrays_or_parameter_exports=False)
    for target in (ROOT / "integration-copy-v1.json", DEST / "integration-copy.json"):
        with target.open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: report[key] for key in ("files", "bytes", "arrays_or_parameter_exports")}, sort_keys=True))


if __name__ == "__main__":
    main()
