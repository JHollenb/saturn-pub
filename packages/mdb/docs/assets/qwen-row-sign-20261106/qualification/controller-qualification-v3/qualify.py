"""Collector-only transport mechanics, isolated from real services."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = "/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python"
attempt = ROOT / "attempt-01"
attempt.mkdir(exist_ok=False)
cohort = attempt / "cohort"
sources = {"artifacts/collect_compact.py": ROOT.parent / "collect_compact.py",
    "artifacts/controller-qualification-v3/test_collect_compact.py": ROOT / "test_collect_compact.py",
    "qualification/collect_flux_remote_directory.py": Path("/Users/jakeholl/domains/saturn-pub/packages/mdb/qualification/collect_flux_remote_directory.py")}
for name, source in sources.items():
    target = cohort / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
command = [PY, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(attempt / "tmp"), "artifacts/controller-qualification-v3/test_collect_compact.py"]
result = subprocess.run(command, cwd=cohort, capture_output=True, text=True, timeout=60)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(attempt / "stdout.txt").write_text(result.stdout)
(attempt / "stderr.txt").write_text(result.stderr)
receipt = dict(command=command, cwd=str(cohort), returncode=result.returncode, source_hashes={n: sha(p) for n,p in sources.items()},
    stdout_sha256=sha(attempt / "stdout.txt"), stderr_sha256=sha(attempt / "stderr.txt"),
    scope="Synthetic request/member/budget/CLI/pre-network transport guards only; no remote contact, collection, model or scientific data access.")
(attempt / "execution.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True)); print(result.stdout); print(result.stderr)
raise SystemExit(result.returncode)
