"""Isolated installed-wheel controller checks; no actual staging or services."""
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MDB = Path("/Users/jakeholl/domains/saturn-pub/packages/mdb")
PY = "/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python"
attempt = ROOT / "attempt-01"
attempt.mkdir(exist_ok=False)
cohort = attempt / "cohort"
files = ["stage_qwen_cached_qkv.py", "stage_qwen_cached_qkv_saved_audit.py", "stage_qwen_row_sign.py",
    "stage_qwen_sign_shortcut.py", "stage_saved_flux_mask_pair_audit.py", "qwen_cached_qkv_campaign.py", "audit_qwen_cached_qkv.py"]
producer = runpy.run_path(str(MDB / "qualification/qwen_cached_qkv_campaign.py"))
reader = runpy.run_path(str(MDB / "qualification/audit_qwen_cached_qkv.py"))
files.extend(producer["HELPERS"])
files.extend(reader["HELPERS"])
sources = {"qualification/" + name: MDB / "qualification" / name for name in files}
for name in ("test_stage_qwen_cached_qkv.py", "test_stage_qwen_cached_qkv_saved_audit.py"):
    sources["tests/" + name] = MDB / "tests" / name
for name, source in sources.items():
    target = cohort / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
command = [PY, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(attempt / "tmp"),
    "tests/test_stage_qwen_cached_qkv.py", "tests/test_stage_qwen_cached_qkv_saved_audit.py"]
result = subprocess.run(command, cwd=cohort, capture_output=True, text=True, timeout=120)
(attempt / "stdout.txt").write_text(result.stdout)
(attempt / "stderr.txt").write_text(result.stderr)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(command=command, cwd=str(cohort), returncode=result.returncode, source_hashes={name: sha(path) for name,path in sources.items()},
    stdout_sha256=sha(attempt / "stdout.txt"), stderr_sha256=sha(attempt / "stderr.txt"),
    scope="Synthetic metadata/API/layout/source/resource/payload guards, CLI/delegation and actual installed MDB bootstrap argv transport only. Zero original array/weight/model/tokenizer/CUDA reads or calls, services, actual stage or job submission.")
(attempt / "execution.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True)); print(result.stdout); print(result.stderr)
raise SystemExit(result.returncode)
