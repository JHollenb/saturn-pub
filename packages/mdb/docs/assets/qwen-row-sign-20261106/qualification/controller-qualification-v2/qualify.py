"""Installed, isolated transport-only qualification; no real collection or job."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MDB = Path("/Users/jakeholl/domains/saturn-pub/packages/mdb")
PY = "/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python"
attempt = ROOT / sys.argv[1]
attempt.mkdir(exist_ok=False)
cohort = attempt / "cohort"
sources = {"qualification/stage_qwen_row_sign_saved_audit.py": MDB / "qualification/stage_qwen_row_sign_saved_audit.py",
    "tests/test_stage_qwen_row_sign_saved_audit.py": MDB / "tests/test_stage_qwen_row_sign_saved_audit.py",
    "artifacts/collect_compact.py": ROOT.parent / "collect_compact.py",
    "artifacts/controller-qualification-v2/test_collect_compact.py": ROOT / "test_collect_compact.py"}
for name in ("stage_saved_flux_mask_pair_audit.py", "audit_qwen_row_sign.py", "audit_qwen_sign_shortcut.py",
    "audit_flux_raw_to_image.py", "audit_flux_functional_qwen_prefix.py", "audit_flux_functional_pass.py", "retrieve_reader_evidence.py"):
    sources["qualification/" + name] = MDB / "qualification" / name
for name, source in sources.items():
    target = cohort / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
command = [PY, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(attempt / "tmp"),
    "tests/test_stage_qwen_row_sign_saved_audit.py", "artifacts/controller-qualification-v2/test_collect_compact.py"]
started = time.time()
result = subprocess.run(command, cwd=cohort, capture_output=True, text=True, timeout=120)
(attempt / "stdout.txt").write_text(result.stdout)
(attempt / "stderr.txt").write_text(result.stderr)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(command=command, cwd=str(cohort), seconds=time.time() - started, returncode=result.returncode,
    source_hashes={name: sha(path) for name, path in sources.items()}, stdout_sha256=sha(attempt / "stdout.txt"),
    stderr_sha256=sha(attempt / "stderr.txt"), scope="Transport-only synthetic metadata, delegation, checksum/member guards and installed bootstrap-to-real-reader-worker publication with explicitly stubbed numerical audit. Actual installed Atlas core is qualified independently by reader owner. No services, stage, collection, neural calls or original arrays/weights.")
(attempt / "execution.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True))
print(result.stdout)
print(result.stderr)
raise SystemExit(result.returncode)
