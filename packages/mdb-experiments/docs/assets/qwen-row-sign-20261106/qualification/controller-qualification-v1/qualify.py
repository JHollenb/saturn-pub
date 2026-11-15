"""Snapshot controller sources outside checkout; run only model-free mechanics."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/Users/jakeholl/domains/saturn-pub/packages/mdb")
PY = "/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/bin/python"
OWNED = ["qualification/stage_qwen_row_sign.py", "qualification/stage_qwen_row_sign_saved_audit.py",
         "tests/test_stage_qwen_row_sign.py", "tests/test_stage_qwen_row_sign_saved_audit.py"]
DEPENDENCIES = ["stage_qwen_sign_shortcut.py", "stage_saved_flux_mask_pair_audit.py", "audit_qwen_row_sign.py",
    "audit_qwen_sign_shortcut.py", "audit_flux_raw_to_image.py", "audit_flux_functional_qwen_prefix.py",
    "audit_flux_functional_pass.py", "retrieve_reader_evidence.py", "qwen_row_sign_campaign.py",
    "qwen_row_sign_shortcut.py", "qwen_sign_shortcut_campaign.py", "qwen_sign_shortcut.py",
    "qwen_h18_state_history.py", "flux_functional_qwen_prefix.py", "flux_functional_qwen_prefix_worker.py",
    "flux_source_receiver.py", "flux_image_tail_microscope.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


attempt = ROOT / sys.argv[1]
attempt.mkdir(exist_ok=False)
cohort = attempt / "cohort"
hashes = {}
for name in [*OWNED, *("qualification/" + name for name in DEPENDENCIES)]:
    source, target = SOURCE / name, cohort / name
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = sha(source)
commands = [[PY, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(attempt / "tmp"),
    "tests/test_stage_qwen_row_sign.py", "tests/test_stage_qwen_row_sign_saved_audit.py"]]
if "--partial" in sys.argv:
    commands[0] = commands[0][:-2] + ["tests/test_stage_qwen_row_sign.py", "-k", "not exact_producer"]
elif "--native" in sys.argv:
    commands[0] = commands[0][:-2] + ["tests/test_stage_qwen_row_sign.py"]
elif "--saved" in sys.argv:
    commands[0] = commands[0][:-2] + ["tests/test_stage_qwen_row_sign_saved_audit.py"]
start = time.time()
result = subprocess.run(commands[0], cwd=cohort, capture_output=True, text=True, timeout=120)
(attempt / "stdout.txt").write_text(result.stdout)
(attempt / "stderr.txt").write_text(result.stderr)
receipt = dict(command=commands[0], cwd=str(cohort), returncode=result.returncode, seconds=time.time() - start,
    source_hashes=hashes, stdout_sha256=sha(attempt / "stdout.txt"), stderr_sha256=sha(attempt / "stderr.txt"),
    scope="Isolated installed-wheel synthetic metadata/CLI/bootstrap guards only; zero actual staging, jobs, services, original arrays, models or weights.")
runtime = subprocess.run([PY, "-I", "-B", "-c", "import json,sys,platform,hashlib;from mdb.payload import runtime_distributions,seal_package;print(json.dumps(dict(python=platform.python_version(),executable=sys.executable,packages={n:dict(files=len(seal_package(n)),sha256=hashlib.sha256(json.dumps(seal_package(n),sort_keys=True).encode()).hexdigest()) for n,_,_ in runtime_distributions(require_installed=True)}),sort_keys=True))"], capture_output=True, text=True, check=True)
receipt["installed_runtime"] = json.loads(runtime.stdout)
(attempt / "execution.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
print(json.dumps(receipt, sort_keys=True))
print(result.stdout)
print(result.stderr)
raise SystemExit(result.returncode)
