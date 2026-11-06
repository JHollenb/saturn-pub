"""Build only the owned module over a clean commit and test its installed wheel."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / "saturn-pub"
BASE = "698fe49ebfe3b9d64078b48fa5f583a7b427e104"
OUT = ROOT / "installed-qualification-v1"
PINNED = {
    "src/mdb/computation_applicability.py": "820e971a6128aa7f49cd169734fd2a123e8a60924b25a6d08f6d0a46d33996f3",
    "tests/test_computation_applicability.py": "6a98faabbc2ff4142be18e0cc6ac9041d41b56232a8ce9586b4d3412a54fdbe9",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    commands = []
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    def run(label, command, cwd=ROOT, timeout=120):
        before = time.monotonic()
        result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout)
        for stream in ("stdout", "stderr"):
            (OUT / (label + "." + stream)).write_text(getattr(result, stream))
        commands.append(dict(label=label, command=command, cwd=str(cwd), returncode=result.returncode,
                             elapsed_s=time.monotonic() - before))
        (OUT / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(label + " failed: " + result.stderr[-1500:] + result.stdout[-1500:])
        return result.stdout

    archive = subprocess.check_output(["git", "archive", BASE, "packages/mdb"], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(OUT / "clean", filter="data")
    package = OUT / "clean/packages/mdb"
    for name, expected in PINNED.items():
        path = REPO / "packages/mdb" / name
        if sha(path) != expected:
            raise ValueError("Owned source changed: " + name)
        shutil.copyfile(path, package / name)
    source_manifest = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py")) if p.is_file()}
    (OUT / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    run("build", ["uv", "build", "--offline", "--wheel", "--out-dir", str(OUT / "wheels"), str(package)])
    venv = OUT / "venv"
    run("venv", ["uv", "venv", "--offline", "--python", "3.12", str(venv)])
    python = str(venv / "bin/python")
    wheel = next((OUT / "wheels").glob("mdb_runtime-*.whl"))
    saturn = ROOT.parent / "mdb-qwen-cached-qkv-20261106/run-01/payload/saturn_pub-0.4.1-py3-none-any.whl"
    run("install-runtime", ["uv", "pip", "install", "--offline", "--python", python, "--no-deps", str(wheel), str(saturn)])
    run("install-pytest", ["uv", "pip", "install", "--offline", "--python", python, "pytest"])
    cohort = OUT / "tests"
    cohort.mkdir()
    shutil.copyfile(package / "tests/test_computation_applicability.py", cohort / "test_computation_applicability.py")
    run("tests", [python, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                  "--basetemp", str(OUT / "test-tmp"), "test_computation_applicability.py"], cwd=cohort)
    inspection = run("inspect", [python, "-I", "-B", "-c",
        "import hashlib,json,sys; from pathlib import Path; import mdb.computation_applicability as m; p=Path(m.__file__).resolve(); "
        "assert 'site-packages' in p.parts; assert 'torch' not in sys.modules and 'numpy' not in sys.modules; "
        "print(json.dumps(dict(module=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),torch_imported=False,numpy_imported=False)))"])
    report = dict(schema="mdb-source-applicability-installed-qualification-v1", base_commit=BASE,
        owned_overlays=PINNED, wheel=dict(path=str(wheel), bytes=wheel.stat().st_size, sha256=sha(wheel)),
        saturn_wheel_sha256=sha(saturn), installed=json.loads(inspection), python=python,
        tests_stdout_sha256=sha(OUT / "tests.stdout"), commands_sha256=sha(OUT / "commands.json"),
        source_manifest_sha256=sha(OUT / "source-manifest.json"), peer_uncommitted_sources_included=False,
        model_calls=0, scope="Installed pure metadata API mechanics; no original scientific prediction.")
    (OUT / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    print((OUT / "tests.stdout").read_text())


if __name__ == "__main__":
    main()
