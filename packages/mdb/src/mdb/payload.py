"""Seal installed package bytes and ship wheels, never sibling checkouts."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import shutil
import zipfile
from pathlib import Path


def package_files(package):
    module = importlib.import_module(package)
    root = Path(module.__file__).resolve().parent
    return {path.relative_to(root).as_posix(): path for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}}


def seal_package(package):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in package_files(package).items()}


def build_runtime_wheel(staging, *, package="mdb", distribution="mdb-runtime", version="0.1.0"):
    """Build a deterministic dependency-free payload wheel from installed bytes.

    The enclosing launcher resolves execution dependencies explicitly. The
    normal distribution built by hatchling retains its optional dependencies.
    """
    name = distribution.replace("-", "_")
    target = Path(staging) / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    files = {package + "/" + relative: path.read_bytes() for relative, path in package_files(package).items()}
    # Sealed execution wheels are redistributed package bytes. Preserve the
    # installed distribution's license notices alongside those bytes.
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    for record in (installed.files or ()) if installed is not None else ():
        if "/licenses/" in str(record):
            files[info + "/licenses/" + record.name] = installed.locate_file(record).read_bytes()
    files[info + "/METADATA"] = (f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\nRequires-Python: >=3.10\n").encode()
    files[info + "/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: mdb-payload\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for relative, raw in sorted(files.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        writer.writerow([relative, "sha256=" + encoded, len(raw)])
    writer.writerow([info + "/RECORD", "", ""])
    files[info + "/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, raw in sorted(files.items()):
            member = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, raw)
    return target


def runtime_distributions(*, require_installed=False):
    """Resolve only explicitly installed public distributions."""
    try:
        importlib.metadata.version("mrun")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("MDB requires mrun-pub without the conflicting private mrun distribution")
    def version(distribution, fallback="0.1.0"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            if require_installed:
                raise RuntimeError("worker staging requires the installed " + distribution) from None
            # The dependency-free toolkit can describe an inert worker command.
            # Actual staging requires installed bytes and distribution metadata.
            return fallback
    return (("mdb", "mdb-runtime", version("mdb-runtime")),
            ("saturn_pub", "saturn-pub", version("saturn-pub", "0.4.1")),
            ("mrun", "mrun-pub", version("mrun-pub")))


def runtime_wheel_names():
    return tuple(f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
                 for _, distribution, version in runtime_distributions())


def stage_worker(staging, *, worker, config, request, config_bytes=None, additional_workers=(), additional_files=(), **kwargs):
    if kwargs:
        raise ValueError("MDB payloads do not support workspace source staging: " + ", ".join(kwargs))
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    raw = config_bytes if config_bytes is not None else (json.dumps(config, sort_keys=True) + "\n").encode()
    if json.loads(raw) != dict(config):
        raise ValueError("configuration bytes do not describe the supplied configuration")
    expected = request.get("config_bytes_sha256")
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("sealed configuration bytes changed")
    sources = [(Path(worker), Path(worker).name)] + [(Path(p), Path(p).name) for p in additional_workers]
    sources += [(Path(p), str(relative)) for p, relative in additional_files]
    reserved = {"saturn_run_config.json", "saturn_debug_request.json", "payload-seal.json",
                *runtime_wheel_names()}
    from .job_support import reject_staged_target_collisions
    sources = reject_staged_target_collisions(staging, sources, reserved_targets=reserved)
    for source, target in sources:
        output = staging / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
    wheels = [build_runtime_wheel(staging, package=package, distribution=distribution, version=version)
              for package, distribution, version in runtime_distributions(require_installed=True)]
    seal = {"packages": {name: seal_package(name) for name, _, _ in runtime_distributions(require_installed=True)},
            "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "files": {target.as_posix(): hashlib.sha256((staging / target).read_bytes()).hexdigest()
                      for _, target in sources},
            "wheels": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels},
            "worker": Path(worker).name, "worker_sha256": hashlib.sha256(Path(worker).read_bytes()).hexdigest()}
    (staging / "saturn_run_config.json").write_bytes(raw)
    (staging / "saturn_debug_request.json").write_text(json.dumps(request, sort_keys=True) + "\n")
    (staging / "payload-seal.json").write_text(json.dumps(seal, sort_keys=True) + "\n")


def worker_command(worker_name, *, include_observability_extras=False):
    if include_observability_extras:
        raise ValueError("MDB worker payloads require explicit observability dependencies")
    from .execution_environment import intent
    environment = intent()
    command = ["/usr/bin/env", "PYTHONUNBUFFERED=1", "PYTHONNOUSERSITE=1", "PYTHONPATH=",
        "UV_LINK_MODE=" + environment["uv_link_mode"],
        "UV_CACHE_DIR=" + environment["uv_cache_dir"], "TORCH_HOME=" + environment["torch_home"],
        "LLM_MODELS_EXTRA_ROOT=" + environment["extra_model_roots"],
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "DIFFUSERS_OFFLINE=1", "TOKENIZERS_PARALLELISM=false",
        "SATURN_RESIDENT_WORKER=1", "SATURN_RUN_CONFIG=saturn_run_config.json", "SATURN_DEBUG_REQUEST=saturn_debug_request.json",
        "uv", "run", "--no-project", "--offline", "--python", environment["python"]]
    for requirement in (*(f"./{name}" for name in runtime_wheel_names()), *environment["requirements"]):
        command.extend(["--with", requirement])
    command.extend(["python", "-u", "-m", "mdb.worker_bootstrap", worker_name])
    return command
