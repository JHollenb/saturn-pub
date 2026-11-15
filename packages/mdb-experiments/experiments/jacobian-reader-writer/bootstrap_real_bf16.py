"""Verify the sealed payload and run the real BF16 worker from the shipped wheel."""

from __future__ import annotations

import hashlib
import importlib.abc
import json
import runpy
import stat
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHEEL_NAME = "mdb_runtime-0.1.0-py3-none-any.whl"
WHEEL_SHA256 = "1120c5299ffe096a76db84ceaa71cd44348db8f03a46e6bf17e72e433b96ea2e"
BLOCKED_IMPORTS = ("saturn", "saturn_pub", "common", "common_harness", "mrun")


class BlockWorkspaceImports(importlib.abc.MetaPathFinder):
    """Keep the remote data plane independent of workspace/upstream packages."""

    def find_spec(self, fullname: str, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in BLOCKED_IMPORTS):
            raise ImportError(f"workspace import blocked in sealed MDB worker: {fullname}")
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_payload() -> Path:
    manifest_path = HERE / "payload-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("payload manifest must be a non-empty object")
    for name, expected in manifest.items():
        relative = Path(name)
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or relative == Path(".")
            or ".." in relative.parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise RuntimeError("payload manifest contains an invalid entry")
        path = HERE / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"sealed payload identity mismatch: {name}")
    wheel = HERE / WHEEL_NAME
    if manifest.get(WHEEL_NAME) != WHEEL_SHA256 or sha256_file(wheel) != WHEEL_SHA256:
        raise RuntimeError("qualified MDB wheel identity mismatch")
    return wheel


def extract_wheel(wheel: Path) -> Path:
    """Hydrate the verified pure-Python wheel for runtime source-file consumers."""
    destination = HERE / "installed-mdb"
    if destination.exists():
        raise RuntimeError("lease-local MDB extraction target already exists")
    destination.mkdir()
    seen: set[str] = set()
    expanded = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            normalized = relative.as_posix().rstrip("/")
            if (
                relative.is_absolute()
                or relative == Path(".")
                or ".." in relative.parts
                or not normalized
                or normalized in seen
            ):
                raise RuntimeError("qualified wheel contains an unsafe or duplicate member")
            seen.add(normalized)
            mode = (member.external_attr >> 16) & 0o170000
            if member.is_dir():
                if mode not in {0, stat.S_IFDIR}:
                    raise RuntimeError("qualified wheel directory has an invalid file type")
                (destination / relative).mkdir(parents=True, exist_ok=True)
                continue
            if mode not in {0, stat.S_IFREG}:
                raise RuntimeError("qualified wheel contains a non-regular member")
            expanded += member.file_size
            if expanded > 64 << 20:
                raise RuntimeError("qualified wheel expands beyond 64 MiB")
            data = archive.read(member)
            if len(data) != member.file_size:
                raise RuntimeError("qualified wheel member is truncated")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return destination.resolve()


def main() -> None:
    wheel = verify_payload()
    installed = extract_wheel(wheel)
    sys.meta_path.insert(0, BlockWorkspaceImports())
    sys.path.insert(0, str(installed))
    import mdb

    installed_prefix = str(installed) + "/"
    if not str(Path(mdb.__file__).resolve()).startswith(installed_prefix):
        raise RuntimeError(f"MDB did not import from the qualified wheel: {mdb.__file__}")
    sys.argv = [
        "real_bf16_worker.py",
        "--output",
        "output",
        "--protocol",
        "protocol.json",
    ]
    runpy.run_path(str(HERE / "real_bf16_worker.py"), run_name="__main__")


if __name__ == "__main__":
    main()
