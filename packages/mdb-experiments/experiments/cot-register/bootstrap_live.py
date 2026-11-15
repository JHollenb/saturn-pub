"""Execute a content-bound standalone MDB wheel in an admitted resident worker."""

import hashlib
import importlib.abc
import json
import runpy
import sys
import zipfile
from pathlib import Path


class NoWorkspace(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {"saturn", "saturn_pub", "common", "common_harness", "mrun"}:
            raise ImportError("data plane attempted workspace import: " + fullname)
        return None


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "payload-manifest.json").read_text())
    for name, expected in inventory.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError("unsafe staged payload path")
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError("staged payload identity differs: " + name)
    wheel = root / "mdb_runtime-0.1.0-py3-none-any.whl"
    installed = root / "installed-mdb"
    installed.mkdir(exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError("unsafe wheel path")
        archive.extractall(installed)
    sys.path.insert(0, str(installed))
    sys.meta_path.insert(0, NoWorkspace())
    import mdb
    if not Path(mdb.__file__).resolve().is_relative_to(installed.resolve()):
        raise RuntimeError("MDB was not imported from the staged wheel")
    runpy.run_path(str(root / "live_trial.py"), run_name="__main__")


if __name__ == "__main__":
    main()
