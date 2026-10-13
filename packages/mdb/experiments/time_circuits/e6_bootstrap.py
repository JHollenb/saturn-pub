"""Load sealed payload wheels without modifying the shared tracer environment."""
import importlib.abc
import json
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
seal = json.loads((root / "payload-seal.json").read_text())
site = root / "sealed-packages"
site.mkdir(exist_ok=False)
for name, expected in seal["wheels"].items():
    wheel = root / name
    if sha256(wheel.read_bytes()).hexdigest() != expected:
        raise RuntimeError("sealed research wheel changed")
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise RuntimeError("unsafe wheel member")
        archive.extractall(site)
sys.path.insert(0, str(site))


class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "saturn_pub", "common_harness"}:
            raise ImportError("research worker attempted an upstream runtime import")
        return None


sys.meta_path.insert(0, NoUpstream())
from mdb.worker_bootstrap import main

sys.argv = ["mdb.worker_bootstrap", "worker_e6.py"]
main()
