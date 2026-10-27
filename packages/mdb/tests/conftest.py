"""Allow public Saturn-pub reuse while blocking private workspace packages."""
import importlib.abc
import sys


class NoPrivateRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "common", "common_harness"}:
            raise ImportError("MDB runtime attempted private workspace import: " + fullname)
        return None


sys.meta_path.insert(0, NoPrivateRuntime())
