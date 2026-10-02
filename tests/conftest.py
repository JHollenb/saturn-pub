"""Every runtime regression runs with upstream packages forbidden."""
import importlib.abc
import sys


class NoUpstreamRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "saturn_pub", "common_harness"}:
            raise ImportError("standalone runtime attempted upstream import: " + fullname)
        return None


sys.meta_path.insert(0, NoUpstreamRuntime())
