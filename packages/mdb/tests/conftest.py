"""Allow public Saturn-pub reuse while blocking private workspace packages."""
import importlib.abc
import sys


class NoPrivateRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "common", "common_harness"}:
            raise ImportError("MDB runtime attempted private workspace import: " + fullname)
        return None


sys.meta_path.insert(0, NoPrivateRuntime())


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path_factory, monkeypatch):
    """No test reads the developer's real config: the user file is looked up in an empty
    directory, ``MDB_CONFIG`` and every settings variable are unset, and the process settings
    are re-resolved per test."""
    from mdb.core import settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))
    monkeypatch.delenv("MDB_CONFIG", raising=False)
    for key in settings.KEYS.values():
        for name in key.env:
            monkeypatch.delenv(name, raising=False)
    settings.reset()
    yield
    settings.reset()
