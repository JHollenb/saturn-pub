"""Allow public Saturn-pub reuse while blocking private workspace packages."""
import importlib.abc
import sys


class NoPrivateRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "common", "common_harness"}:
            raise ImportError("MDB runtime attempted private workspace import: " + fullname)
        return None


sys.meta_path.insert(0, NoPrivateRuntime())


from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

CHECKOUT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path_factory, monkeypatch):
    """No test reads the developer's real config: HOME and the user file point at empty
    directories, a project file found inside this checkout is ignored, ``MDB_CONFIG`` and every
    settings variable are unset, and the process settings are re-resolved per test."""
    from mdb.core import custody, settings

    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))
    real_find = settings.find_project_file

    def find_outside_checkout(start):
        """A developer's mdb.toml in this checkout must not leak into the tests."""
        found = real_find(start)
        if found is not None and CHECKOUT in found[0].resolve().parents:
            return None
        return found

    monkeypatch.setattr(settings, "find_project_file", find_outside_checkout)
    custody._STATE["sink_error"] = None
    monkeypatch.delenv("MDB_CONFIG", raising=False)
    for key in settings.KEYS.values():
        for name in key.env:
            monkeypatch.delenv(name, raising=False)
    settings.reset()
    yield
    settings.reset()
