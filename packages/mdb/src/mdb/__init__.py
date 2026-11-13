"""MDB's lazy public API. Importing the package never loads a model backend.

The engine's own names (``attach``, ``open_tiny``, ``Debugger``) are exported here. Any other
public name (``BenchSession``, ``Project``, ``StateGraph``, ...) belongs to the experiment lanes
and is resolved on first use through the ``mdb.lanes.exports`` entry-point group
(``mdb.core.plugins``); without the ``mdb-lanes`` plug-in such a name raises ``AttributeError``
saying so.
"""
from importlib import import_module

__version__ = "0.1.0"
__all__ = ["attach", "open_tiny", "Debugger", "trace", "resolve"]

_EXPORTS = {
    "attach": ("ops.debugger", "attach"),
    "open_tiny": ("ops.debugger", "open_tiny"),
    "Debugger": ("ops.debugger", "Debugger"),
    "trace": ("engine.trace", "trace"),
    "resolve": ("core.resolve", "resolve"),
}


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module("." + module, __name__), attribute)
    elif name.startswith("__"):
        raise AttributeError(name)
    else:
        from .core import plugins

        found, value = plugins.export(name)
        if not found:
            if not plugins.installed():
                raise AttributeError(
                    f"module 'mdb' has no attribute {name!r}; lane APIs (BenchSession, Project, "
                    f"StateGraph, ...) come from the {plugins.PLUGIN_DIST} plug-in, which is not "
                    "installed (no 'mdb.lanes.exports' entry points found)")
            raise AttributeError(f"module 'mdb' has no attribute {name!r}")
    globals()[name] = value
    return value
