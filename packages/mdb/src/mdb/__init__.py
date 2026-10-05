"""MDB's lazy public API. Importing the package never loads a model backend."""
from importlib import import_module

__version__ = "0.1.0"
__all__ = ["BenchSession", "BenchError", "CausalContextVM", "MambaStateVM"]

_EXPORTS = {
    "BenchSession": ("bench", "BenchSession"),
    "BenchError": ("errors", "BenchError"),
    "CausalContextVM": ("causal_context_vm", "CausalContextVM"),
    "MambaStateVM": ("mamba_state_vm", "MambaStateVM"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), attribute)
    globals()[name] = value
    return value
