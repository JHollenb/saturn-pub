"""MDB's lazy public API. Importing the package never loads a model backend."""
from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "BenchSession",
    "BenchError",
    "CausalContextVM",
    "MambaStateVM",
    "StateGraph",
    "GraphNode",
    "GraphEdge",
    "GraphIntegrityError",
    "GraphSnapshot",
    "GraphVerification",
    "NodeBinding",
    "Evidence",
    "StateGraphError",
    "ExecutionBinding",
    "GraphBinding",
    "SourceNode",
    "StateNode",
    "ActionNode",
    "OutputNode",
    "evidence",
]

_EXPORTS = {
    "BenchSession": ("bench", "BenchSession"),
    "BenchError": ("errors", "BenchError"),
    "CausalContextVM": ("causal_context_vm", "CausalContextVM"),
    "MambaStateVM": ("mamba_state_vm", "MambaStateVM"),
    "StateGraph": ("state_graph", "StateGraph"),
    "GraphNode": ("state_graph", "GraphNode"),
    "GraphEdge": ("state_graph", "GraphEdge"),
    "GraphIntegrityError": ("state_graph", "GraphIntegrityError"),
    "GraphSnapshot": ("state_graph", "GraphSnapshot"),
    "GraphVerification": ("state_graph", "GraphVerification"),
    "NodeBinding": ("state_graph", "NodeBinding"),
    "Evidence": ("state_graph", "Evidence"),
    "StateGraphError": ("state_graph", "StateGraphError"),
    "ExecutionBinding": ("state_graph", "ExecutionBinding"),
    "GraphBinding": ("state_graph", "GraphBinding"),
    "SourceNode": ("state_graph", "SourceNode"),
    "StateNode": ("state_graph", "StateNode"),
    "ActionNode": ("state_graph", "ActionNode"),
    "OutputNode": ("state_graph", "OutputNode"),
    "evidence": ("state_graph", "evidence"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), attribute)
    globals()[name] = value
    return value
