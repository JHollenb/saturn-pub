"""Interop with external mechanistic-interpretability tooling.

Saturn does not reimplement these tools; it works next to them. The SAELens /
TransformerLens division of labor lives in ``examples/interop_saelens.py`` and
``docs/interop.md``. The flagship integration with Anthropic / Decode Research's
``circuit-tracer`` lives in :mod:`saturn_pub.interop.circuit_tracer`: it arbitrates each
attribution-graph edge against the native consumer. Importing this package does not import
torch or ``circuit_tracer``; both load on demand.
"""

from __future__ import annotations

from .circuit_tracer import (
    EdgeRow,
    EdgeVerdictTable,
    SyntheticAttributionGraph,
    build_synthetic_graph,
    classify_graph_vs_native,
    default_edge_rule,
    native_edge_test,
    native_group_intervention,
    select_edges,
    verify_edge_bundle,
)

__all__ = [
    "native_edge_test",
    "native_group_intervention",
    "classify_graph_vs_native",
    "verify_edge_bundle",
    "select_edges",
    "default_edge_rule",
    "EdgeRow",
    "EdgeVerdictTable",
    "SyntheticAttributionGraph",
    "build_synthetic_graph",
]
