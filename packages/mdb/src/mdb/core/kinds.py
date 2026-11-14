"""Record kinds of the analysis layer, declared where every reader finds them.

``mdb.core.records`` loads this module, so a process that only reads a project (no torch, no
``mdb.analysis``) can verify ``forecast``, ``whatif``, ``circuit`` and ``symbol-table``
records. The analysis modules use these same constants; re-declaring an identical spec is a
no-op and a different one fails, so the two can never drift apart silently.
"""

from __future__ import annotations

from .records import declare_kind

FORECAST = declare_kind(
    "forecast",
    required=("schema", "target", "sealed_at", "method", "operands", "predicted", "bound"),
    plain_numeric=("sealed_at.row", "sealed_at.index", "sealed_at.transition",
                   "sealed_at.receipts.count", "target.row", "target.index", "target.indices",
                   "operands.*.shape", "predicted.shape", "predicted.token",
                   "edits_at_seal.*.row", "edits_at_seal.*.layer",
                   "method.summary.count", "method.summary.global_steps.count",
                   "method.summary.proof_counts.*.count", "method.summary.layer_counts.*.layer",
                   "method.summary.layer_counts.*.count", "method.summary.chain_pointer.bytes",
                   "operands_summary.count", "operands_summary.listed",
                   "operands_summary.rest.count", "operands_summary.pointer.bytes",
                   "emulated.shape", "emulated.token", "emulated.rounding_origin.head"),
)

WHATIF = declare_kind(
    "whatif",
    required=("schema", "edit", "consumer", "sealed_at", "mode", "prediction"),
    plain_numeric=("edit.row", "edit.layer", "edit.mask_count", "edit.shape", "consumer.row",
                   "consumer.index", "consumer.indices", "sealed_at.row", "sealed_at.index",
                   "sealed_at.transition", "operands.*.shape", "edit.cell.**",
                   "edit.origin.shape", "operands_summary.count", "operands_summary.listed",
                   "operands_summary.rest.count", "operands_summary.pointer.bytes"),
)

CIRCUIT = declare_kind(
    "circuit",
    required=("schema", "consumer", "where", "root", "closure"),
    plain_numeric=("k", "depth", "count", "edits.count"),
)

# ``circuit test ID ref=NAME``: a circuit patched natively from a reference run (or a prompt
# set's mean). Every number is labelled; the component list is a sha256-checked blob.
CIRCUIT_TEST = declare_kind(
    "circuit-test",
    required=("schema", "circuit", "consumer", "where", "reference", "components", "margins",
              "resolution", "ratios"),
)

SYMBOL_TABLE = declare_kind(
    "symbol-table",
    required=("schema", "where", "consumer", "bindings", "unbound", "page"),
    plain_numeric=("page.offset", "page.limit", "page.total", "rows", "bindings.*.rows",
                   "bindings.*.token", "bindings.*.omitted", "bindings.*.count",
                   "accounting.heads"),
)

__all__ = ["CIRCUIT", "CIRCUIT_TEST", "FORECAST", "SYMBOL_TABLE", "WHATIF"]
