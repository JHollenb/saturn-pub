"""Record kinds of the analysis layer, declared where every reader finds them.

``mdb.core.records`` loads this module, so a process that only reads a project (no torch, no
``mdb.analysis``) can verify ``forecast``, ``whatif``, ``circuit`` and ``symbol-table``
records. The analysis modules use these same constants; re-declaring an identical spec is a
no-op and a different one fails, so the two can never drift apart silently.
"""

from __future__ import annotations

from .records import declare_kind

# The prefill split a record's numbers were computed under (M26, M54): ``prefill`` prompt rows
# batched, ``first_row`` the run's first stepped row (its root), ``prompt_rows`` the prompt
# length. Optional in every kind below (records written before M54 lack it; their cut
# fingerprint alone binds the split); ``sweep`` requires it.
PROCEDURE = ("procedure.prefill", "procedure.first_row", "procedure.prompt_rows")


def procedure(run) -> dict:
    """The ``procedure`` field of a record about ``run``: its prefill split as plain data."""
    prefill = int(run.prefill)
    rows = len(run.tokens)
    first = int(getattr(run, "first_row", prefill))
    batched = "no row batched" if prefill == 0 else f"rows 0..{prefill - 1} batched"
    return {"prefill": prefill, "first_row": first, "prompt_rows": rows,
            "default": prefill == rows - 1,
            "text": f"prefill={prefill}: {batched}, first stepped row t{first}"}


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
                   "emulated.shape", "emulated.token", "emulated.rounding_origin.head",
                   *PROCEDURE),
)

WHATIF = declare_kind(
    "whatif",
    required=("schema", "edit", "consumer", "sealed_at", "mode", "prediction"),
    plain_numeric=("edit.row", "edit.layer", "edit.mask_count", "edit.shape", "consumer.row",
                   "consumer.index", "consumer.indices", "sealed_at.row", "sealed_at.index",
                   "sealed_at.transition", "operands.*.shape", "edit.cell.**",
                   "edit.origin.shape", "operands_summary.count", "operands_summary.listed",
                   "operands_summary.rest.count", "operands_summary.pointer.bytes", *PROCEDURE),
)

CIRCUIT = declare_kind(
    "circuit",
    required=("schema", "consumer", "where", "root", "closure"),
    plain_numeric=("k", "depth", "count", "edits.count", *PROCEDURE),
)

# ``circuit test ID ref=NAME``: a circuit patched natively from a reference run (or a prompt
# set's mean). Every number is labelled; the component list is a sha256-checked blob.
CIRCUIT_TEST = declare_kind(
    "circuit-test",
    required=("schema", "circuit", "consumer", "where", "reference", "components", "margins",
              "resolution", "ratios"),
    plain_numeric=PROCEDURE,
)

SYMBOL_TABLE = declare_kind(
    "symbol-table",
    required=("schema", "where", "consumer", "bindings", "unbound", "page"),
    plain_numeric=("page.offset", "page.limit", "page.total", "rows", "bindings.*.rows",
                   "bindings.*.token", "bindings.*.omitted", "bindings.*.count",
                   "accounting.heads", *PROCEDURE),
)

# ``symbol NAME = whatif|compare.a|EACH:ITEM`` (M23): a named symbol declared from a transplant
# result: the writer without its row (role), the source row (role-relative or absolute, with the
# prompt's and prefix's sha256), the readers as ``reads`` lists them, the consumer and the
# measured effect, the donor and the run. Labelled numbers and pointers only.
SYMBOL = declare_kind(
    "symbol",
    required=("schema", "name", "role", "row", "consumer", "evidence", "origin"),
    plain_numeric=("role.writer.layer", "role.readers.rows", "role.readers.count",
                   "role.readers.listed.*.row", "role.readers.listed.*.layer", "row.row",
                   "row.token", "row.prompt_rows", "consumer.row", "consumer.indices",
                   *PROCEDURE),
)

# ``sweep PORT rows= layers= = ORIGIN --consumer C``: the position x layer patching table (M35,
# M54). The table itself is a sha256-checked JSON blob (``table``, a ``core.bounded`` pointer);
# the record keeps the base, the top cells, the cost and the procedure of both runs.
SWEEP = declare_kind(
    "sweep",
    required=("schema", "port", "origin", "consumer", "stop", "rows", "layers", "base", "top",
              "table", "cost", "procedure"),
    plain_numeric=("top.*.row", "top.*.layer", "cells.count", *PROCEDURE,
                   *(p.replace("procedure", "origin_procedure") for p in PROCEDURE)),
)

__all__ = ["CIRCUIT", "CIRCUIT_TEST", "FORECAST", "PROCEDURE", "SWEEP", "SYMBOL", "SYMBOL_TABLE",
           "WHATIF", "procedure"]
