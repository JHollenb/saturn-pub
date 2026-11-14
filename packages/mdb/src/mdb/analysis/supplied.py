"""Library-supplied forecasts and vector edits (advisor memo 2026-11-14 "showcase", B-c).

A library (nnsight ``grads`` x activation, a transcoder edge, a TransformerLens patch on a
processed bridge) nominates an edit and predicts its effect on a consumer. mdb seals that
prediction before the native arm runs, as a ``supplied-forecast`` record holding the library's
number (``supplied``, with the library, its version and the address map used), and beside it
mdb's own terms when computed (``exact``: the float64 chain; ``linear``: the first-order term).
``judge`` scores the record against the native change with mdb's floor (``verdict.spacing``,
``floor_spacings`` x the consumer dtype's spacing at the larger magnitude): the supplied number
never becomes a measurement.

``vector_edit`` is the supplied-vector edit at any cache cell or at the pending row's ports:
``Run.set`` / ``Run.set_cell`` already take tensors; this records the origin (library, address
map) with the edit so the manifest says where the vector came from.
"""

from __future__ import annotations

from typing import Any

from mdb.core.kinds import SUPPLIED_FORECAST, procedure
from mdb.core.records import seal_record

from .finite import EXACT, LOCAL, MEASURED, SUPPLIED, num
from .forecast import Refusal, run_session_id

SCHEMA = "mdb-supplied-forecast-v1"


def seal(run: Any, *, target: dict[str, Any], edit: dict[str, Any], value: float, library: str,
         version: str, address_map: dict[str, str], method: str, exact: float | None = None,
         linear: float | None = None, session_id: str | None = None) -> dict[str, Any]:
    """Seal one supplied forecast (returns the sealed record)."""
    if not library or not version:
        raise Refusal("a supplied forecast names its library and version")
    row, index = run.parse(run.where)
    body: dict[str, Any] = {
        "schema": SCHEMA, "target": dict(target), "edit": dict(edit),
        "supplied": num(float(value), SUPPLIED),
        "library": {"name": library, "version": version, "method": method},
        "address_map": dict(address_map),
        "sealed_at": {"where": run.where, "row": num(row, EXACT), "index": num(index, EXACT)},
        "procedure": procedure(run),
    }
    if exact is not None:
        body["exact"] = num(float(exact), EXACT)
    if linear is not None:
        body["linear"] = num(float(linear), LOCAL)
    return seal_record(session_id=session_id or run_session_id(run), kind="supplied-forecast",
                       model_identity=run.adapter.model_identity,
                       execution=dict(run.adapter.execution), authority=SUPPLIED, body=body)


def judge(record: dict[str, Any], native_delta: float, *, magnitude: float, dtype: Any,
          floor_spacings: float = 2.0, extra_floor: float = 0.0,
          kernel_path: str | None = None, calibration: Any = None,
          stable_floors: float | None = None) -> dict[str, Any]:
    """Score a sealed supplied forecast against the native change. ``magnitude`` is the
    consumer's size for the spacing (e.g. the larger |logit| of the margin's pair);
    ``extra_floor`` adds a measured kernel-path delta (batched vs stepped) to the floor.

    The floor is a noise bound, not a stability margin (SC6): ``stability`` marks the verdict
    ``stable`` only when ``|native| > k`` floors, else ``kernel-sensitive`` (``native_effect``
    and ``sign_agrees`` are unchanged). ``k`` comes from the model's ``calibration``
    (``stability.Calibration``, its JSON or a path), else ``stable_floors``, else
    ``stability.STABLE_FLOORS`` = 4; ``kernel_path`` names the realisation ``native_delta`` was
    measured on (``mdb-stepped``, ``batched``, or any declared string; ``undeclared`` if None)."""
    from .stability import stability
    from .verdict import spacing

    body = record["body"]
    supplied = float(body["supplied"]["value"])
    floor = floor_spacings * spacing(abs(magnitude), dtype) + abs(extra_floor)
    native = float(native_delta)
    out: dict[str, Any] = {
        "record": record.get("id"), "supplied": supplied, "native": num(native, MEASURED),
        "floor": num(floor, MEASURED), "error": num(native - supplied, MEASURED),
        "ratio": (num(supplied / native, MEASURED) if native else None),
        "native_effect": abs(native) > floor,
        "sign_agrees": (supplied > 0) == (native > 0) if abs(native) > floor else None,
    }
    out.update(stability(native, floor, kernel_path=kernel_path, calibration=calibration,
                         k=stable_floors))
    for key in ("exact", "linear"):
        if key in body:
            out[f"{key}_error"] = num(native - float(body[key]["value"]), MEASURED)
    return out


def vector_edit(run: Any, address: str, value: Any, *, library: str, version: str,
                address_map: dict[str, str], op: str = "replace", **cell: Any) -> dict[str, Any]:
    """Stage a library-supplied vector at ``address``: a pending-row port ``t<r>.L<l>.<port>``
    (``Run.set``) or a cache cell ``cache.L<l>.<k|v>`` with ``rows=``/``kv=`` (``Run.set_cell``);
    the origin names the library."""
    origin = {"kind": "supplied", "library": library, "version": version,
              "address_map": dict(address_map)}
    if address.startswith("cache."):
        _, layer, tensor = address.split(".")
        return run.set_cell(int(layer[1:]), tensor, value, op, origin=origin, **cell)
    return run.set(address, value, op, origin=origin)


__all__ = ["SCHEMA", "SUPPLIED_FORECAST", "judge", "seal", "vector_edit"]
