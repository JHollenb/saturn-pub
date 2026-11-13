"""Reductions of a live run's values, read the way the ``print`` op reads them.

The browser names an address of the live run (``{kind: "occurrence", address}``). The server
reads the value with ``Run.read(address)`` and its label with ``Run.read_info(address)``: the same
engine reads ``print`` makes, so every port the run has computed resolves (the cursor's layer
from the port cache; an earlier layer by re-entering it from its retained boundary cut; a
boundary from its mark). Nothing is exported or stored: a live reduction writes nothing to the
project. The tensor never leaves the server; the reduction is ``serve.reduce.reduce_array``,
relabelled with the value's authority. Runs on the live session's worker thread.

Cost: a value outside the port cache (any layer but the cursor's pending one) is recomputed by
re-entering its layer, once for ``read`` and once for ``read_info``; the web page therefore asks
only for the rows it shows. An address that names another origin (``branch:...`` or
``@cut...``) is not a value of the live run: it goes through the project's stored occurrences
(``serve.reduce.resolve``, the recorded path), never through the live run.
"""
from __future__ import annotations

from .reduce import ReduceError, reduce_array
from .schema import Ev

VALUE_FIELDS = ("min", "max", "mean", "std", "absmax", "l2")


def names_origin(address: str) -> bool:
    """True when ``address`` names an origin other than the live run (``b:...``, ``@cut...``)."""
    base = address.strip().partition("[")[0]
    return base.startswith("@") or ":" in base


def read_live(session, spec: dict):
    """``(array, info)`` of the live value at ``spec["address"]``; ``ReduceError`` when the run
    has not computed it or the engine declines to label it."""
    import torch

    run = session.ctx.run
    if run is None:
        raise ReduceError("unresolved", "the live session has no run yet (start one)", 409)
    address = spec.get("address")
    if not isinstance(address, str) or not address.strip():
        raise ReduceError("bad_pointer", "an occurrence pointer names an address")
    address = address.strip()
    if "[" in address:
        raise ReduceError("bad_pointer", "a live reduction reads a whole value; select with the "
                          "reduction's params (slice, topk) instead of an address selector")
    try:
        value = run.read(address)
        provenance = run.read_info(address)
    except (ValueError, KeyError) as error:  # not computed yet, before the run, a bad address
        raise ReduceError("unresolved", f"cannot read {address!r}: {error}", 409) from error
    authority = provenance.get("authority")
    if authority is None or provenance.get("state") != "resolved":
        raise ReduceError(provenance.get("state") or "refused",
                          f"{address}: the engine declined to label the value "
                          f"({provenance.get('reason') or 'no reason given'})", 409)
    authority = {"derived": "derived-exact"}.get(authority, authority)  # as `print` labels it
    value = value.detach().cpu()
    array =value.to(torch.float32).numpy() if value.is_floating_point() else value.numpy()
    info = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "authority": authority}
    return array, info


def relabel(result, authority: str, note: str):
    """``result`` with its value statistics labelled ``authority`` (``reduce_array`` labels them
    ``measured``); counts and the derived histogram/tile domains keep their own labels."""
    if authority not in Ev.__args__:
        raise ReduceError("refused", f"the value's authority {authority!r} is not a record label",
                          409)
    data = result.model_dump(mode="json")
    data["note"] = note
    if authority != "measured":
        for key in VALUE_FIELDS:
            if isinstance(data.get(key), dict) and data[key]["ev"] == "measured":
                data[key]["ev"] = authority
        for entry in data.get("entries") or []:
            if entry["v"]["ev"] == "measured":
                entry["v"]["ev"] = authority
        if data.get("ev") == "measured":
            data["ev"] = authority
    return type(result).model_validate(data)


def reduce_live(session, spec: dict, op: str, params: dict):
    """One reduction of a live value (``project`` reads its direction from the live run too)."""
    array, info = read_live(session, spec)
    result = reduce_array(array, info, spec, op, params,
                          direction_of=lambda d: read_live(session, d)[0])
    return relabel(result, info["authority"],
                   f"live run at {session.where}, read as `print` reads it (Run.read)")
