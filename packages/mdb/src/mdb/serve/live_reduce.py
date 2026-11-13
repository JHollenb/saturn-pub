"""Reductions of a live run's values: an occurrence pointer resolved through ``mdb.resolve``.

The browser names an address of the live run (``{kind: "occurrence", address}``). The server
asks the debugger for the occurrence pointer of that value (``Debugger.export``: the replay
recipe of the cursor, its cut, the value's digest and authority) and resolves it with
``mdb.resolve(pointer, session=debugger, tiers=("live",))``: the value comes from the live run's
port cache and is checked against the pointer's digest. The tensor never leaves the server; the
reduction is ``serve.reduce.reduce_array``, relabelled with the value's authority. Runs on the
live session's worker thread.
"""
from __future__ import annotations

from .reduce import ReduceError, reduce_array
from .schema import Ev

VALUE_FIELDS = ("min", "max", "mean", "std", "absmax", "l2")


def read_live(session, spec: dict):
    """``(array, info)`` of the live value at ``spec["address"]``; ``ReduceError`` when the run
    has not computed it or the pointer does not resolve."""
    import torch

    from mdb.core.errors import OpError
    from mdb.core.resolve import resolve

    run = session.ctx.run
    if run is None:
        raise ReduceError("unresolved", "the live session has no run yet (start one)", 409)
    address = spec.get("address")
    if not isinstance(address, str) or not address.strip():
        raise ReduceError("bad_pointer", "an occurrence pointer names an address")
    try:
        pointer = session.debugger.export(run, address.strip())
    except OpError as error:
        raise ReduceError("unresolved", error.message, 409) from error
    except ValueError as error:  # an address the run has not computed, a bad address
        raise ReduceError("unresolved", f"cannot read {address.strip()!r}: {error}",
                          409) from error
    found = resolve(pointer, session=session.debugger, tiers=("live",))
    if found.state != "resolved":
        raise ReduceError(found.state, found.reason or found.state, 409)
    value = found.value.detach().cpu()
    array = value.to(torch.float32).numpy() if value.is_floating_point() else value.numpy()
    info = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "authority": found.authority, "tier": found.tier, "sha256": found.sha256}
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
                   f"live run, {info['tier']} tier, value sha256 {info['sha256'][:12]}")
