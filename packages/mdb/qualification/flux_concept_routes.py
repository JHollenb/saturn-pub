"""Factor stored conditioning from working text through public typed FLUX Acts.

These are whole-field carriers, not discovered object masks or named concepts.
All arms use two Acts and one native projection from the same recipient parent.
"""
from __future__ import annotations


def prepare_hypotheses(root, *, model_identity, input_identity, source_identity, cuts, source_sha256, config):
    """Use the public transaction contract for observed external CUDA frontiers."""
    from mdb.forecast_transaction import ForecastTransaction

    tx = ForecastTransaction(root, model_identity=model_identity, input_identity=input_identity,
        source_identity=source_identity,
        frontier={"kind": "architectural", "cuts": cuts, "source_sha256": source_sha256,
                  "allowed_prefix": "four observed native prefixes through step1 projection; no rendered outcomes",
                  "naming_policy": "structural function IDs; semantic aliases remain candidates"},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    try:
        tx.predict("route_hypotheses", lambda context: {"predictions": config["hypotheses"]})
        return tx, tx.seal()
    except BaseException:
        tx.close()
        raise


def branch_carriers(session, parent, projected, donor_projected, arm, *, name, on_receipt=None):
    from saturn_pub import Act

    if arm not in {"sham", "text", "conditioning", "both"}:
        raise ValueError("unknown carrier arm")
    point = parent.execution_point
    if point["phase"] != "project" or projected.execution_point["phase"] != "joint" or projected.execution_point["index"] != 0:
        raise ValueError("carrier branch requires pre/post projection cuts")
    if point["logical_step"] != projected.execution_point["logical_step"] or point["logical_step"] != donor_projected.execution_point["logical_step"]:
        raise ValueError("carrier donors must share the recipient timestep")
    source = donor_projected.payload
    recipient = projected.payload
    branch, receipts = session.fork(parent), []

    def apply(slot, value):
        act = Act.replace(slot, value, name=name + "." + slot)
        receipt = branch.apply(act)
        row = {"act": act.manifest(), "receipt": receipt.to_dict()}
        if on_receipt:
            row["receipt_id"] = on_receipt(receipt)
        receipts.append(row)

    apply("conditioning", source["conditioning"] if arm in {"conditioning", "both"} else recipient["conditioning"])
    receipt = branch.continue_(1)
    if on_receipt:
        on_receipt(receipt)
    apply("text", source["text"] if arm in {"text", "both"} else recipient["text"])
    # Preserve closure: context and image must be the recipient's projection.
    import torch

    actual = branch.capture(retain=False).payload
    expected = dict(recipient)
    expected["conditioning"] = source["conditioning"] if arm in {"conditioning", "both"} else recipient["conditioning"]
    expected["text"] = source["text"] if arm in {"text", "both"} else recipient["text"]

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, (tuple, list)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return type(a) is type(b) and a == b

    if not equal(actual, expected):
        raise ValueError("carrier intervention changed undeclared recipient state")
    return branch, receipts
