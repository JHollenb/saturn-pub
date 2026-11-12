"""Legal frontier reconstruction, typed state restoration and Euler forecasts."""
from __future__ import annotations

import itertools


def recipes():
    return [{"id": "L%dI%dT%d" % bits, "latent": bits[0], "image": bits[1], "text": bits[2]}
        for bits in itertools.product((0, 1), repeat=3)]


def project_frontier(adapter, cut):
    """Reconstitute the same-step project input from saved L/E/schedule.

    This is a derived initial closure, not a historical rewind receipt. Its
    twelve-step reproduction of the saved pre6 cut must be checked by caller.
    No late latent write is admitted and no adapter write rule is weakened.
    """
    from saturn_pub import Session

    if (cut.execution_point["logical_step"] != 3 or cut.execution_point["phase"] != "single"
            or cut.execution_point["index"] != 6):
        raise ValueError("requires declared step3 pre-single6 input")
    state = cut.payload
    state.update(phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
    session = Session(adapter, state, parent=cut.fingerprint)
    return session.capture()


def restore_slots(nav, replacements, *, label, store):
    from saturn_pub import Act
    from saturn_pub.values import describe

    rows = []
    for slot, value in replacements.items():
        before = nav.capture()
        act = Act.replace(slot, value, name=label + "." + slot)
        receipt = nav.apply(act)
        after = nav.capture()
        store.save(before)
        store.save(after)
        store.receipt(receipt)
        if describe(after.payload) != describe({**before.payload, slot: value}):
            raise ValueError("state restoration changed an undeclared slot")
        rows.append({"slot": slot, "parent": before.fingerprint, "post_act": after.fingerprint,
            "act": act.manifest(), "receipt": receipt.to_dict()})
    return rows


def predict_euler(latent, velocity, sigma_delta):
    """Native lane's BF16 scalar/product promotion, F32 sum, BF16 output."""
    import numpy as np

    if (latent.dtype != np.float32 or velocity.dtype != np.float32 or latent.shape != velocity.shape
            or latent.ndim != 3 or not np.isfinite(latent).all() or not np.isfinite(velocity).all()):
        raise ValueError("finite widened BF16 latent/velocity are required")
    def bf16(x):
        bits = np.asarray(x, np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)
    # TensorIterator promotes a zero-dimensional F32 dt to the dimensional
    # BF16 operand's common dtype before this multiplication.
    product = bf16(bf16(np.float32(sigma_delta)) * velocity)
    return bf16(latent + product)


def terminal_from_forecast(adapter, reference, latent, *, parent):
    """Create a declared terminal closure for VAE consumption, no model steps."""
    from saturn_pub import Session

    if reference.execution_point["logical_step"] != 4 or reference.execution_point["phase"] != "project":
        raise ValueError("forecast consumer requires complete reference closure")
    state = reference.payload
    state["latent"] = latent
    return Session(adapter, state, parent=parent)


def observe_live(nav):
    """Separate working I/T, scheduler-only velocity and disposed final state.

    The current public adapter omits noise_prediction from read addresses even
    though its declared surface includes it. Capture is the supported fallback.
    """
    from saturn_pub.values import describe

    point = nav.point
    live = point["logical_step"] < 4 and point["phase"] != "project"
    values = {slot: describe(nav.session.read(slot)) if live else describe(None) for slot in ("image", "text")}
    values["noise_prediction"] = describe(nav.capture().payload["noise_prediction"]) if point["phase"] == "scheduler" else describe(None)
    return values


__all__ = ["recipes", "project_frontier", "restore_slots", "predict_euler", "terminal_from_forecast", "observe_live"]
