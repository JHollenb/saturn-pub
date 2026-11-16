"""Gate-side attribution of gated MLP units (binary ground truth 2026-11-16, run 6).

On the binary's gated-MLP dispatch, contribution attribution on ``f = gate * up`` named the
right unit or nothing; it named nothing on 3.7% of lane-steps, all of them steps where the
selected unit's output was zero (a ``mov`` of 0): the selection lived only in the gate (MEASURED,
RUN6.md (a)). A unit can be *selected but zero-valued*: its gate is on, and its value channel
carries ~nothing, so it contributes ~0 and contribution ranking never lists it.

``selected_zero(gate, contribution)`` marks those units: ``|gate| >= gate_floor`` and
``|contribution| <= zero_tol`` (absolute thresholds; ``relative`` scales them by the row's
largest ``|gate|`` and ``|contribution|``). For a SwiGLU layer the gate activation is
``sg = silu(g)`` and the contribution is the unit's direct share of the consumer through
``u``: ``rho^F c^T W_down[:, i] sg_i u_i`` (``swiglu_units``, ``derived-exact`` on the native
captures). The x86 dispatch uses the same rule with its one-hot gate
(``machines.x86.mlp.attribute``).
"""

from __future__ import annotations

from typing import Any

from .finite import EXACT, MEASURED, SUPPLIED, f64, num

GATE_FLOOR = 0.25    # relative: |sg_i| >= 0.25 max_j |sg_j| (supplied)
ZERO_TOL = 0.01      # relative: |contribution_i| <= 0.01 max_j |contribution_j| (supplied)


def _torch():
    import torch

    return torch


def selected_zero(gate: Any, contribution: Any, *, gate_floor: float, zero_tol: float,
                  relative: bool = False) -> Any:
    """Bool mask of units with an active gate and a ~zero contribution (last axis = units)."""
    torch = _torch()
    gate, contribution = torch.as_tensor(gate).abs(), torch.as_tensor(contribution).abs()
    if relative:
        gate_floor = gate_floor * gate.amax(-1, keepdim=True)
        zero_tol = zero_tol * contribution.amax(-1, keepdim=True)
    return (gate >= gate_floor) & (gate > 0) & (contribution <= zero_tol)


def swiglu_units(caps: Any, rho: Any, c: Any, *, top: int = 8, gate_floor: float = GATE_FLOOR,
                 zero_tol: float = ZERO_TOL) -> dict[str, Any]:
    """Per layer: the units' direct contributions to the explained margin and the selected but
    zero-valued units. ``caps`` a ``RowCapture`` (ports ``sg``, ``u``), ``rho``/``c`` the final
    norm scale and readout of ``explain_margin``. Lists the ``top`` units by |contribution| and
    the ``top`` selected-zero units by |sg| over all layers, with per-layer counts."""
    torch = _torch()
    rows, zero_rows, per_layer = [], [], []
    for layer, ports in enumerate(caps.ports):
        if "sg" not in ports or "u" not in ports:
            per_layer.append({"layer": layer, "state": "refused", "reason": "no sg/u capture"})
            continue
        adapter = caps.adapter
        down = getattr(adapter.backbone, adapter.spec.layers)[layer].mlp.down_proj.weight
        sg, u = f64(ports["sg"]).reshape(-1), f64(ports["u"]).reshape(-1)
        share = float(rho) * (f64(c) @ f64(down)) * sg * u
        mask = selected_zero(sg, share, gate_floor=gate_floor, zero_tol=zero_tol, relative=True)
        on = sg.abs() >= gate_floor * sg.abs().max()
        per_layer.append({"layer": layer, "selected": int(on.sum()),
                          "selected_zero": int(mask.sum()),
                          "sum": num(float(share.sum()), EXACT, "P19")})
        for i in share.abs().topk(min(top, share.numel())).indices.tolist():
            rows.append((abs(float(share[i])), layer, i, float(share[i]), float(sg[i]),
                         float(u[i])))
        for i in torch.nonzero(mask).flatten().tolist():
            zero_rows.append((abs(float(sg[i])), layer, i, float(share[i]), float(sg[i]),
                              float(u[i])))

    def listed(found: list[tuple]) -> list[dict[str, Any]]:
        return [{"unit": f"L{layer}.f[i={i}]", "contribution": num(share, EXACT, "P19"),
                 "sg": num(sg, MEASURED), "u": num(u, MEASURED)}
                for _, layer, i, share, sg, u in sorted(found, reverse=True)[:top]]

    return {"rule": f"selected = |sg_i| >= {gate_floor:g} x max |sg| of the layer; zero-valued = "
                    f"|contribution_i| <= {zero_tol:g} x max |contribution| of the layer; "
                    "contribution_i = rho^F c^T W_down[:, i] sg_i u_i",
            "gate_floor": num(gate_floor, SUPPLIED), "zero_tol": num(zero_tol, SUPPLIED),
            "top_contribution": listed(rows), "selected_zero": listed(zero_rows),
            "per_layer": per_layer}


__all__ = ["GATE_FLOOR", "ZERO_TOL", "selected_zero", "swiglu_units"]
