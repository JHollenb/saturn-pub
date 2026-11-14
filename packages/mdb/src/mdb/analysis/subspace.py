"""Subspace bindings: a named subspace of one value cell (DAS review 2026-11-14, entry 11b).

A ``SubspaceBinding`` is a sub-cell binding like ``components.ComponentBinding``: it holds no
cells of its own (the partition of value cells is unchanged) and names a ``k``-dimensional
subspace of the value cell ``cache.L<layer>.v[kv<g>,row=r]`` (``d_h`` coordinates) by an
orthonormal basis ``Q`` (``d_h, k``, float64), with its provenance and the authority of the basis
itself (``BASIS_AUTHORITIES``):

* ``derived-exact``: a deterministic function of the weights and a declared token set, e.g. the
  span of the symbol table's content readout directions mapped into the cell
  (``readout_subspace``: ``W_g^T (gamma_F * W_U[t])`` per token ``t``, P06/P07/P19, PCA top ``k``);
* ``fitted``: learnt from data, e.g. a pyvene ``LowRankRotatedSpaceIntervention`` trained by DAS
  (``from_pyvene``) or class means of cell values (``class_mean_subspace``); the provenance
  must name the data and the training (items digest, steps, seed, job);
* ``supplied``: the caller's basis, taken as given.

Numbers computed FROM a basis are labelled with the record vocabulary (``finite.AUTHORITIES``):
``mass`` (the share of a cell value's energy, or of a value difference's, in the subspace) and
the direct consumer share at fixed attention are ``derived-exact``; an interchange result
(``interchange``: the cell set to ``v + Q Q^T (v_src - v)`` on an mdb branch and run to the
consumer) is ``measured``. ``iia`` summarises interchange records with the showcase floor rule
(flip: ``M0 > floor`` and ``M_after < -m * floor``, ``m`` = 1 and the audit's 4-floor margin).
Measured IIA can be attached to the binding as labels (``attach``), so the symbol table
(``symbols.table(subspaces=...)``) reports per binding: basis, authority, provenance, the mass of
the native cell at the cursor and the attached IIA.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .finite import EXACT, MEASURED, f64, num
from .forecast import Refusal
from .symbols import Binding

BASIS_AUTHORITIES = ("fitted", "derived-exact", "derived-local", "supplied")
SCHEMA = "mdb-subspace-binding-v1"


def _torch():
    import torch

    return torch


def orthonormal(basis: Any, tol: float = 1e-9) -> Any:
    """An orthonormal float64 basis of the column span of ``basis`` (QR, rank-revealing by the
    diagonal of R); ``Refusal`` when the span is empty."""
    torch = _torch()
    m = f64(basis)
    if m.dim() == 1:
        m = m[:, None]
    q, r = torch.linalg.qr(m)
    d = r.diagonal().abs()
    if d.numel() == 0 or float(d.max()) == 0.0:
        raise Refusal("a subspace binding needs a non-empty span")
    keep = d > tol * d.max()
    return q[:, keep].contiguous()


def principal_angles(a: Any, b: Any) -> list[float]:
    """Principal angles in degrees (ascending) between the spans of two orthonormal bases."""
    torch = _torch()
    s = torch.linalg.svdvals(f64(a).T @ f64(b)).clamp(-1.0, 1.0)
    return [float(x) for x in torch.rad2deg(torch.arccos(s))]


@dataclass
class SubspaceBinding(Binding):
    """A subspace of the value cell ``cache.L<layer>.v[kv<kv>]`` (module docstring). ``row`` is
    the bound row (``None``: a row-free template the caller applies to any row, e.g. a carrier
    rule's row)."""

    layer: int = -1
    kv: int = -1
    row: int | None = None
    basis: Any = None
    authority: str = "supplied"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    @property
    def cell(self) -> str:
        where = "" if self.row is None else f",row={self.row}"
        return f"cache.L{self.layer}.v[kv{self.kv}{where}]"

    def digest(self) -> str:
        return hashlib.sha256(f64(self.basis).contiguous().numpy().tobytes()).hexdigest()

    def project(self, value: Any) -> Any:
        q = self.basis
        return q @ (q.T @ f64(value).reshape(-1))

    def swap(self, base: Any, source: Any) -> Any:
        """``base + Q Q^T (source - base)``: the subspace interchange (float64)."""
        b = f64(base).reshape(-1)
        return b + self.project(f64(source).reshape(-1) - b)

    def mass(self, value: Any, reference: Any = None) -> dict[str, Any]:
        """The share of ``|value - reference|^2`` inside the subspace (derived-exact given the
        basis); ``reference`` defaults to zero (the raw value)."""
        x = f64(value).reshape(-1)
        if reference is not None:
            x = x - f64(reference).reshape(-1)
        total = float(x @ x)
        inside = float((self.basis.T @ x).pow(2).sum())
        return num(inside / total if total else 0.0, EXACT)

    def angles(self, other: Any) -> list[float]:
        return principal_angles(self.basis, other.basis if isinstance(other, SubspaceBinding) else other)

    def at(self, row: int) -> list[Any]:                              # no cells of its own
        return []

    def bound(self, row: int) -> "SubspaceBinding":
        """This template bound to ``row`` (same basis and provenance)."""
        return SubspaceBinding(self.name, "subspace", [], [], layer=self.layer, kv=self.kv, row=row,
                               basis=self.basis, authority=self.authority,
                               provenance=dict(self.provenance), labels=dict(self.labels))

    def attach(self, name: str, value: Any, authority: str = MEASURED) -> None:
        """Attach a result (e.g. ``iia``) as a label ``(value, authority)``."""
        self.labels[name] = (value, authority)

    def decompose(self, run: Any) -> Any:
        """This bound subspace as a ``components.ComponentBinding`` of the native cell at the cursor:
        parts ``subspace`` (``Q Q^T v``) and ``complement`` (``v - Q Q^T v``, the residual), with
        ``linear = native = v`` (no residual-stream image). One decomposition family: the result
        feeds ``components.component_table`` / ``symbols.table(components=...)`` unchanged; its
        ``basis`` carries this binding's description (authority, provenance, digest)."""
        from .components import Component, ComponentBinding

        if self.row is None:
            raise Refusal(f"subspace {self.name} is a template: bind it to a row first (bound(row))")
        v = f64(run.read_cache(self.layer, "v", staged=False)[0, self.kv, self.row]).cpu()
        inside = self.project(v)
        parts = [Component("subspace", None, None, None, inside),
                 Component("complement", None, None, None, v - inside)]
        return ComponentBinding(self.name, "component", [], [], row=self.row, layer=self.layer,
                                kv=self.kv, basis={**self.describe(), "kind": "subspace"},
                                components=parts, native=v, linear=v,
                                provenance={"decomposition": "subspace", "basis_authority": self.authority,
                                            **self.provenance})

    def describe(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "name": self.name, "cell": self.cell, "k": self.k,
                "authority": self.authority, "provenance": self.provenance,
                "digest": self.digest(),
                "labels": {k: {"value": v[0], "authority": v[1]} for k, v in self.labels.items()}}

    def to_json(self) -> dict[str, Any]:
        return {**self.describe(), "layer": self.layer, "kv": self.kv, "row": self.row,
                "basis": f64(self.basis).tolist()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SubspaceBinding":
        torch = _torch()
        sb = subspace_binding(data["name"], data["layer"], data["kv"],
                              torch.tensor(data["basis"], dtype=torch.float64),
                              authority=data["authority"], provenance=data["provenance"],
                              row=data.get("row"), orthonormalize=False)
        if sb.digest() != data["digest"]:
            raise Refusal(f"subspace {data['name']}: basis digest mismatch")
        for key, val in data.get("labels", {}).items():
            sb.labels[key] = (val["value"], val["authority"])
        return sb


def subspace_binding(name: str, layer: int, kv: int, basis: Any, *, authority: str = "supplied",
                     provenance: dict[str, Any] | None = None, row: int | None = None,
                     orthonormalize: bool = True) -> SubspaceBinding:
    """A subspace binding over ``basis`` (``d_h, k``; orthonormalised unless told it already
    is, which is then checked to 1e-6)."""
    if authority not in BASIS_AUTHORITIES:
        raise Refusal(f"basis authority must be one of {', '.join(BASIS_AUTHORITIES)}")
    if authority == "fitted" and not provenance:
        raise Refusal("a fitted subspace needs its provenance (data, training, job)")
    torch = _torch()
    q = orthonormal(basis) if orthonormalize else f64(basis)
    err = float((q.T @ q - torch.eye(q.shape[1], dtype=torch.float64)).abs().max())
    if err > 1e-6:
        raise Refusal(f"subspace {name}: basis not orthonormal (max |Q^T Q - I| = {err:.2e})")
    where = f"cache.L{layer}.v[kv{kv}" + ("" if row is None else f",row={row}") + "]"
    return SubspaceBinding(name, "subspace", [], [where], layer=layer, kv=kv, row=row, basis=q,
                           authority=authority, provenance=dict(provenance or {}))


def readout_directions(adapter: Any, layer: int, kv: int, tokens: list[int]) -> Any:
    """``W_g^T (gamma_F * W_U[t])`` per token (n, d_h) float64: the symbol table's content readout
    (``symbols.content``, P19 at unit final rho) of a KV-group write, mapped into the cell
    (``W_g`` = the group's summed ``W_O`` blocks, ``circuit_map.group_writers``)."""
    torch = _torch()
    from .circuit_map import group_writers
    from .reads import layer_module

    final = getattr(adapter.backbone, adapter.spec.final_norm)
    gamma = f64(final.weight).reshape(-1)
    w_u = adapter.model.lm_head.weight
    rows = f64(w_u.index_select(0, torch.tensor(list(tokens), dtype=torch.long,
                                                 device=w_u.device))).cpu() * gamma.cpu()
    w_g = group_writers(layer_module(adapter, layer).self_attn.o_proj.weight, adapter.num_heads,
                        adapter.num_kv_heads)[kv].cpu()
    return rows @ w_g


def _pca(rows: Any, k: int, center: bool) -> tuple[Any, Any]:
    torch = _torch()
    x = f64(rows)
    if center:
        x = x - x.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(x, full_matrices=False)
    if k > vt.shape[0]:
        raise Refusal(f"k={k} exceeds the {vt.shape[0]} directions available")
    return vt[:k].T.contiguous(), s


def readout_subspace(adapter: Any, name: str, layer: int, kv: int, tokens: list[int], k: int, *,
                     set_name: str = "tokens", center: bool = True) -> SubspaceBinding:
    """The top-``k`` principal subspace of the cell-space readout directions of ``tokens``
    (derived-exact: weights + the declared token set)."""
    toks = sorted(dict.fromkeys(int(t) for t in tokens))
    q, s = _pca(readout_directions(adapter, layer, kv, toks), k, center)
    prov = {"method": "PCA of W_g^T (gamma_F * W_U[t]) over a declared token set",
            "set": set_name, "tokens_sha256": hashlib.sha256(json.dumps(toks).encode()).hexdigest(),
            "n_tokens": len(toks), "centered": center,
            "explained": float((s[:k] ** 2).sum() / (s ** 2).sum())}
    return subspace_binding(name, layer, kv, q, authority=EXACT, provenance=prov,
                            orthonormalize=False)


def class_mean_subspace(name: str, layer: int, kv: int, values: dict[str, list[Any]], k: int, *,
                        provenance: dict[str, Any]) -> SubspaceBinding:
    """The top-``k`` principal subspace of the centred class means of cell values (label ->
    values): the supervised mean-difference baseline (``fitted``)."""
    torch = _torch()
    means = torch.stack([torch.stack([f64(v).reshape(-1) for v in vs]).mean(0) for vs in values.values()])
    q, s = _pca(means, k, True)
    prov = {"method": "PCA of centred class means of cell values", "classes": sorted(values),
            "explained": float((s[:k] ** 2).sum() / (s ** 2).sum()), **provenance}
    return subspace_binding(name, layer, kv, q, authority="fitted", provenance=prov,
                            orthonormalize=False)


def from_pyvene(intervention: Any, name: str, layer: int, kv: int, *,
                provenance: dict[str, Any]) -> SubspaceBinding:
    """The subspace of a trained pyvene ``LowRankRotatedSpaceIntervention`` (``rotate_layer.
    weight``, ``(d_h, k)`` with orthonormal columns under its orthogonal parametrisation): its
    interchange ``base + (R s - R b) R^T`` is ``swap`` of this binding (``fitted``)."""
    weight = intervention.rotate_layer.weight.detach()
    return subspace_binding(name, layer, kv, weight, authority="fitted",
                            provenance={"tool": "pyvene", "intervention": type(intervention).__name__,
                                        **provenance}, orthonormalize=False)


def interchange(debugger: Any, sb: SubspaceBinding, ids: list[int], row: int, source_value: Any,
                consumer: Any, floor: float | None = None, keep: bool = False) -> dict[str, Any]:
    """Measure the subspace interchange on mdb branches (``branches.Branches``): a run of ``ids``
    with prefill = ``row`` branched at ``t<row>.L<layer>.h``; arms ``M`` (the cell's value
    projection set to ``swap(native, source_value)``) and ``M_whole`` (set to ``source_value``),
    each run to the consumer (``forecast.Margin``/``Logit``, or a callable of the last row's
    logits), against the base ``M0`` and ONE floor shared with every other arm of the branch
    point: the caller's ``floor``, else the branches' default (2 spacings at the base margin)
    for a margin or logit. ``branches`` in the result is the ``Branches`` object (seal it, add
    nulls, compare) when ``keep``; ``arms`` is its plain table (JSON-able) always."""
    from .branches import Branches, edit

    torch = _torch()
    adapter = debugger.adapter
    dh = adapter.head_dim
    last = len(ids) - 1
    run = debugger.run(list(ids), name="subspace", prefill=row)
    br = Branches(run, consumer, at=f"t{row}.L{sb.layer}.h", row=last, floor=floor,
                  name=f"interchange:{sb.name}")
    native = f64(br.reference.read_cache(sb.layer, "v", staged=False)[0, sb.kv, row]).cpu()
    shape = run.port_shape("v")
    mask = torch.zeros(shape, dtype=torch.bool)
    mask.reshape(-1)[sb.kv * dh:(sb.kv + 1) * dh] = True
    for key, value in (("M", sb.swap(native, source_value)), ("M_whole", f64(source_value))):
        full = torch.zeros(shape, dtype=adapter.dtype)
        full.reshape(-1)[sb.kv * dh:(sb.kv + 1) * dh] = value.reshape(-1).to(adapter.dtype)
        br.add(key, [edit(f"t{row}.L{sb.layer}.v", full, mask=mask,
                          origin={"kind": "subspace-interchange", "subspace": sb.name,
                                  "digest": sb.digest(), "arm": key})])
    rows = {r["name"]: r for r in br.run()}
    out = {"M0": num(br.M0, MEASURED), "M": rows["M"]["M"], "M_whole": rows["M_whole"]["M"]}
    out["mass_of_delta"] = sb.mass(f64(source_value) - native)
    shared = floor if floor else br.floor
    if shared:                                 # effect in floors: the numeric column the
        out["floor"] = num(float(shared), EXACT if floor else MEASURED)  # band derives from
        for key in ("M0", "M", "M_whole"):
            out[f"{key}_floors"] = num(out[key]["value"] / float(shared), MEASURED)
    out["arms"] = br.table()
    if keep:
        out["branches"] = br
    return out


def iia(records: list[dict[str, Any]], arm: str = "M", margins: tuple[float, ...] = (1.0, 4.0)) -> dict[str, Any]:
    """Interchange intervention accuracy over records with ``M0``, ``floor`` and ``arm`` (plain
    floats or labelled numbers): flip at margin ``m`` = ``M0 > m*floor and M_arm < -m*floor``."""
    def val(x: Any) -> float:
        return float(x["value"]) if isinstance(x, dict) else float(x)

    out: dict[str, Any] = {}
    for m in margins:
        elig = [r for r in records if val(r["M0"]) > m * val(r["floor"])]
        flips = sum(val(r[arm]) < -m * val(r["floor"]) for r in elig)
        out[f"margin{m:g}"] = {"flips": flips, "eligible": len(elig),
                               "iia": (flips / len(elig)) if elig else None}
    return out


def _labelled(value: Any, authority: str) -> Any:
    """Numbers inside an attached label wrapped as ``num(x, authority)`` (records hold no bare
    numbers); strings, None and booleans kept."""
    if isinstance(value, dict):
        return {k: _labelled(v, authority) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_labelled(v, authority) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return num(value, authority)
    return value


def table_row(run: Any, sb: SubspaceBinding, consumer: Any = None) -> dict[str, Any]:
    """One subspace binding's symbol-table row at the cursor: the native cell value's mass in the
    subspace and, under a scalar consumer, the direct consumer share of the in-subspace part at
    fixed attention (P19), plus the attached labels (e.g. measured IIA)."""
    if sb.row is None:
        raise Refusal(f"subspace {sb.name} is a template: bind it to a row first (bound(row))")
    v = f64(run.read_cache(sb.layer, "v", staged=False)[0, sb.kv, sb.row]).cpu()
    prov = json.dumps(sb.provenance, sort_keys=True, default=str)
    row: dict[str, Any] = {
        "schema": SCHEMA, "name": sb.name, "cell": sb.cell, "k": num(sb.k, EXACT),
        "basis_authority": sb.authority, "digest": sb.digest(), "provenance_json": prov,
        "labels": {k: {"result": _labelled(val, auth), "source_authority": auth}
                   for k, (val, auth) in sb.labels.items()},
        "mass_native": sb.mass(v)}
    if consumer is not None:
        from .components import write_of
        from .reads import readout

        reader = readout(run, consumer)
        if reader.scalar:
            c = f64(reader.vector).reshape(-1).cpu()
            j = int(reader.row)
            whole = float(c @ write_of(run, sb, v, j))
            part = float(c @ write_of(run, sb, sb.project(v), j))
            row["cell_to_consumer"] = num(whole, EXACT, "P19")
            row["subspace_to_consumer"] = num(part, EXACT, "P19")
            row["share_of_cell"] = num(part / whole, EXACT) if whole else None
    return row


__all__ = ["BASIS_AUTHORITIES", "SubspaceBinding", "class_mean_subspace", "from_pyvene", "iia",
           "interchange", "orthonormal", "principal_angles", "readout_directions",
           "readout_subspace", "subspace_binding", "table_row"]
