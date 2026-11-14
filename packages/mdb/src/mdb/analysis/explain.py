"""Per-layer explanation of a logit margin (P19) and per-port reconstruction from producers.

Operands come from one native pending-token row: every port of every layer (``engine.ports``),
the key/value cache after the row, the native final-norm output (``fn``) and logits. Weights are
read from the adapter that produced the capture. Every number carries exactly one authority label
(``measured | derived-exact | derived-local | bound | supplied``) and a proof ID; first-order
numbers are ``derived-local`` and never presented as a decomposition. Every result has a state
(``resolved | unresolved | refused``) with a reason when it is not resolved.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import finite
from .finite import BOUND, EXACT, LOCAL, MEASURED, SUPPLIED, f64, f64_at, num

_RUN_STOPS = ("logits", "sample")
# Edits at these ports change the residual stream at a layer boundary; explain_margin books them as
# their own listed terms ``L{l}.edit_{port}`` instead of letting them fall into rounding.
_BOUNDARY_EDIT_PORTS = ("h", "h_out")


@dataclass
class RowCapture:
    """All ports of one pending-token row, with the operands the identities need.

    ``ports[l][name]`` is the native capture at layer ``l`` (shapes as in ``engine.ports``);
    ``keys``/``values`` are the per-layer caches after the row (``(1, H_KV, T, d_h)``, keys
    post-RoPE); ``normed`` is the native final-norm output and ``logits`` the native logits.
    ``edits`` maps layer -> edit manifests realised in THIS row (``edits_this_row`` is the same
    data keyed ``(row, layer)``); ``edits_history`` maps ``(row, layer)`` of an EARLIER row ->
    manifests (they reach this row only through the cache). ``originals[l]`` holds the pre-edit
    capture of each edited port of layer ``l`` (the engine's ``controller.originals``); it is
    what lets ``explain_margin`` book an ``h``/``h_out`` edit as its own term.
    ``originals_missing`` lists ``(layer, port)`` of this-row ``h``/``h_out`` edits whose pre-edit
    value was not retained (``captures_from_run`` without ``recompute_originals``);
    ``originals_source`` says where each held original came from.

    ``infos[l][port]`` is ``Run.read_info`` of that value when the capture came from a ``Run``
    (``captures_from_run``); a ``capture_row`` capture leaves it empty and ``_info`` summarises
    the same fields from ``edits`` (the manifests) and ``realisations[l]`` (the controller's
    realisation record of layer ``l``).
    """

    family: str
    tokens: tuple[int, ...]
    position: int
    ports: list[dict[str, Any]]
    keys: list[Any]
    values: list[Any]
    normed: Any
    logits: Any
    adapter: Any
    edits: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    originals: list[dict[str, Any]] = field(default_factory=list)
    edits_history: dict[tuple[int, int], list[dict[str, Any]]] = field(default_factory=dict)
    originals_missing: list[tuple[int, str]] = field(default_factory=list)
    originals_source: dict[tuple[int, str], str] = field(default_factory=dict)
    infos: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    realisations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def edits_this_row(self) -> dict[tuple[int, int], list[dict[str, Any]]]:
        return {(self.position, layer): v for layer, v in self.edits.items()}


def capture_row(adapter: Any, tokens: Sequence[int], *,
                edits: Mapping[int, Sequence[Any]] | None = None) -> RowCapture:
    """Run the pending row of ``tokens`` once through every layer with an all-port controller.

    ``edits`` maps a layer to ``engine.ports.PortEdit`` objects committed at that layer. Prefix
    rows come from the adapter's native prefill.
    """
    from mdb.engine.ports import PortController, program_for

    program = program_for(adapter)
    edits = dict(edits or {})
    session = adapter.session(list(tokens))
    session.continue_()
    ports, checks, manifests, originals, sources, realisations = [], [], {}, [], {}, []
    for layer in range(adapter.layers):
        controller = PortController(program, layer, edits=tuple(edits.get(layer, ())))
        session.continue_(1, ports=controller)
        ports.append(dict(controller.captures))
        checks.append(dict(controller.checks))
        realisations.append(dict(getattr(controller, "realisation", None) or {}))
        originals.append({k: v.clone() for k, v in controller.originals.items()})
        for key in controller.originals:
            if key in _BOUNDARY_EDIT_PORTS:
                sources[(layer, key)] = "controller.originals"
        if edits.get(layer):
            manifests[layer] = [edit.manifest() for edit in edits[layer]]
    payload = session.capture(retain=False).payload
    keys = [k.clone() for k in payload["keys"]]
    values = [v.clone() for v in payload["values"]]
    session.continue_()
    normed = session.read("hidden")
    session.continue_()
    logits = session.read("logits")
    return RowCapture(program.family, tuple(int(t) for t in tokens), len(tokens) - 1, ports, keys,
                      values, normed, logits, adapter, manifests, checks, originals, {}, [], sources,
                      [], realisations)


def captures_from_run(run: Any, *, recompute_originals: bool = False) -> RowCapture:
    """Read a ``RowCapture`` from an ``engine.run.Run`` whose cursor is at ``logits``/``sample``.

    Every layer port is read through ``Run.read`` (which replays the row's committed edits); the
    key/value caches through ``Run.read_cache``. Edits come from ``Run.committed_edits()``, keyed
    by ``(row, layer)``: those of the explained row go to ``edits`` (layer -> records), those of
    earlier rows to ``edits_history``.

    The pre-edit value of an ``h``/``h_out`` edit of the explained row is not retained for a
    committed layer. With ``recompute_originals=True`` it is rebuilt: ``h`` through
    ``Run.original(target, recompute=True)`` (a re-probe that leaves the run unchanged);
    ``h_out`` (which ``Run.original`` does not return) as the native add ``r + m`` of the
    captured ``r`` and ``m`` (the add the layer performed before the edit). Without it the edits
    land in ``originals_missing`` and ``explain_margin`` reports their terms ``unresolved``.
    With it, the pre-edit native value of every other port an edit or realisation wrote is also
    rebuilt (``Run.original(..., recompute=True)``) for ``explain_port``'s ``original`` field.
    ``Run.read_info`` of every port of the row is kept in ``RowCapture.infos``.
    """
    where = run.where
    row_text, label = where.split(".", 1)
    if label not in _RUN_STOPS:
        raise ValueError(f"explain needs the cursor at t<row>.logits or .sample, not {where}")
    row = int(row_text[1:])
    program = run.program
    layers = run.adapter.layers
    ports = [{name: run.read(f"t{row}.L{layer}.{name}") for name in program.names}
             for layer in range(layers)]
    tokens = tuple(int(t) for t in run.session.read("tokens")[0].tolist())
    edits: dict[int, list[dict[str, Any]]] = {}
    history: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in run.committed_edits():
        if int(record["row"]) != row:
            history.setdefault((int(record["row"]), int(record["layer"])), []).append(record)
        else:
            edits.setdefault(int(record["layer"]), []).append(record)
    infos = [{name: run.read_info(f"t{row}.L{layer}.{name}") for name in program.names}
             for layer in range(layers)]
    originals: list[dict[str, Any]] = [{} for _ in range(layers)]
    missing: list[tuple[int, str]] = []
    source: dict[tuple[int, str], str] = {}
    for layer, records in sorted(edits.items()):
        for port in sorted({program.spec(r["port"]).name for r in records}
                           & set(_BOUNDARY_EDIT_PORTS)):
            value = None
            if recompute_originals and port == "h":
                value = run.original(f"t{row}.L{layer}.h", recompute=True)
                source[(layer, port)] = "Run.original(recompute=True)"
            elif recompute_originals:
                value = ports[layer]["r"] + ports[layer]["m"]
                source[(layer, port)] = "native add r + m of the captured r and m"
            if value is None:
                source.pop((layer, port), None)
                missing.append((layer, port))
            else:
                originals[layer][port] = value.clone()
    if recompute_originals:
        # pre-edit native value of every other port an edit or realisation wrote (explain_port)
        for layer in range(layers):
            for name, info in infos[layer].items():
                if (info["kind"] in ("edit", "realisation") and name not in _BOUNDARY_EDIT_PORTS):
                    value = run.original(f"t{row}.L{layer}.{name}", recompute=True)
                    if value is not None:
                        originals[layer][name] = value.clone()
    return RowCapture(program.family, tokens, row, ports,
                      [run.read_cache(layer, "k")[..., : row + 1, :].clone()
                       for layer in range(layers)],
                      [run.read_cache(layer, "v")[..., : row + 1, :].clone()
                       for layer in range(layers)],
                      run.read(f"t{row}.fn"), run.read(f"t{row}.logits"), run.adapter, edits,
                      [], originals, history, missing, source, infos, [])


def _as_capture(source: Any) -> RowCapture:
    return source if isinstance(source, RowCapture) else captures_from_run(source)


def _info(caps: RowCapture, layer: int, port: str) -> dict[str, Any]:
    """How the value at ``L{layer}.{port}`` of the captured row came about (``Run.read_info`` keys).

    A capture from a ``Run`` carries ``Run.read_info`` itself (``caps.infos``). A ``capture_row``
    capture has only the edit manifests, so the same keys are summarised from them with the
    engine's rules: ``supplied`` only when the last write is an unmasked ``replace``; any other
    written value is ``derived-exact`` with its ``ops``; ``o`` under an ``r`` edit is a
    ``realisation``; ``r`` realised inexactly is ``measured``; ``s`` (and ``p``/``A`` under an
    ``s``/``p`` edit) is a ``replica`` whose check is read from ``caps.checks``.
    """
    if caps.infos:
        return caps.infos[layer][port]
    program = _program(caps.adapter)
    names, spec = program.names, program.spec
    index = names.index(port)
    records = caps.edits.get(layer, [])

    def name_of(record):
        return spec(record["port"]).name

    boundary = ([r for r in caps.edits.get(layer - 1, []) if name_of(r) == "h_out"]
                if port == "h" and layer > 0 else [])
    here = [r for r in records if name_of(r) == port]
    upstream = [{"row": int(row), "layer": int(l), "port": r["port"], "op": r["op"],
                 "staged": False}
                for (row, l), recs in sorted(caps.edits_history.items()) for r in recs]
    for l, recs in sorted(caps.edits.items()):
        for r in recs:
            if r in boundary:
                continue
            if l < layer or (l == layer and names.index(name_of(r)) < index):
                upstream.append({"row": caps.position, "layer": int(l), "port": r["port"],
                                 "op": r["op"], "staged": False})
    info: dict[str, Any] = {
        "edited": False, "by": None, "kind": "native", "authority": "measured",
        "state": "resolved", "reason": None, "ops": [], "selective": False,
        "upstream_edits": upstream, "replica_equal": None, "replica_checked": None,
        "max_abs_diff": None, "max_abs_diff_authority": None, "realisation_exact": None,
        "realisation_max_abs_error": None, "same_port_edit": None}
    edited_ports = {name_of(r) for r in records}
    r_edits = [r for r in records if name_of(r) == "r"]
    realisation = caps.realisations[layer] if layer < len(caps.realisations) else {}
    base = (port == "s" or (port == "p" and "s" in edited_ports)
            or (port == "A" and bool(edited_ports & {"s", "p"})))
    writes = boundary + here
    if port == "o" and r_edits:
        info.update(kind="realisation", authority=EXACT, by="r", ops=[r["op"] for r in r_edits],
                    selective=any("mask_count" in r for r in r_edits))
        if "exact" in realisation:
            info.update(realisation_exact=bool(realisation["exact"]),
                        realisation_max_abs_error=float(realisation["max_abs_error"]))
    elif writes:
        numel = int(caps.ports[layer][port].numel())
        last = writes[-1]
        full = "mask_count" not in last or int(last["mask_count"]) == numel
        info.update(edited=True, kind="edit", ops=[r["op"] for r in writes],
                    by=",".join(sorted({"h_out" if r in boundary else port for r in writes},
                                       key=names.index)),
                    selective=any("mask_count" in r for r in writes),
                    authority="supplied" if last["op"] == "replace" and full else EXACT)
        if port == "r" and "exact" in realisation:
            info.update(realisation_exact=bool(realisation["exact"]),
                        realisation_max_abs_error=float(realisation["max_abs_error"]))
            if not realisation["exact"]:
                info.update(kind="realisation", authority=MEASURED)
    if base and not info["edited"] and info["kind"] == "native":
        info.update(kind="replica", authority=EXACT)
    if port in ("s", "A") and layer < len(caps.checks):
        check = caps.checks[layer].get("s" if port == "s" else "mix_replica")
        info["replica_checked"] = check is not None
        if check is not None:
            info.update(replica_equal=bool(check["equal"]),
                        max_abs_diff=float(check.get("max_abs_diff", 0.0)),
                        max_abs_diff_authority=MEASURED)
            if base and info["authority"] != "supplied" and not check["equal"]:
                info.update(state="refused", authority=None,
                            reason=f"replica check failed: max |replica - native| = "
                                   f"{float(check.get('max_abs_diff', 0.0))}")
    return info


def _vec(x: Any) -> Any:
    return f64(x).reshape(-1)


def _final_norm(adapter: Any) -> tuple[Any, float]:
    norm = getattr(adapter.backbone, adapter.spec.final_norm)
    return norm.weight, float(norm.variance_epsilon)


# Config attributes that change the logits after the head matmul. The margin identity models
# ``logits = W x + b`` only; a softcap or scale the code does not model is refused, not ignored.
_LOGIT_MODIFIERS = (("final_logit_softcapping", None), ("final_logit_softcap", None),
                    ("logit_scale", 1.0), ("logits_scaling", 1.0))

# Rounding bounds of ``within_rounding``. residual_rounding and head are TRUE bounds with ``k = 1``
# (see ``_stream_rounding_bound`` and ``_head_bound``: round-to-nearest, stated premises).
# final_norm is still the a-priori ``k = 2`` multiple of ``||W_a - W_b||_1 ulp(max|fn|)`` (the
# native norm rounds twice: the normalised value, then the gain product) and is flagged
# ``heuristic: True``; its measured worst ratio |remainder| / (k scale) is 0.07 on the test models.
_K_FINAL_NORM = 2


def _ulp(x: float, eps: float) -> float:
    """Spacing of a float format with machine epsilon ``eps`` at ``x``: ``eps 2^floor(log2|x|)``."""
    x = abs(float(x))
    return 0.0 if x == 0.0 else eps * 2.0 ** math.floor(math.log2(x))


def _half_ulp(z: Any) -> Any:
    """Per-coordinate ``1/2 ulp(|z_i|)`` of ``z``'s own dtype as float64 (0 where ``z_i = 0``)."""
    import torch

    eps = float(torch.finfo(z.dtype).eps)
    x = f64(z).reshape(-1).abs()
    spacing = eps * torch.exp2(torch.floor(torch.log2(x.clamp_min(1e-300))))
    return 0.5 * torch.where(x > 0, spacing, torch.zeros_like(x))


def _stream_rounding_bound(caps: RowCapture, originals: Sequence[Mapping[str, Any]],
                           c: Any) -> float:
    """``sum_i |c_i| sum_adds 1/2 ulp(|z_i|)`` over the residual adds of the row (without ``rho``).

    Each native add ``z = fl(x + y)`` rounds once, to nearest, in the dtype of the stream, so
    ``|z - (x + y)| <= 1/2 ulp(z)`` per coordinate (``ulp`` = spacing of the binade of the result;
    this also holds when the sum rounds up into the next binade). The adds are ``r = h + o`` and
    ``h_out = r + m`` of every layer; ``z`` is the captured ``r`` / ``h_out`` (for an edited
    ``h_out`` the pre-edit native add result when it is retained). The remainder is the exact sum
    of these roundings (``e = sum_l [r - (h + o)] + [h_out - (r + m)]`` telescopes), so
    ``|rho c^T e| <= rho sum_i |c_i| sum_adds 1/2 ulp(|z_i|)``.
    """
    total = 0 * c
    for layer, ports in enumerate(caps.ports):
        held = originals[layer] if layer < len(originals) else {}
        total = total + _half_ulp(ports["r"]) + _half_ulp(held.get("h_out", ports["h_out"]))
    return float(c.abs() @ total)


def _head_bound(caps: RowCapture, weight: Any, bias: Any, a: int, b: int,
                acc: Any = None) -> tuple[float | None, list]:
    """Bound on ``|native margin - (W_a - W_b).fn - (b_a - b_b)|`` and its premises.

    Each native logit is ``fl(W_j . fn + b_j)``. A length-``d`` dot product in accumulation
    precision ``u`` obeys ``|fl(w.x) - w.x| <= gamma_d sum_i |w_i x_i|``, ``gamma_d = d u / (1 - d u)``
    (any summation order, FMA allowed); a 16-bit output is then rounded once more
    (``1/2 ulp`` of the logit), and a bias add rounds once more. The margin is the float64
    difference of the two native logits, so the bound is the sum over ``j in {a, b}``.

    ``acc`` (``accumulation.Accumulation`` of the run) gives ``u`` for a 16-bit head: float32
    unless CUDA reduced-precision reductions are allowed; when ``gamma_d`` does not exist there
    the bound is None and the premises' last entry is the refusal.
    """
    import torch

    dtype = caps.logits.dtype
    sixteen = dtype in (torch.bfloat16, torch.float16)
    x = f64(caps.normed).reshape(-1)
    d = x.numel()
    if sixteen and acc is not None:
        premise = acc.premise
        gamma = acc.gamma(d)
        if gamma is None:
            return None, ["round-to-nearest", f"logit dtype {dtype}", acc.refusal(d)]
    else:
        u = 2.0 ** -24 if sixteen else float(torch.finfo(dtype).eps) / 2.0
        premise = "matmul accumulated in " + ("float32 (16-bit inputs)" if sixteen else str(dtype))
        gamma = d * u / (1.0 - d * u)
    logits = caps.logits.reshape(-1)
    rounds = (1 if sixteen else 0) + (1 if bias is not None else 0)
    spacing = float(torch.finfo(dtype).eps)
    total = 0.0
    for j in (a, b):
        total += gamma * float((f64_at(weight, j) * x).abs().sum())   # one row, not the head
        total += rounds * 0.5 * _ulp(float(logits[j]), spacing)
    premises = ["round-to-nearest", f"logit dtype {dtype}",
                premise,
                "head output rounded once" if sixteen else "no extra output rounding"]
    if bias is not None:
        premises.append("bias added with one rounding")
    return total, premises


# Engine edit-record numbers: measurements of the realised edit; the row-sum tolerance and a scalar
# edit operand are supplied constants. Counts, shapes, rows and layers stay structural integers.
_MEASURED_RECORD_KEYS = ("cast_max_abs_error", "realised_max_abs_error", "changed_fraction",
                         "row_sum_min", "row_sum_max", "min")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _label_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of an engine edit record with its measured/supplied numbers as labelled numbers."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key in _MEASURED_RECORD_KEYS and _is_number(value):
            out[key] = num(value, MEASURED, "P19")
        elif key == "row_sum_tolerance" and _is_number(value):
            out[key] = num(value, SUPPLIED, "P19")
        elif key == "value" and isinstance(value, Mapping):
            out[key] = {k: (num(v, SUPPLIED, "P19") if k == "value" and _is_number(v) else v)
                        for k, v in value.items()}
        else:
            out[key] = value
    return out


def _label_edits(edits: Mapping[Any, Sequence[Mapping[str, Any]]]) -> dict[Any, list[dict]]:
    return {key: [_label_record(r) for r in records] for key, records in edits.items()}


def _program(adapter: Any) -> Any:
    from mdb.engine.ports import program_for

    return program_for(adapter)


def _readout_refusal(config: Any) -> str | None:
    for name, neutral in _LOGIT_MODIFIERS:
        value = getattr(config, name, neutral)
        if value is not None and value != neutral:
            return f"config.{name} = {value!r} modifies the logits and is not modelled"
    return None


def _rope_refusal(config: Any) -> str | None:
    """Non-default RoPE (type, scaling, partial rotary width) is not what ``rope_rotate`` models."""
    params = getattr(config, "rope_parameters", None)
    scaling = getattr(config, "rope_scaling", None)
    sources = [x for x in (params, scaling) if isinstance(x, Mapping)]
    for source in sources:
        kind = source.get("rope_type", source.get("type"))
        if kind not in (None, "default"):
            return f"rope type {kind!r} is not modelled (default unscaled RoPE only)"
    factors = [source.get("partial_rotary_factor") for source in sources]
    factors.append(getattr(config, "partial_rotary_factor", None))
    for factor in factors:
        if factor not in (None, 1.0):
            return f"partial_rotary_factor {factor!r} is not modelled (full rotary width only)"
    return None


def explain_margin(source: Any, a: int, b: int, *, first_order: bool = False) -> dict[str, Any]:
    """Per-layer signed contributions to ``M_ab = logit_a - logit_b`` at the pending token (P19).

    The final residual is ``h_L = h_0 + sum_l (o_l + m_l) + edits + e``, where ``e`` collects the
    measured rounding of every native residual add (``r - (h + o)``, ``h_out - (r + m)``) and any
    boundary mismatch that is not an edit. With ``c = Gamma^F (W_a - W_b)^T`` and the final-norm
    scale ``rho^F(h_L)`` evaluated once at the full measured final state (chapter 02: shared, not
    attributed), the listed terms are ``rho^F c^T h_0`` and ``rho^F c^T o_l``, ``rho^F c^T m_l``
    (``derived-exact``; each row also carries its scale-free ``projection`` ``c^T xi``).

    A user edit at ``h`` or ``h_out`` of the explained row is a listed term
    ``L{l}.edit_{port} = rho^F c^T (edited - original)`` (the pre-edit value is the engine's
    ``controller.originals``), never part of ``residual_rounding``; with an ``h`` edit at layer 0
    the ``embedding`` term uses the pre-edit ``h_0``. If the head has a bias, ``head_bias =
    b_a - b_b`` is a listed term (``derived-exact``). A config logit softcap/scale that is not
    modelled makes the result ``refused``.

    Labels of the listed terms come from ``read_info`` of the port each term reads (``o`` for
    ``L{l}.attn``, ``m`` for ``L{l}.mlp``; an ``r`` edit shows up as ``o`` with ``kind:
    realisation``; edit_h / edit_h_out from their own op): a value that is exactly a supplied
    tensor (a pure ``replace``) is ``supplied``; add / scale / mixed ops or a realisation are
    ``derived-exact`` (``op``, ``ops``), an inexact ``r`` realisation is ``measured``; a native
    value is ``derived-exact`` (the identity over the captured value). Every term written by an
    edit carries ``edit`` (the edit address(es), ``t{row}.L{l}.{port}``) and ``kind``.
    ``listed_sum`` is ``derived-exact`` (exact arithmetic over its terms) with ``includes_edits``
    (term names written by edits) and, when any term is ``supplied``, ``includes_supplied``.

    Remainders: ``residual_rounding`` ``rho^F c^T e`` (measured), float64 ``unlisted`` roundoff,
    ``final_norm`` and ``head`` (measured), so listed + remainders equals the native margin to
    float64 roundoff. They are gaps by construction, so each of the first, third and fourth also
    carries a ``scale`` (the bound), ``k`` and ``within_rounding = |remainder| <= k * scale``:

    * residual_rounding: ``scale = rho^F sum_i |c_i| sum_adds 1/2 ulp(|z_i|)`` (``bound``, ``k =
      1``): every native add rounds once to nearest, error <= 1/2 ulp of its result per
      coordinate, ``rho^F`` is the final-norm scale ``rho_F`` and the ulp is that of the dtype of
      the residual ports ``h, r, h_out`` (not the logits dtype); see ``_stream_rounding_bound``;
    * head: ``scale = gamma_d sum_{j in a,b} sum_i |W_ji fn_i|`` plus ``1/2 ulp`` of each logit per
      output rounding (``bound``, ``k = 1``; ``gamma_d = d u / (1 - d u)``); see ``_head_bound``;
    * final_norm: ``scale = ||W_a - W_b||_1 u(max|fn|)``, ``k = 2``, ``heuristic: True`` (an
      a-priori multiple, not a derived bound).

    ``first_order=True`` adds a ``derived-local`` column ``lambda_r^T o_l``,
    ``lambda_{h_{l+1}}^T m_l``, ``lambda_{h_0}^T h_0`` from ``circuit_native.consumer_gradient``
    (model-dtype autograd adjoints at each layer input) and the P14 pullback
    ``lambda_r = (I + J_M)^T lambda_{h_{l+1}}``; those numbers do not sum to the margin. On a row
    with edits in this row or in the run's history the column is ``{state: "refused", reason}``
    (no numbers). Edits of earlier rows reach this row only through the cache: they are reported
    as ``edits_history`` (with a count) and not as terms.
    """
    import torch

    caps = _as_capture(source)
    adapter = caps.adapter
    config = adapter.model.config
    base: dict[str, Any] = {
        "schema": "mdb-explain-margin-v1", "proof": "P19", "tokens": {"a": int(a), "b": int(b)},
        "position": caps.position, "family": caps.family,
    }
    refusal = _readout_refusal(config)
    layers = len(caps.ports)
    originals = list(caps.originals) + [{}] * (layers - len(caps.originals))
    if refusal is not None:
        return {**base, "state": "refused", "reason": refusal, "edits": _label_edits(caps.edits)}
    missing = set(caps.originals_missing)
    program = _program(adapter)
    ops: dict[tuple[int, str], list[str]] = {}
    last_write: dict[tuple[int, str], Mapping[str, Any]] = {}
    for layer, records in caps.edits.items():
        for record in records:
            port = program.spec(record["port"]).name
            if port in _BOUNDARY_EDIT_PORTS:
                ops.setdefault((layer, port), []).append(record["op"])
                last_write[(layer, port)] = record
    head = adapter.model.lm_head
    weight, head_bias = head.weight, getattr(head, "bias", None)
    gamma, eps = _final_norm(adapter)
    delta_h: dict[int, Any] = {}
    delta_out: dict[int, Any] = {}
    for layer in range(layers):
        if "h" in originals[layer]:
            delta_h[layer] = _vec(caps.ports[layer]["h"]) - _vec(originals[layer]["h"])
        if "h_out" in originals[layer]:
            delta_out[layer] = _vec(caps.ports[layer]["h_out"]) - _vec(originals[layer]["h_out"])
    embedding = _vec(caps.ports[0]["h"])
    if 0 in delta_h:
        embedding = embedding - delta_h[0]
    components: dict[str, Any] = {"embedding": embedding}
    rounding = []
    edit_names: dict[str, tuple[int, str]] = {}
    for layer, ports in enumerate(caps.ports):
        h, o, r = _vec(ports["h"]), _vec(ports["o"]), _vec(ports["r"])
        m, out = _vec(ports["m"]), _vec(ports["h_out"])
        if layer in delta_h:
            components[f"L{layer}.edit_h"] = delta_h[layer]
            edit_names[f"L{layer}.edit_h"] = (layer, "h")
        components[f"L{layer}.attn"] = o
        components[f"L{layer}.mlp"] = m
        if layer in delta_out:
            components[f"L{layer}.edit_h_out"] = delta_out[layer]
            edit_names[f"L{layer}.edit_h_out"] = (layer, "h_out")
        boundary = 0 * h if layer + 1 == layers else _vec(caps.ports[layer + 1]["h"]) - out
        rounding.append({"attn_add": r - (h + o), "mlp_add": out - (r + m), "boundary": boundary})
    # The raw gaps contain the supplied edits: ``mlp_add_l`` holds the h_out edit of layer l,
    # ``boundary_{l-1}`` holds the h edit of layer l >= 1 (the layer-0 h edit sits in no gap: the
    # embedding term uses the pre-edit h_0). They are removed here and listed as edit terms.
    removed = sum((-d for layer, d in delta_h.items() if layer >= 1), 0 * embedding)
    removed = removed - sum(delta_out.values(), 0 * embedding)
    e_total = sum(sum(item.values()) for item in rounding) + removed
    components["residual_rounding"] = e_total
    h_last = _vec(caps.ports[-1]["h_out"])
    logits = caps.logits.reshape(-1)
    native_margin = f64(logits[a]) - f64(logits[b])
    result = finite.readout_margin(h_last, weight, gamma, eps, a, b, components=components,
                                   native_margin=native_margin,
                                   native_normed=_vec(caps.normed), out_bias=head_bias)
    rho = result.extras["rho"]
    c = result.extras["c"]

    reason_missing = "pre-edit value not retained; pass recompute_originals=True"

    infos = {(layer, port): _info(caps, layer, port) for layer in range(layers)
             for port in ("o", "m")}
    refused_ports = {key: info for key, info in infos.items() if info["state"] == "refused"}

    def claim(value, authority, proof):
        """A labelled number, or an inspection-only record when the result is refused."""
        if refused_ports:
            return {"value": float(value), "authority": None, "inspection_only": True,
                    "proof": proof}
        return num(value, authority, proof)

    def term(name, layer, port):
        entry = {"name": name, "layer": layer, "port": port}
        authority = EXACT
        if name in edit_names:
            # The edit's own ops decide the label (the engine's rule): the value is ``supplied``
            # only when the last write is an unmasked replace; an add/scale operand, a masked
            # replace or a mix is derived-exact.
            kinds = ops[edit_names[name]]
            last = last_write[edit_names[name]]
            full = "mask_count" not in last or int(last["mask_count"]) == caps.ports[
                edit_names[name][0]][edit_names[name][1]].numel()
            authority = SUPPLIED if last["op"] == "replace" and full else EXACT
            entry.update(op="+".join(sorted(set(kinds))), ops=list(kinds),
                         edit=[f"t{caps.position}.L{layer}.{edit_names[name][1]}"])
        elif layer is not None and port.split(".")[-1] in ("o", "m"):
            # the label comes from Run.read_info of the port the term reads
            info = infos[(layer, port.split(".")[-1])]
            if info["state"] == "refused":
                entry.update(state="refused", reason=info["reason"])
            elif info["kind"] != "native":
                authority = info["authority"]
                entry.update(kind=info["kind"], op="+".join(sorted(set(info["ops"]))),
                             ops=list(info["ops"]),
                             edit=[f"t{caps.position}.L{layer}.{p}"
                                   for p in (info["by"] or "").split(",") if p])
        entry["value"] = claim(result.terms[name], authority, "P19")
        entry["projection"] = claim(float(c @ components[name]), authority, "P19")
        return entry

    def shell(name, layer, port):
        entry = {"name": name, "layer": layer, "port": port, "state": "unresolved",
                 "reason": reason_missing}
        if (layer, port.split(".")[-1]) in ops:
            entry["op"] = "+".join(sorted(set(ops[(layer, port.split(".")[-1])])))
        return entry

    terms = [shell("embedding", None, "L0.h") if (0, "h") in missing
             else term("embedding", None, "L0.h")]
    for layer in range(layers):
        if (layer, "h") in missing:
            terms.append(shell(f"L{layer}.edit_h", layer, f"L{layer}.h"))
        elif f"L{layer}.edit_h" in components:
            terms.append(term(f"L{layer}.edit_h", layer, f"L{layer}.h"))
        terms.append(term(f"L{layer}.attn", layer, f"L{layer}.o"))
        terms.append(term(f"L{layer}.mlp", layer, f"L{layer}.m"))
        if (layer, "h_out") in missing:
            terms.append(shell(f"L{layer}.edit_h_out", layer, f"L{layer}.h_out"))
        elif f"L{layer}.edit_h_out" in components:
            terms.append(term(f"L{layer}.edit_h_out", layer, f"L{layer}.h_out"))
    if head_bias is not None:
        terms.append({"name": "head_bias", "layer": None, "port": "lm_head.bias",
                      "value": claim(result.terms["head_bias"], EXACT, "P19")})
    per_layer_rounding = [
        {key: claim(float(rho * (c @ vec)), MEASURED, "P01") for key, vec in item.items()}
        for item in rounding
    ]
    # rounding bounds (see the docstring)
    diff = f64_at(weight, a) - f64_at(weight, b)
    stream_dtype = caps.ports[0]["h"].dtype
    bound_rr = float(rho) * _stream_rounding_bound(caps, originals, c)
    from .accumulation import accumulation

    head_bound, head_premises = _head_bound(caps, weight, head_bias, a, b,
                                            accumulation(caps.adapter, caps.logits.dtype))
    scale_fn = float(diff.abs().sum()) * _ulp(
        float(_vec(caps.normed).abs().max()), float(torch.finfo(caps.normed.dtype).eps))
    stream_premises = ["round-to-nearest", "one rounding per residual add",
                       f"residual stream dtype {stream_dtype} (dtype of h, r, h_out)"]

    def bounded(entry, scale, k, formula, authority, premises=(), heuristic=False):
        value = abs(entry["value"]["value"])
        entry.update(scale=claim(scale, authority, "P19"), k=int(k), scale_formula=formula,
                     within_rounding=bool(value <= k * scale), heuristic=heuristic,
                     premises=list(premises))
        return entry

    remainders = [
        bounded({"name": "residual_rounding",
                 "value": claim(result.terms["residual_rounding"], MEASURED, "P01"),
                 "per_layer": per_layer_rounding,
                 "edits_removed": claim(float(rho * (c @ removed)),
                                        SUPPLIED if any("replace" in v for v in ops.values())
                                        else EXACT, "P19"),
                 "meaning": "rho^F c^T (native residual-add rounding and boundary mismatch); "
                            "per_layer entries are the raw gaps (they contain supplied h/h_out "
                            "edits) and edits_removed takes those edits out, listed as edit_* "
                            "terms"},
                bound_rr, 1, "rho^F sum_i |c_i| sum_adds 1/2 ulp(|z_i|); k = 1", BOUND,
                stream_premises),
        {"name": "unlisted", "value": claim(result.terms["unlisted"], EXACT, "P19"),
         "meaning": "float64 roundoff of h_L minus the listed components"},
        bounded({"name": "final_norm", "value": claim(result.extras["eta_final_norm"], MEASURED,
                                                      "P19"),
                 "meaning": "(W_a - W_b).(native fn output - rho Gamma h_L)"},
                scale_fn, _K_FINAL_NORM, "||W_a - W_b||_1 ulp(max |fn|); k = 2", EXACT,
                heuristic=True),
        bounded({"name": "head", "value": claim(result.extras["eta_head"], MEASURED, "P19"),
                 "meaning": "native margin - (W_a - W_b).(native fn output) - head bias, float64"},
                head_bound, 1, "gamma_d (sum_i |W_ai x_i| + sum_i |W_bi x_i|) + output roundings; "
                "k = 1", BOUND, head_premises) if head_bound is not None else
        {"name": "head", "value": claim(result.extras["eta_head"], MEASURED, "P19"),
         "meaning": "native margin - (W_a - W_b).(native fn output) - head bias, float64",
         "bound": {"state": "refused", "reason": head_premises[-1]}},
    ]
    unresolved = bool(missing)
    if unresolved:
        # the edit sits inside the raw gaps (and, for an h edit at layer 0, in the embedding):
        # the rounding remainder cannot be separated, so it is not reported as a number
        remainders[0] = {"name": "residual_rounding", "state": "unresolved",
                         "reason": reason_missing}
    listed = sum(t["value"]["value"] for t in terms if "value" in t)
    rem = sum(r["value"]["value"] for r in remainders if "value" in r)
    native = float(native_margin)
    refused = bool(refused_ports)
    reason_refused = "; ".join(f"L{layer}.{port}: {info['reason']}"
                               for (layer, port), info in sorted(refused_ports.items()))
    edited_terms = [t["name"] for t in terms if "edit" in t]
    supplied_terms = [t["name"] for t in terms
                      if t.get("value", {}).get("authority") == SUPPLIED]
    listed_sum = {"state": "unresolved", "reason": reason_missing} if unresolved else claim(
        listed, EXACT, "P19")
    if not unresolved:
        listed_sum["includes_edits"] = edited_terms
        if supplied_terms:
            listed_sum["includes_supplied"] = supplied_terms
    out: dict[str, Any] = {
        **base,
        "state": "refused" if refused else "unresolved" if unresolved else "resolved",
        "reason": reason_refused if refused else reason_missing if unresolved else None,
        "edits": _label_edits(caps.edits),
        "native_margin": claim(native, MEASURED, "P19"),
        "native_dtype": str(caps.logits.dtype),
        "float64_margin": claim(float(result.reference), EXACT, "P19"),
        "eta": claim(native - float(result.reference), MEASURED, "P19"),
        "rho_F": claim(float(rho), EXACT, "P19"),
        "rho_F_note": "final-norm scale evaluated once at the full measured h_L; not attributed",
        "terms": terms,
        "remainders": remainders,
        "all_within_rounding": None if unresolved or refused or any(
            "bound" in r for r in remainders) else all(
            r["within_rounding"] for r in remainders if "k" in r),
        "listed_sum": listed_sum,
        "closure": {"state": "unresolved", "reason": reason_missing} if unresolved else claim(
            listed + rem - native, EXACT, "P19"),
        "closure_rel": {"state": "unresolved", "reason": reason_missing} if unresolved else claim(
            abs(listed + rem - native) / max(abs(native), 1e-300), EXACT, "P19"),
    }
    if caps.edits_history:
        out["edits_history"] = {
            "count": sum(len(v) for v in caps.edits_history.values()),
            "entries": [{"row": row, "layer": layer, "edits": [_label_record(r) for r in v]}
                        for (row, layer), v in sorted(caps.edits_history.items())],
            "note": "edits of earlier rows reach this row through the cache; not listed as terms",
        }
    if first_order:
        out["first_order"] = _first_order(caps, a, b, result, components)
    return out


MAX_TERMS = 64          # listed terms in a result (the record's sequence bound)
_FULL_TERMS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _full_bytes(full: dict[str, Any]) -> bytes:
    return json.dumps(full, sort_keys=True, separators=(",", ":")).encode()


def _per_layer(result: dict[str, Any]) -> list[dict[str, Any]]:
    for rem in result["remainders"]:
        if rem["name"] == "residual_rounding" and "per_layer" in rem:
            return rem["per_layer"]
    return []


def bound_terms(result: dict[str, Any], *, store: Any = None, run: Any = None,
                cap: int = MAX_TERMS) -> dict[str, Any]:
    """Bound the lists of ``result`` that grow with the layers to ``cap`` items (in place;
    returns ``result``): ``terms`` (two per layer plus the embedding) and the residual-rounding
    remainder's ``per_layer`` entries.

    Over the cap ``terms`` keeps the terms with the largest ``|value|`` (unresolved terms first:
    they carry a state the reader must see), in their original order, and ``terms_summary``
    gives ``count``, ``listed``, ``rest`` (``count`` and ``sum``, ``derived-exact``, so listed +
    rest still adds to ``listed_sum``) and ``pointer`` to the full lists. ``per_layer`` keeps the
    entries with the largest magnitude, each tagged with its ``layer``, and the remainder gets
    ``per_layer_count``. The full lists are JSON bytes put in ``store`` (a project
    ``LocalStore``, a tensor pointer to uint8 bytes, checked by ``sha256``) or, with no store,
    held in memory per ``run`` (``load_terms`` / ``load_full`` resolve both). Within the cap
    nothing changes and no summary is added.
    """
    terms, per_layer = result["terms"], _per_layer(result)
    if len(terms) <= cap and len(per_layer) <= cap:
        return result
    import numpy as np

    from mdb.core.pointers import tensor_pointer

    full = {"terms": terms, "per_layer": per_layer}
    raw = _full_bytes(full)
    digest = hashlib.sha256(raw).hexdigest()
    if store is not None:
        stored = store.put_tensor(np.frombuffer(raw, dtype=np.uint8).copy())
        pointer: dict[str, Any] = {
            "state": "stored", "ref": stored["ref"], "bytes": len(raw), "sha256": digest,
            "tensor": tensor_pointer(stored["ref"], sha256=stored["sha256"],
                                     shape=stored["shape"], dtype=stored["dtype"])}
    else:
        if run is not None:
            _FULL_TERMS.setdefault(run, {})[digest] = full
        pointer = {"state": "absent", "sha256": digest,
                   "reason": "no project store: the full lists are held in memory on the "
                             "run (explain.load_full), not stored"}

    def top(items: list[dict[str, Any]], size: float) -> list[tuple[int, dict[str, Any]]]:
        def rank(item: tuple[int, dict[str, Any]]) -> tuple[int, float]:
            return (0, 0.0) if "value" not in item[1] else (1, -abs(item[1]["value"]["value"]))

        return sorted(sorted(enumerate(items), key=rank)[:int(size)], key=lambda item: item[0])

    keep = top(terms, cap)
    kept = {index for index, _ in keep}
    rest = [t for i, t in enumerate(terms) if i not in kept]
    result["terms"] = [t for _, t in keep]
    result["terms_summary"] = {
        "count": len(terms), "listed": len(keep),
        "rest": {"count": len(rest),
                 "sum": claim_sum(sum(t["value"]["value"] for t in rest if "value" in t)),
                 "unresolved": sum(1 for t in rest if "value" not in t)},
        "pointer": pointer}
    if len(per_layer) > cap:
        magnitude = [{"value": {"value": max((abs(c["value"]) for c in entry.values()
                                              if isinstance(c, dict) and "value" in c),
                                             default=0.0)}} for entry in per_layer]
        chosen = top(magnitude, cap)
        for rem in result["remainders"]:
            if rem["name"] == "residual_rounding" and "per_layer" in rem:
                rem["per_layer"] = [{"layer": i, **per_layer[i]} for i, _ in chosen]
                rem["per_layer_count"] = len(per_layer)
    return result


def claim_sum(value: float) -> dict[str, Any]:
    return num(float(value), EXACT, "P19")


def load_full(summary: Mapping[str, Any], *, store: Any = None, run: Any = None
              ) -> dict[str, Any]:
    """The full ``{"terms", "per_layer"}`` lists ``terms_summary`` points at: from ``store``
    (stored blob, checked against the summary's ``sha256``) or from the run's in-memory copy;
    ``ValueError`` if absent or changed."""
    pointer = summary["pointer"]
    if pointer["state"] == "stored":
        from mdb.core.pointers import resolve_tensor

        raw = resolve_tensor(pointer["tensor"], store=store, framework="numpy").tobytes()
        if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
            raise ValueError("the stored term list differs from the summary's sha256")
        return json.loads(raw)
    held = _FULL_TERMS.get(run, {}).get(pointer["sha256"]) if run is not None else None
    if held is None:
        raise ValueError("the full term list is not held: " + pointer["reason"])
    return held


def load_terms(summary: Mapping[str, Any], *, store: Any = None, run: Any = None
               ) -> list[dict[str, Any]]:
    """The full term list ``terms_summary`` points at (see ``load_full``)."""
    return load_full(summary, store=store, run=run)["terms"]


def _layer_module(adapter: Any, layer: int) -> Any:
    """Decoder layer ``layer`` of a saturn_pub ``DecoderAdapter`` (its public backbone/spec)."""
    return getattr(adapter.backbone, adapter.spec.layers)[layer]


def consumer_gradient(session: Any, token: int, contrast: int) -> tuple[float, Any]:
    """``(value, d value / d hidden)`` for ``value = logits[token] - logits[contrast]`` of the
    pending row, from the hidden state entering the session's pending decoder layer.

    The same computation as the lane function ``circuit_native.consumer_gradient`` restricted to
    what ``_first_order`` reads (the primal and the hidden adjoint): the adapter's own layer
    executor under autograd, the past KV fixed (detached clones of the captured cache), final norm,
    LM head and logit softcap; no argmax/rounding derivative. Resident ``DecoderAdapter`` of a
    family with a port program (``engine.ports.FAMILIES``), one pending token, batch one, only.
    """
    import torch
    from saturn_pub.adapters.decoder import DecoderAdapter

    from mdb.engine.ports import PROGRAMS

    adapter = session.adapter
    if (not isinstance(adapter, DecoderAdapter)
            or adapter.spec.model_type not in PROGRAMS
            or adapter.execution.get("residency", "resident") != "resident"):
        raise ValueError("native circuit expansion supports a resident public DecoderAdapter "
                         f"of a port-program family ({', '.join(PROGRAMS)})")
    adapter.validate_execution()
    state = session.capture(retain=False).payload
    index = state["layer"]
    if state["phase"] != "layer" or not 0 <= index < adapter.layers:
        raise ValueError("select hidden at a pending decoder layer before expanding native "
                         "attention/MLP")
    hidden = session.read("hidden")
    if hidden is None or hidden.shape[:2] != (1, 1):
        raise ValueError("native circuit expansion requires one pending token and batch one")
    if max(token, contrast) >= adapter.model.config.vocab_size:
        raise ValueError("consumer token is outside native vocabulary")
    with torch.inference_mode(False), torch.enable_grad():
        hidden = hidden.detach().clone().requires_grad_(True)
        keys = [x.detach().clone() for x in state["keys"]]
        values = [x.detach().clone() for x in state["values"]]
        cache = adapter._cache(keys, values)
        position = session.read("tokens").shape[-1] - 1
        out = hidden
        for layer in range(index, adapter.layers):
            out = adapter._run_layer(layer, out, cache, position)
        logits = adapter._apply_logit_softcap(
            adapter.model.lm_head(getattr(adapter.backbone, adapter.spec.final_norm)(out))
        )[:, -1]
        target = logits[0, token] - logits[0, contrast]
        (gradient,) = torch.autograd.grad(target, [hidden])
        value = float(target.detach())
        gradient = gradient.detach().clone()
    adapter.validate_execution()
    return value, gradient


def _first_order(caps: RowCapture, a: int, b: int, result: Any,
                 components: Mapping[str, Any]) -> dict[str, Any]:
    if caps.edits or caps.edits_history:
        where = []
        if caps.edits:
            where.append(f"edits in this row at layers {sorted(caps.edits)}")
        if caps.edits_history:
            where.append("edits in earlier rows (history) at (row, layer) "
                         f"{sorted(caps.edits_history)}")
        return {"state": "refused", "authority": None,
                "reason": "first-order column needs an unedited row: consumer_gradient replays "
                          "the native suffix without port edits; " + "; ".join(where)}
    adapter = caps.adapter
    layers = len(caps.ports)
    adjoints, primals = [], []
    # One prefill, one layer transition per layer: consumer_gradient only reads the session
    # (a capture it does not retain and cloned reads), so it stays at the same state.
    session = adapter.session(list(caps.tokens))
    session.continue_()
    for layer in range(layers):
        if layer:
            session.continue_()
        value, grad = consumer_gradient(session, int(a), int(b))
        adjoints.append(_vec(grad))
        primals.append(value)
    adjoints.append(result.extras["gradient"])
    rows = [{"name": "embedding", "value": num(float(adjoints[0] @ components["embedding"]),
                                               LOCAL, "P19")}]
    for layer, ports in enumerate(caps.ports):
        mlp = _layer_module(adapter, layer).mlp
        norm = _layer_module(adapter, layer).post_attention_layernorm
        lam_out = adjoints[layer + 1]
        lam_r = finite.mlp_pullback(lam_out, _vec(ports["r"]), mlp.gate_proj.weight,
                                    mlp.up_proj.weight, mlp.down_proj.weight, norm.weight,
                                    float(norm.variance_epsilon)).total
        rows.append({"name": f"L{layer}.attn", "value": num(
            float(lam_r @ components[f"L{layer}.attn"]), LOCAL, "P14")})
        rows.append({"name": f"L{layer}.mlp", "value": num(
            float(lam_out @ components[f"L{layer}.mlp"]), LOCAL, "P14")})
    logits = caps.logits.reshape(-1)
    native = float(f64(logits[a]) - f64(logits[b]))
    return {
        "state": "resolved",
        "reason": None,
        "authority": LOCAL,
        "rows": rows,
        "adjoint_source": "consumer_gradient (model dtype autograd, fixed past "
                          "KV) at each layer input; P14 float64 pullback through each MLP; P19 "
                          "float64 readout gradient for the last layer",
        "consumer_primal_minus_native": [num(p - native, MEASURED, "P19") for p in primals],
        "note": "first-order products lambda^T xi; they are not a decomposition of the margin",
    }


_PRODUCERS: dict[str, tuple[str, ...]] = {
    "a": ("h",), "q": ("a",), "k": ("a",), "v": ("a",), "qn": ("q",), "kn": ("k",),
    "s": ("qr", "kr"), "p": ("s",), "A": ("p", "values"), "o": ("A",), "r": ("h", "o"),
    "n": ("r",), "g": ("n",), "u": ("n",), "sg": ("g",), "f": ("sg", "u"), "m": ("f",),
    "h_out": ("r", "m"),
}


def _theta(config: Any) -> float:
    rope = getattr(config, "rope_parameters", None)
    if isinstance(rope, Mapping) and "rope_theta" in rope:
        return float(rope["rope_theta"])
    return float(config.rope_theta)


def explain_port(source: Any, layer: int, port: str) -> dict[str, Any]:
    """Reconstruct one captured port from its producers by the identity that defines it.

    Returns the port program's operator and declared proofs, the producers used, the identity
    (proof ID, function, kind), the float64 reconstruction, the native capture and the measured
    remainder ``eta = native - reconstruction``. ``kr`` is reconstructed for the pending row only
    (cached pre-RoPE keys are not captured); ``h`` at layer 0 is the embedding lookup.

    ``eta`` is the native output minus the float64 identity on the captured operands, so it exists
    only for a value the native code produced. A native value carries ``eta_max_abs`` /
    ``eta_rel`` (``measured`` against the native capture, ``against: "native"``). The engine's ``s``
    capture is a replica (not a native tensor): ``kind`` of its ``read_info`` is ``replica`` and its
    ``eta`` is ``derived-exact`` with ``against: "replica"`` in BOTH ``eta_max_abs`` and
    ``result.summary()``; the same holds for ``p``/``A`` under an ``s``/``p`` edit. A replica whose
    check failed is ``state: "refused"``; one whose check has not run stays resolved with
    ``replica_equal: None`` and a note. A value written by an edit or a realisation
    (``value_kind in {edit, realisation}``) has NO ``eta``: it returns ``authority`` (from
    ``read_info``), ``edit`` (``by``, ``ops``, ``kind``), ``edit_delta_max_abs`` (``measured``: the
    edit's displacement from the identity, never a remainder) and, when the pre-edit original is
    held, ``original.eta_max_abs`` (the native remainder of the original). ``under_edits`` lists the
    edits that fired before the value in program order. ``qr``/``kr`` under a RoPE the identity
    does not model (non-default type or scaling, partial rotary width) return ``state: "refused"``
    with the reason instead of a large ``eta``. ``schema``, ``state``, ``reason``, ``against``,
    ``value_kind``, ``under_edits`` are always present.
    """
    from mdb.engine.ports import program_for

    caps = _as_capture(source)
    adapter = caps.adapter
    spec_program = program_for(adapter)
    spec = spec_program.spec(port)
    module = _layer_module(adapter, layer)
    attn = module.self_attn
    ports = caps.ports[layer]
    heads, kv_heads, d_h = adapter.num_heads, adapter.num_kv_heads, adapter.head_dim
    t = caps.position
    kappa = finite.gqa_kappa(heads, kv_heads)
    q_src, k_src = spec_program.spec("qr").inputs[0], spec_program.spec("kr").inputs[0]
    notes: list[str] = []
    producers = _PRODUCERS.get(port, ())
    native = ports[port]
    if spec.name in ("qr", "kr"):
        refusal = _rope_refusal(adapter.model.config)
        if refusal is not None:
            return {"schema": "mdb-explain-port-v1", "layer": layer, "port": spec.name,
                    "operator": spec.operator, "proofs_declared": list(spec.proofs),
                    "proof": "P04", "function": "rope_rotate", "state": "refused",
                    "reason": refusal, "against": "native"}

    def norm(x, mod, nat, width=None):
        x = f64(x) if width is None else f64(x).reshape(-1, width)
        nat = nat if width is None else nat.reshape(-1, width)
        return finite.rms_norm(x.reshape(-1, x.shape[-1]), mod.weight, float(mod.variance_epsilon),
                               native=nat.reshape(-1, nat.shape[-1]))

    def lin(x, mod, nat, proof="P04"):
        return finite.affine(_vec(x), mod.weight, mod.bias, native=nat.reshape(-1), proof=proof)

    if port == "h":
        if layer == 0:
            producers = ("embedding", "token")
            res = finite.embedding_lookup(getattr(adapter.backbone, adapter.spec.embed).weight,
                                          caps.tokens[-1], native=native.reshape(-1))
        else:
            producers = (f"L{layer - 1}.h_out",)
            prev = caps.ports[layer - 1]["h_out"]
            res = finite.residual_add(_vec(prev), 0 * _vec(prev), native=native.reshape(-1))
            notes.append("layer boundary: h_l equals the previous layer output")
    elif port == "a":
        res = norm(ports["h"], module.input_layernorm, native)
    elif port in ("q", "k", "v"):
        res = lin(ports["a"], getattr(attn, f"{port}_proj"), native)
    elif port in ("qn", "kn"):
        source_port = port[0]
        res = norm(ports[source_port], getattr(attn, f"{source_port}_norm"), native, width=d_h)
    elif port == "qr":
        producers = (q_src,)
        qbar = f64(ports[producers[0]]).reshape(heads, d_h)
        res = finite.rope_rotate(qbar, t, _theta(adapter.model.config),
                                 native=native[0, :, 0, :])
    elif port == "kr":
        producers = (k_src,)
        kbar = f64(ports[producers[0]]).reshape(kv_heads, d_h)
        res = finite.rope_rotate(kbar, t, _theta(adapter.model.config),
                                 native=native[0, :, t, :])
        notes.append("pending row only; cached rows' pre-RoPE keys are not captured")
    elif port == "s":
        keys = ports["kr"][0][kappa]
        res = finite.scores(f64(ports["qr"][0, :, 0, :]), keys, scaling=float(attn.scaling),
                            native=native[0, :, 0, :])
        notes.append("single pending row: no attention mask, every cached source visible")
    elif port == "p":
        res = finite.softmax(ports["s"][0, :, 0, :], native=native[0, :, 0, :])
    elif port == "A":
        values = caps.values[layer][0][kappa]
        res = finite.transport(ports["p"][0, :, 0, :], values, native=native.reshape(heads, d_h))
        pending = f64(caps.values[layer][0, :, t, :]).reshape(-1)
        notes.append(f"value cache row t vs captured v max abs diff "
                     f"{float((pending - _vec(ports['v'])).abs().max()):.3g}")
    elif port == "o":
        res = finite.head_write(_vec(ports["A"]), attn.o_proj.weight, attn.o_proj.bias,
                                native=native.reshape(-1))
    elif port == "r":
        res = finite.residual_add(_vec(ports["h"]), _vec(ports["o"]), native=native.reshape(-1))
    elif port == "n":
        res = norm(ports["r"], module.post_attention_layernorm, native)
    elif port in ("g", "u"):
        mod = module.mlp.gate_proj if port == "g" else module.mlp.up_proj
        res = lin(ports["n"], mod, native, proof="P10")
    elif port == "sg":
        res = finite.swiglu_gate(_vec(ports["g"]), native=native.reshape(-1))
    elif port == "f":
        res = finite.swiglu_product(_vec(ports["sg"]), _vec(ports["u"]),
                                    native=native.reshape(-1))
    elif port == "m":
        res = lin(ports["f"], module.mlp.down_proj, native, proof="P10")
    elif port == "h_out":
        res = finite.residual_add(_vec(ports["r"]), _vec(ports["m"]), native=native.reshape(-1))
    else:
        raise ValueError(f"no identity registered for port {port!r}")
    entry = finite.IDENTITIES.get(res.proof, {}).get("functions", {}).get(res.name, {})
    info = _info(caps, layer, spec.name)
    out: dict[str, Any] = {
        "schema": "mdb-explain-port-v1",
        "layer": layer,
        "port": spec.name,
        "operator": spec.operator,
        "proofs_declared": list(spec.proofs),
        "proof": res.proof,
        "function": res.name,
        "kind": res.kind,
        "section": entry.get("section"),
        "producers": list(producers),
        "reconstruction": res.total,
        "native": native,
        "native_dtype": str(native.dtype),
        "value_kind": info["kind"],
        "under_edits": [dict(e) for e in info["upstream_edits"]],
        "state": "resolved",
        "reason": None,
        "against": "native",
        "notes": notes,
        "result": res,
    }
    if info["state"] == "refused":
        # the value rests on a replica whose check failed: no claim about it
        res.native = None
        out.update(state="refused", reason=info["reason"], against="replica",
                   replica_equal=info["replica_equal"], replica_checked=info["replica_checked"])
        return out
    target = res.reference if res.reference is not None else res.total

    def gap(value, against=None):
        against = f64(target) if against is None else against
        return float((f64(value).reshape(f64(res.native).shape) - against).abs().max())

    if info["kind"] in ("edit", "realisation"):
        # An edit or a realisation wrote this value: it is not a native output, so ``eta`` (native
        # minus the identity on the captured operands) does not exist for it. The displacement of
        # the written value from the identity is the edit's, not a rounding remainder.
        delta = gap(native)
        original = caps.originals[layer].get(spec.name) if layer < len(caps.originals) else None
        out.update(
            authority=info["authority"],
            edit={"by": info["by"], "ops": list(info["ops"]), "kind": info["kind"]},
            edit_delta_max_abs=num(delta, MEASURED, res.proof),
            edit_delta_meaning="max |written value - identity on the captured operands|: the "
                               "displacement the edit or realisation produced, not a remainder",
        )
        against = None
        if spec.name == "r" and original is not None:
            # r's operand o is itself the realisation of the r edit: the original r is compared
            # with the add of the ORIGINAL operands (h + pre-edit o), or not at all
            held = caps.originals[layer].get("o")
            against = None if held is None else f64(ports["h"]).reshape(-1) + _vec(held)
            original = None if held is None else original
        if original is not None:
            out["original"] = {"eta_max_abs": num(gap(original, against), MEASURED, res.proof),
                               "meaning": "native output of the pre-edit value minus the identity "
                                          "on the captured operands (the remainder eta)"}
        else:
            notes.append("pre-edit original not retained: no native eta for this value "
                         "(captures_from_run(recompute_originals=True) rebuilds it)")
        res.native = None  # the native-remainder fields of summary() are not defined here
        notes.append("value written by an edit/realisation: no eta; see edit_delta_max_abs")
        return out
    eta_authority = EXACT if info["kind"] == "replica" else MEASURED
    if info["kind"] == "replica":
        out.update(against="replica", replica_equal=info["replica_equal"],
                   replica_checked=info["replica_checked"])
        notes.append(f"{spec.name} is a replica computed from captured operands (for s: qr, kr), "
                     "not a native tensor: its comparison against native is derived-exact")
        if info["replica_equal"] is None:
            notes.append("replica check not run at this cursor: the comparison is unchecked")
    res.eta_authority = eta_authority
    out["eta_max_abs"] = num(res.eta_max_abs, eta_authority, res.proof)
    out["eta_rel"] = num(res.eta_rel, eta_authority, res.proof)
    return out
