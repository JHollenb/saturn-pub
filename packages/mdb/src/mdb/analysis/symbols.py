"""The symbol table at the cursor: one row per binding, with definition sites, lifetime, named
readers and status, and an ``Unbound`` row so the table sums to the whole.

A binding is a set of cache cells ``(row, layer, KV head, k|v)``. The engine-native binding is
a token row ``t<s>``: its value enters the cache at every layer (``t<s>.L*.k``,
``t<s>.L*.v``). A user symbol (``declare(run, name, addresses)``) is any set of cells (``t2``,
``t2.L0.v``, ``cache.L1.v[kv0,row=2]``); declared symbols must not overlap, and a token row
holds exactly the cells no declared symbol holds (an explicit cell set), so the bindings
partition the cells. A read ``(reader row, layer, head, source)`` belongs to the binding that
holds its value cell ``v``: the value is what the read carries into the head (P06). ``k``
cells set the weights; a binding of ``k`` cells only receives no read.

At the reader rows (the cursor's row by default) each binding gets, from the exact value mix
(P06/P07, ``analysis.reads``):

* ``mass``: the attention weight its reads receive, summed over heads and layers (``count``
  reads, ``mean_mass`` per read); each weight is labelled as ``Run.read_info`` labels it
  (``supplied`` under a ``p``/``s`` edit), the sum ``derived-exact`` (``supplied`` if any part
  is);
* ``to_consumer``: its exact contribution to the consumer (``derived-exact``): a margin or logit
  (default at ``logits``/``sample``: the margin between the row's two largest logits), a
  residual port, or with no consumer mid-row the norm of each read's share of its layer's ``o``.
  A head whose block of ``A`` (or whose ``o``) an edit wrote does not reach the consumer through
  its value mix: those reads are ``refused`` and so is the binding's total;
* ``readers``: the top few (row, layer, head) reads by contribution, each with mass and effect;
* ``lifetime``: a strip per reader row, one mark per layer (``legend``): ``#`` read with mass at
  or above ``min_mass`` (default 0.1, stated), ``.`` read below it, ``-`` the binding holds no
  value cell at that layer, ``?`` not computed;
* ``status``: ``edited`` (``Run.read_info`` of its ``k``/``v`` says an edit wrote it or an edit
  of its row lies upstream), else ``read`` (some read at or above ``min_mass``), ``live`` (none
  yet and some layer of the reader row has not computed its ``p``), ``unread``.

Content (optional, ``sets=``): each binding's write into the reader row, summed over its reads
(``w_B = sum p[h,s] W_o,h v[kappa(h),s]``, P06/P07), is read through the final-norm readout
``rho * gamma * W_U[t]`` (P19, ``rho`` at the native final state of the cursor's row) over
declared candidate sets (``Candidates``: e.g. countries, capitals, entities): per set the best
token, its margin over the runner-up and the rank of a declared true token (derived-exact,
float64), plus the vocabulary top 5 (derived-local, ``sae.logit_lens`` in float32). With
``content_layers=True`` a content-lifetime strip per set: per layer of the reader row, ``C``
when that layer's write names the binding's overall best token, ``c`` when it is in that
layer's top 3, ``.`` read but naming another, ``-`` no value cell of the binding at that layer.
``hops=`` (binding name -> ``{"native_delta", "direct_delta"}``) adds a ``hop`` field: the
measured native change of the consumer under a replacement of the binding, the change of its
direct path under the same replacement (derived-exact, from a table on the edited run) and the
indirect share ``1 - direct_delta / native_delta`` (the part routed through later layers).

``Unbound`` holds, per head, the mass of the bindings not on this page, and the part of the
consumer no listed binding's reads carry, split into its parts: the embedding, MLP (and head
bias) terms (``derived-exact``, P19), the attention not listed (the layers' native ``o`` read
exactly, minus the listed reads), any edit of the residual stream, and the measured remainder
with its rounding bound (``analysis.circuits``). ``accounting`` checks that the bindings
partition every head's mass (the sum over all bindings equals ``sum_s p``); a broken partition
is ``refused`` and so is ``Unbound``. With a project the table is sealed as ``symbol-table``
records, one per page of at most ``PAGE`` bindings.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass, field
from typing import Any

from mdb.core.kinds import SYMBOL_TABLE, procedure
from mdb.core.records import RecordError, seal_record

from .finite import BOUND, EXACT, MEASURED, SUPPLIED, f64, num
from .forecast import Margin, Refusal, run_session_id
from .reads import (
    Source,
    claim,
    computed,
    edited_block,
    first_row,
    head_mix,
    parse_source,
    readout,
    refused,
    size,
    source_label,
    unit_roundoff,
)

SCHEMA = "mdb-symbol-table-v1"
PAGE = 16
STATUSES = ("live", "read", "unread", "edited", "unresolved")
LEGEND = {"#": "read with mass at or above min_mass", ".": "read below min_mass",
          "-": "no value cell of this binding at that layer", "?": "not computed"}
_U32 = 2.0 ** -24
# one line per added column (the full reference: docs/SYMBOL-TABLE-SCHEMA.md)
COLUMN_NOTES = {
    "contrast": "sum over its reads of p (c.W_O v_s - c.W_O vbar_h): the effect against the "
                "head's own mean value (derived-exact P19)",
    "mask_delta": "sum over its reads of -contrast/(1-p): each read masked alone with "
                  "renormalisation, DIRECT path only, every other input held (derived-exact)",
    "share": "to_consumer / consumer_total; share_of_abs divides by |consumer_total|",
    "mass_nonsink": "mass renormalised per head over the non-sink sources (sink rows: "
                    "report.sink); mass and to_consumer stay exact",
    "sink_ratio": "t0 only: its value-vector norm over the median row's, median over layers",
    "causal": "status unknown (observed only) or a measured liveness result (analysis.liveness)",
}
_PORTS = ("k", "v")


@dataclass
class Binding:
    """A named set of cache cells: ``cells`` are ``Source`` items (row, layers, KV heads,
    ports)."""

    name: str
    kind: str
    cells: list[Source]
    addresses: list[str]
    _by_row: dict[int, list[Source]] | None = field(default=None, repr=False, compare=False)
    # the circuit-map slot that bound it (``circuit_map.bind``), and labels (review 2.7):
    # ``labels[name] = (value, authority)``, supplied or derived
    role: str | None = field(default=None, compare=False)
    slot: str | None = field(default=None, compare=False)
    map_digest: str | None = field(default=None, compare=False)
    labels: dict[str, Any] = field(default_factory=dict, compare=False)

    def at(self, row: int) -> list[Source]:
        """The cells of source row ``row`` (indexed once: a symbol can hold thousands of rows)."""
        if self._by_row is None:
            index: dict[int, list[Source]] = {}
            for cell in self.cells:
                index.setdefault(cell.row, []).append(cell)
            self._by_row = index
        return self._by_row.get(row, [])

    def whole(self, row: int) -> bool:
        """Whether the binding holds every k/v cell of ``row`` (a whole-row cell ``t<row>``)."""
        return any(c.layers is None and c.kv is None and c.ports is None for c in self.at(row))

    def covers(self, row: int, layer: int, kv: int, port: str) -> bool:
        return any(c.covers(layer, kv, port) for c in self.at(row))

    @classmethod
    def component(cls, run: Any, name: str, cell: str, basis: Any, *,
                  features: list[int] | None = None) -> Any:
        """A sub-cell binding: the value cell ``cell`` split over a residual feature basis into
        per-feature components, the decoder bias and the error (``analysis.components``). It
        holds no cells of its own (the partition of value cells is unchanged); its components
        add up to the cell's float64 image exactly."""
        from .components import component_binding

        return component_binding(run, name, cell, basis, features=features)

    @classmethod
    def subspace(cls, name: str, layer: int, kv: int, basis: Any, *, authority: str = "supplied",
                 provenance: dict[str, Any] | None = None, row: int | None = None) -> Any:
        """A sub-cell binding: a named subspace of the value cell ``cache.L<layer>.v[kv<kv>]``
        with its basis authority (``fitted`` / ``derived-exact`` / ``supplied``) and provenance
        (``analysis.subspace``). It holds no cells of its own."""
        from .subspace import subspace_binding

        return subspace_binding(name, layer, kv, basis, authority=authority,
                                provenance=provenance, row=row)


_DECLARED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def declared(run: Any) -> dict[str, Binding]:
    """The user symbols declared on ``run`` in this process, by name."""
    if run not in _DECLARED:
        _DECLARED[run] = {}
    return _DECLARED[run]


def declare(run: Any, name: str, addresses: list[str]) -> Binding:
    """Declare (or redeclare) symbol ``name`` over ``addresses``; ``Refusal`` when it overlaps
    another declared symbol, names a row the cursor's row cannot read or is not a cache row."""
    if not name or not name.replace("_", "").replace("-", "").isalnum() or name == "Unbound":
        raise Refusal(f"a symbol name is letters, digits, '_' or '-' (and not Unbound), not "
                      f"{name!r}")
    if not addresses:
        raise Refusal("a symbol needs at least one address")
    here = int(run.where.split(".")[0][1:])
    cells = [parse_source(run, text) for text in addresses]
    late = [c for c in cells if c.row > here]
    if late:
        raise Refusal(f"t{late[0].row} has not been computed at {run.where}")
    for cell in cells:
        if cell.layers is None or cell.row < first_row(run):
            continue
        for layer in cell.layers:
            for port in cell.ports or _PORTS:
                address = f"t{cell.row}.L{layer}.{port}"
                if not computed(run, address):
                    raise Refusal(f"{address} has not been computed at {run.where}")
    binding = Binding(name, "declared", cells, [c.text for c in cells])
    layers, kv = run.adapter.layers, run.adapter.num_kv_heads
    for other in declared(run).values():
        if other.name == name:
            continue
        for cell in cells:
            for layer in range(layers):
                for k in range(kv):
                    for port in _PORTS:
                        if cell.covers(layer, k, port) and other.covers(cell.row, layer, k,
                                                                         port):
                            raise Refusal(f"{name} overlaps symbol {other.name} at "
                                          f"cache.L{layer}.{port}[kv{k},row={cell.row}]")
    declared(run)[name] = binding
    return binding


@dataclass(frozen=True)
class Candidates:
    """A declared candidate token set for content readouts; ``truth`` is the token the caller
    expects (its rank is reported), or None. ``info``: the ``candidates.CandidateSet`` it was
    built from (words, pieces, provenance), when it was built from words."""

    name: str
    tokens: tuple[int, ...]
    truth: int | None = None
    info: Any = field(default=None, compare=False, repr=False)


def readout_basis(run: Any, row: int) -> tuple[float, Any]:
    """``(rho, gamma)`` of the final-norm readout at ``row`` (P19): ``rho`` from the native
    final state ``t<row>.L<last>.h_out``, ``gamma`` the final norm gain in float64."""
    adapter = run.adapter
    last = adapter.layers - 1
    h_last = f64(run.read(f"t{row}.L{last}.h_out")).reshape(-1)
    norm = getattr(adapter.backbone, adapter.spec.final_norm)
    rho = float((h_last.pow(2).mean() + float(norm.variance_epsilon)) ** -0.5)
    return rho, f64(norm.weight).reshape(-1)


def _scores(run: Any, write: Any, tokens: tuple[int, ...], basis: tuple[float, Any]) -> Any:
    """``rho * gamma . (W_U[t] * write)`` for the tokens (rows sliced, then cast; float64)."""
    torch = _torch()
    rho, gamma = basis
    weight = run.adapter.model.lm_head.weight
    index = torch.tensor(list(tokens), dtype=torch.long, device=weight.device)
    rows = f64(weight.index_select(0, index)).cpu()
    return rho * (rows * gamma) @ f64(write).reshape(-1)


def content(run: Any, write: Any, sets: list[Candidates], basis: tuple[float, Any], *,
            tokenizer: Any = None, top: int = 5, flags: bool = False) -> dict[str, Any]:
    """The content readout of one binding's write (see the module docstring). ``flags``: add
    ``truth`` and the candidate flags (``candidates.content_flags``) per set."""
    from .candidates import content_flags
    from .sae import logit_lens

    out: dict[str, Any] = {}
    for cand in sets:
        if len(cand.tokens) < 2:
            raise Refusal(f"candidate set {cand.name} needs at least two tokens")
        scores = _scores(run, write, cand.tokens, basis)
        order = scores.argsort(descending=True)
        best, second = int(order[0]), int(order[1])
        item: dict[str, Any] = {
            "best": cand.tokens[best],
            "score": num(float(scores[best]), EXACT, "P19"),
            "margin": num(float(scores[best] - scores[second]), EXACT, "P19"),
            "size": len(cand.tokens)}
        if tokenizer is not None:
            item["text"] = tokenizer.decode([cand.tokens[best]])
        if cand.truth is not None and cand.truth in cand.tokens:
            where = cand.tokens.index(cand.truth)
            item["truth_rank"] = int((scores > scores[where]).sum()) + 1
            item["names_truth"] = cand.tokens[best] == cand.truth
            if flags:
                item["truth"] = int(cand.truth)
        if flags:
            item["flags"] = content_flags(cand, cand.tokens[best], sets)
        out[cand.name] = item
    ids = logit_lens(run, write, top=top)
    out["vocab_top"] = {"tokens": ids, "authority": "derived-local",
                        "note": "float32 ranking of gamma * write over the vocabulary"}
    if tokenizer is not None:
        out["vocab_top"]["text"] = [tokenizer.decode([i]) for i in ids]
    return out


def _content_strip(run: Any, writes: dict[int, Any], layers: int, cand: Candidates,
                   best: int, basis: tuple[float, Any]) -> str:
    marks = []
    for layer in range(layers):
        if layer not in writes:
            marks.append("-")
            continue
        scores = _scores(run, writes[layer], cand.tokens, basis)
        order = [cand.tokens[int(i)] for i in scores.argsort(descending=True)[:3]]
        marks.append("C" if order[0] == best else "c" if best in order else ".")
    return "".join(marks)


def _silu_prime(x: Any) -> Any:
    torch = _torch()
    sig = torch.sigmoid(x)
    return sig * (1 + x * (1 - sig))


def _rfft_parts(weight_rows: Any, delta: Any, gw: Any = None) -> Any:
    """Parseval split of ``weight_rows @ delta`` over channel-DFT bins: (rows, bins) float64 whose
    row sums equal ``weight_rows @ delta`` (real rfft, x.y = (1/d) sum_k s_k Re(conj(X_k) Y_k),
    s_k = 1 at DC and Nyquist, else 2). ``gw``: the rows' rfft when already taken."""
    torch = _torch()
    d = int(delta.numel())
    if gw is None:
        gw = torch.fft.rfft(weight_rows, dim=1)
    dn = torch.fft.rfft(delta.reshape(-1))
    scale = torch.full((gw.shape[1],), 2.0, dtype=torch.float64)
    scale[0] = 1.0
    if d % 2 == 0:
        scale[-1] = 1.0
    return (gw.conj() * dn[None, :]).real * (scale / d)[None, :]


@dataclass(frozen=True)
class CellTarget:
    """The content of one value cell (row ``row``, layer ``layer``, KV head ``kv``) as a scalar:
    ``C(v) = d . W_g v`` with ``W_g = sum_{h in kappa^-1(g)} W_O,h`` and
    ``d = gamma_F * (W_U[a] - W_U[b])`` (the cell's write at unit mass through the final-norm
    readout, per unit final rho; P06/P07, P19)."""

    row: int
    layer: int
    kv: int
    a: int
    b: int

    @property
    def text(self) -> str:
        return f"cell t{self.row}.L{self.layer}.v[kv{self.kv}] M[{self.a},{self.b}]"


def _cell_direction(run: Any, target: CellTarget) -> Any:
    """``W_g^T d`` (head_dim, float64)."""
    from .circuit_map import group_writers

    adapter = run.adapter
    final = getattr(adapter.backbone, adapter.spec.final_norm)
    w_u = adapter.model.lm_head.weight
    d = f64(final.weight).reshape(-1) * (f64(w_u[target.a]) - f64(w_u[target.b])).cpu()
    from .reads import layer_module
    o_weight = layer_module(adapter, target.layer).self_attn.o_proj.weight
    w_g = group_writers(o_weight, adapter.num_heads, adapter.num_kv_heads)[target.kv]
    return w_g.T @ d


def cell_value(run: Any, target: CellTarget, direction: Any = None) -> float:
    """``C(v)`` of the cell as the run's cache holds it (float64)."""
    direction = _cell_direction(run, target) if direction is None else direction
    v = f64(run.read_cache(target.layer, "v", staged=False))[0, target.kv, target.row]
    return float(direction @ v.reshape(-1))


def cell_readout(run: Any, target: CellTarget) -> Any:
    """A ``Readout`` (kind ``cell``) of ``C`` as an affine functional of the residual entering
    the cell's layer at its row: with no v norm, ``v_g = W_V^g (rho gamma_l * h) + b_V^g`` with
    ``h = t<row>.L<layer>.h`` and ``rho`` its input-norm scale, so
    ``C = c . h + bias`` with ``c = rho_0 gamma_l * (W_V^g)^T W_g^T d`` at the native rho_0 and
    ``bias = d . W_g b_V^g`` (0 without a V bias, e.g. Qwen3; Qwen2 has one). The bias cancels
    in every difference, so ``hop_edge`` uses ``c`` alone (derived-exact). ``native`` is ``C``
    read from the cache cell."""
    from .reads import Readout, layer_module

    module = layer_module(run.adapter, target.layer)
    if getattr(module.self_attn, "v_norm", None) is not None:
        raise Refusal("cell readout: the model normalises V (not an affine functional)")
    dh = run.adapter.head_dim
    direction = _cell_direction(run, target)
    rows = slice(target.kv * dh, (target.kv + 1) * dh)
    w_v = f64(module.self_attn.v_proj.weight)[rows]                                   # (dh, d)
    b_v = getattr(module.self_attn.v_proj, "bias", None)
    bias = 0.0 if b_v is None else float(direction @ f64(b_v)[rows])
    norm = module.input_layernorm
    gamma, eps = f64(norm.weight).reshape(-1), float(norm.variance_epsilon)
    h = f64(run.read(f"t{target.row}.L{target.layer}.h")).reshape(-1)
    rho = float((h.pow(2).mean() + eps) ** -0.5)
    vector = rho * gamma * (w_v.T @ direction)
    return Readout("cell", target.text, target.row, target, layer=target.layer, port="h",
                   vector=vector, rho=rho, bias=bias,
                   native=cell_value(run, target, direction), native_authority=MEASURED,
                   premises=["v = W_V^g n + b_V^g with n = rho gamma h: no v norm (checked); "
                             "the V bias enters only as the constant bias",
                             "rho of the cell layer's input norm at the native state"])


def hop_edge(run: Any, edited: Any, reader: Any, *, cell_layer: int, top_neurons: int = 8,
             spectral: bool = False, top_bins: int = 32, seed: int = 0,
             cache: dict | None = None, neurons: bool = True) -> dict[str, Any]:
    """The exact hop ladder of a finite cell edit at the consumer's row (advisor memo 4.2).

    With a ``cell`` reader (``cell_readout``) the consumer is a value cell's content at an
    intermediate row and layer: the ladder runs over layers ``cell_layer .. reader.layer - 1``
    at that row, the native delta is the change of ``C`` read from the two caches, and the
    norm term uses the cell layer's input norm (advisor memo 2026-11-14 §1.6).

    ``run`` is the native run and ``edited`` the same prompt with the cell edit staged and run
    natively to the consumer's logits; ``reader`` the native scalar ``Readout`` (``c``, native
    rho). ``Delta M_native = c.delta + sum_l c.Delta m_l + sum_{l>cell} c.Delta o_l + Delta_norm
    + Delta_round``: ``delta`` = the change of the cell layer's attention write at the row,
    ``Delta m_l`` / ``Delta o_l`` the captured MLP and attention-write changes (re-reads), the
    norm term ``(rho_1 - rho_0) w . h_out,1`` (exact finite RMS identity) and the rounding the
    measured remainder. Per neuron (finite SwiGLU identity, exact): ``Delta f_i = b0 Delta s +
    s0 Delta b + Delta s Delta b`` on the captured ``sg`` and ``u``; effect ``c . W_down[:, i]
    Delta f_i`` (P15); the top ``top_neurons`` by |effect| per layer are listed and the rest
    summed. ``linear`` is the one-layer Jacobian term on the layer's actual input change (P14,
    derived-local), never the effect; ``nonlinear = exact - linear``. With ``spectral`` the
    ``Delta a`` of every neuron is split over channel-DFT bins by Parseval and the top
    ``top_bins`` bins' share of the |effect|-weighted credit is reported against the same share
    under a random channel permutation and against the neuron basis at the same unit fraction.
    ``cache``: a dict the caller keeps across calls on the same model; it holds, per layer, the
    consumer's ``c . W_down`` (keyed by the bytes of ``c``) and the spectral permutation. The
    float64 gate/up weights (and their channel rfft) are cast for one layer at a time and
    dropped before the next, so the float64 peak is one layer's matrices, not every layer's
    (the same full-matrix float64 operations, so the same values).
    ``neurons=False``: the layer terms only (direct, captured MLP writes, re-reads, norm,
    rounding) from the two runs' ports; no MLP weight is read (no float64 weight copies), so the
    per-neuron split, ``exact``/``linear`` per layer and ``spectral`` are omitted.
    """
    torch = _torch()
    from .reads import layer_module

    if reader is None or not (reader.scalar or reader.kind == "cell"):
        raise Refusal("hop_edge needs a scalar consumer (a margin, a logit or a cell)")
    j, layers = reader.row, run.adapter.layers
    top_layer = reader.layer if reader.kind == "cell" else None
    end = layers if top_layer is None else top_layer
    if not 0 <= cell_layer < end:
        raise Refusal(f"cell layer {cell_layer} is not below the consumer's layer {end}")
    c = f64(reader.vector).reshape(-1)

    def port(r, layer, name):
        return f64(r.read(f"t{j}.L{layer}.{name}")).reshape(-1)

    def margin(r):
        if top_layer is not None:
            return cell_value(r, reader.target)
        logits = r.read(f"t{j}.logits").reshape(-1)
        t = reader.target
        return float(f64(logits[t.a]) - f64(logits[t.b])) if isinstance(t, Margin) else \
            float(f64(logits[t.i]))

    native_delta = margin(edited) - margin(run)
    direct = float(c @ (port(edited, cell_layer, "o") - port(run, cell_layer, "o")))
    upstream = float(c @ (port(edited, cell_layer, "h") - port(run, cell_layer, "h")))
    gen = torch.Generator().manual_seed(seed)
    cache = {} if cache is None else cache

    c_key = hashlib.sha256(c.contiguous().numpy().tobytes()).hexdigest()

    def consumer_down(layer, module):
        """``c @ W_down`` (float64, full matrix), the only use of ``W_down`` here."""
        key = ("cd", layer, c_key)
        if key not in cache:
            cache[key] = c @ f64(module.mlp.down_proj.weight)   # the cast is dropped here
        return cache[key]

    def spectra(layer, gate):
        key = ("perm", layer, seed)
        if key not in cache:
            cache[key] = torch.randperm(gate.shape[1], generator=gen)
        perm = cache[key]
        return torch.fft.rfft(gate, dim=1), perm, torch.fft.rfft(gate[:, perm], dim=1)
    mlp, reread, closure_sum = [], [], direct + upstream
    for layer in range(cell_layer, end):
        module = layer_module(run.adapter, layer)
        if layer > cell_layer:
            d_o = float(c @ (port(edited, layer, "o") - port(run, layer, "o")))
            reread.append({"layer": layer, "effect": num(d_o, EXACT, "P07"),
                           "meaning": "c . Delta o at the consumer row: later heads re-reading "
                                      "changed rows, not an MLP path"})
            closure_sum += d_o
        if not neurons:
            captured = float(c @ (port(edited, layer, "m") - port(run, layer, "m")))
            closure_sum += captured
            mlp.append({"layer": layer, "captured": num(captured, MEASURED)})
            continue
        g0, g1 = port(run, layer, "g"), port(edited, layer, "g")
        s0, s1 = port(run, layer, "sg"), port(edited, layer, "sg")
        b0, b1 = port(run, layer, "u"), port(edited, layer, "u")
        ds, db = s1 - s0, b1 - b0
        terms = torch.stack([b0 * ds, s0 * db, ds * db])           # (3, d_ff)
        df = terms.sum(0)
        gate, up = f64(module.mlp.gate_proj.weight), f64(module.mlp.up_proj.weight)
        cd = consumer_down(layer, module)                           # (d_ff,)
        effect = cd * df
        exact = float(effect.sum())
        captured = float(c @ (port(edited, layer, "m") - port(run, layer, "m")))
        closure_sum += captured
        # one-layer Jacobian on the actual input change (P14)
        r0, r1 = port(run, layer, "r"), port(edited, layer, "r")
        norm = module.post_attention_layernorm
        gamma_l, eps = f64(norm.weight).reshape(-1), float(norm.variance_epsilon)
        rho0 = float((r0.pow(2).mean() + eps) ** -0.5)
        dr = r1 - r0
        dn_lin = gamma_l * (rho0 * dr - (rho0 ** 3 / dr.numel()) * float(r0 @ dr) * r0)
        da_lin, db_lin = gate @ dn_lin, up @ dn_lin
        linear = float((cd * (b0 * _silu_prime(g0) * da_lin + s0 * db_lin)).sum())
        dn_true = port(edited, layer, "n") - port(run, layer, "n")
        den = float(dn_true.norm() * dn_lin.norm())
        da_true = g1 - g0
        da_den = float(da_true.norm() * da_lin.norm())
        order = effect.abs().argsort(descending=True)
        top = order[:top_neurons]
        rest = order[top_neurons:]
        listed = [{"name": f"L{layer}.f{int(i)}", "da": num(float(g1[i] - g0[i]), EXACT),
                   "db": num(float(db[i]), EXACT), "df": num(float(df[i]), EXACT),
                   "b0_ds": num(float(terms[0, i]), EXACT),
                   "s0_db": num(float(terms[1, i]), EXACT),
                   "ds_db": num(float(terms[2, i]), EXACT),
                   "effect": num(float(effect[i]), EXACT, "P15")} for i in top]
        entry = {"layer": layer,
                 "exact": num(exact, EXACT, "P15"),
                 "captured": num(captured, MEASURED),
                 "down_rounding": num(captured - exact, MEASURED),
                 "linear": num(linear, "derived-local", "P14"),
                 "nonlinear": num(exact - linear, EXACT),
                 "shared_gate": num(float((g1 - g0).mean()), EXACT),
                 "input_linearity": {
                     "cos_dn": num(float(dn_true @ dn_lin) / den if den else 0.0, EXACT),
                     "norm_ratio_dn": num(float(dn_lin.norm() / dn_true.norm())
                                          if float(dn_true.norm()) else 0.0, EXACT),
                     "cos_da": num(float(da_true @ da_lin) / da_den if da_den else 0.0, EXACT),
                     "meaning": "the Jacobian input change (J_RMS Delta r, G J_RMS Delta r) "
                                "against the captured Delta n and Delta a"},
                 "neurons": listed,
                 "omitted": {"count": int(rest.numel()),
                             "effect": num(float(effect[rest].sum()), EXACT, "P15")},
                 "top_share": num(float(effect[top].abs().sum() / effect.abs().sum())
                                  if float(effect.abs().sum()) else 0.0, EXACT)}
        if spectral:
            dn = dn_true
            wabs = effect.abs()
            gw, perm, pgw = spectra(layer, gate)
            parts = _rfft_parts(gate, dn, gw)                        # (d_ff, bins)
            credit = (wabs[:, None] * parts.abs()).sum(0)
            pparts = _rfft_parts(None, dn[perm], pgw)
            pcredit = (wabs[:, None] * pparts.abs()).sum(0)
            bins = credit.numel()
            k_neurons = max(1, round(top_bins / bins * effect.numel()))
            spec = torch.fft.rfft(dn).abs().pow(2)
            pspec = torch.fft.rfft(dn[perm]).abs().pow(2)

            def share(x, k):
                return float(x.sort(descending=True).values[:k].sum() / x.sum()) if float(
                    x.sum()) else 0.0
            entry["spectral"] = {
                "bins": bins, "top_bins": top_bins, "neurons_at_same_fraction": k_neurons,
                "parseval_max_error": num(float((parts.sum(1) - (gate @ dn)).abs().max()),
                                          MEASURED),
                "credit_top_bins_share": num(share(credit, top_bins), EXACT),
                "credit_top_bins_share_permuted": num(share(pcredit, top_bins), EXACT),
                "neuron_top_share_same_fraction": num(share(wabs, k_neurons), EXACT),
                "energy_top_bins_share": num(share(spec, top_bins), EXACT),
                "energy_top_bins_share_permuted": num(share(pspec, top_bins), EXACT),
                "meaning": "Parseval split of each neuron's Delta a over channel-rfft bins of "
                           "Delta n, weighted by |effect|; permuted = same with a random channel "
                           "order; neuron share = top neurons at the same unit fraction"}
        mlp.append(entry)
    if top_layer is None:
        last = layers - 1
        h1, h0 = port(edited, last, "h_out"), port(run, last, "h_out")
        final = getattr(run.adapter.backbone, run.adapter.spec.final_norm)
    else:
        h1, h0 = port(edited, top_layer, "h"), port(run, top_layer, "h")
        final = layer_module(run.adapter, top_layer).input_layernorm
    eps_f = float(final.variance_epsilon)
    rho_0 = float((h0.pow(2).mean() + eps_f) ** -0.5)
    rho_1 = float((h1.pow(2).mean() + eps_f) ** -0.5)
    norm_term = (rho_1 - rho_0) / rho_0 * float(c @ h1)
    residual_gap = float(c @ (h1 - h0)) - closure_sum
    total = closure_sum + norm_term
    mlp_total = sum(e["captured"]["value"] for e in mlp)
    reread_total = sum(e["effect"]["value"] for e in reread)

    def share_of(x):
        if not native_delta:
            return {"state": "unresolved", "reason": "native delta 0"}
        return num(x / native_delta, MEASURED)
    return {
        "consumer": {"native_delta": num(native_delta, MEASURED), "kind": reader.kind,
                     "text": reader.text},
        "direct": num(direct, EXACT, "P19"),
        "upstream": num(upstream, EXACT, "P07"),
        "mlp": mlp, "reread": reread,
        "norm": num(norm_term, EXACT),
        "residual_add_rounding": num(residual_gap, MEASURED),
        "rounding": num(native_delta - total - residual_gap, MEASURED),
        "closure": {"sum_of_terms": num(total + residual_gap, MEASURED),
                    "gap": num(native_delta - total - residual_gap, MEASURED)},
        "shares": {"direct": share_of(direct), "mlp": share_of(mlp_total),
                   "reread": share_of(reread_total), "norm": share_of(norm_term),
                   "meaning": "each exact term over the native delta; signed"},
        "totals": {"mlp": num(mlp_total, MEASURED), "reread": num(reread_total, EXACT),
                   "mlp_exact": num(sum(e["exact"]["value"] for e in mlp), EXACT) if neurons
                   else {"state": "not computed", "reason": "neurons=False"},
                   "mlp_linear": num(sum(e["linear"]["value"] for e in mlp), "derived-local")
                   if neurons else {"state": "not computed", "reason": "neurons=False"}},
    }


def gate_switch_forecast(run: Any, edited: Any, reader: Any, *, cell_layer: int,
                         ks: tuple[int, ...] = (8, 32, 128), switch_layers: Any = None,
                         cache: dict | None = None) -> dict[str, Any]:
    """First-order forecasts of a cell edit's consumer change, with and without exact gate
    switches (advisor memo 2026-11-14-open-problems §2.3; the ``hop_edge`` ladder's ``linear``).

    Same runs as ``hop_edge`` (``run`` native, ``edited`` with the cell edit, both run to the
    consumer's logits; ``reader`` a margin or logit ``Readout``). Per MLP layer ``l >= cell_layer``
    at the consumer row, from the base run's ports ``r0, g0, u0, sg0`` and the weights:

    * ``F1`` = ``direct + sum_l c . W_down df_lin``: ``df_lin = b0 silu'(g0) da + s0 db`` with
      ``(da, db) = (W_gate, W_up) J_RMS(r0) Delta r`` on the layer's ACTUAL input change (P14,
      derived-local): the hop-math first-order total (direct + linear MLP).
    * ``F2[k]`` = F1 with the ``k`` neurons per layer of largest ``|da|`` given the exact finite
      SwiGLU of their Jacobian-predicted input: ``silu(g0 + da)(b0 + db) - s0 b0`` (no residual
      recomputation; the inputs are still the actual ``Delta r``).
    * ``F3[k]`` = the chained, passive version: the residual change at the row starts as the
      cell layer's own attention write ``Delta o`` (exact, the direct term) and each layer's MLP
      forecast (``F2``-style with ``k`` exact gates) is fed forward into the next layer's input;
      later attention re-reads are not forecast (zero), so nothing native downstream of the edit
      is used. ``F3`` = ``c . (Delta o_cell + sum_l Delta m_l^F3)``.

    ``switch_layers`` (default every layer from ``cell_layer``) restricts where the exact gates
    are applied (elsewhere ``k = 0``). Returns the three forecasts, the native delta, the per-layer
    ``F1``/``F2`` MLP terms, and the per-layer count of base-run gate sign switches
    (``sign(g1) != sign(g0)``) as measured on the edited run."""
    torch = _torch()
    from .reads import layer_module

    if reader is None or not reader.scalar:
        raise Refusal("gate_switch_forecast needs a scalar consumer (a margin or a logit)")
    j, layers = reader.row, run.adapter.layers
    if not 0 <= cell_layer < layers:
        raise Refusal(f"cell layer {cell_layer} outside 0..{layers - 1}")
    c = f64(reader.vector).reshape(-1)
    t = reader.target
    del cache  # accepted for compatibility; nothing is cached (float64 weights per layer)

    def port(r, layer, name):
        return f64(r.read(f"t{j}.L{layer}.{name}")).reshape(-1)

    def margin(r):
        lg = r.read(f"t{j}.logits").reshape(-1)
        return float(f64(lg[t.a]) - f64(lg[t.b])) if isinstance(t, Margin) else float(f64(lg[t.i]))

    def weights(layer, module):
        """This layer's float64 gate/up/down (cast per layer, never kept across layers)."""
        return (f64(module.mlp.gate_proj.weight), f64(module.mlp.up_proj.weight),
                f64(module.mlp.down_proj.weight))

    allowed = set(range(cell_layer, layers) if switch_layers is None else switch_layers)
    native_delta = margin(edited) - margin(run)
    d_o = port(edited, cell_layer, "o") - port(run, cell_layer, "o")
    direct = float(c @ d_o)
    f1, f2 = direct, {k: direct for k in ks}
    chain = {k: d_o.clone() for k in (0, *ks)}          # Delta residual (F3), per k
    rows = []
    for layer in range(cell_layer, layers):
        module = layer_module(run.adapter, layer)
        gate, up, down = weights(layer, module)
        norm = module.post_attention_layernorm
        gamma_l, eps = f64(norm.weight).reshape(-1), float(norm.variance_epsilon)
        r0 = port(run, layer, "r")
        g0, b0, s0 = port(run, layer, "g"), port(run, layer, "u"), port(run, layer, "sg")
        rho0 = float((r0.pow(2).mean() + eps) ** -0.5)
        cd = c @ down

        def mlp_change(dr, k):
            dn = gamma_l * (rho0 * dr - (rho0 ** 3 / dr.numel()) * float(r0 @ dr) * r0)
            da, db = gate @ dn, up @ dn
            df = b0 * _silu_prime(g0) * da + s0 * db
            if k and layer in allowed:
                top = da.abs().argsort(descending=True)[:k]
                g1 = g0[top] + da[top]
                df[top] = torch.nn.functional.silu(g1) * (b0[top] + db[top]) - s0[top] * b0[top]
            return df

        dr = port(edited, layer, "r") - r0
        df1 = mlp_change(dr, 0)
        term1 = float(cd @ df1)
        f1 += term1
        entry = {"layer": layer, "F1": num(term1, "derived-local", "P14"), "F2": {}}
        for k in ks:
            term = float(cd @ mlp_change(dr, k))
            f2[k] += term
            entry["F2"][str(k)] = num(term, "derived-local")
        for k, vec in chain.items():                  # re-reads of later layers: not forecast
            chain[k] = vec + down @ mlp_change(vec, k)
        g1n = port(edited, layer, "g")
        entry["gate_sign_switches"] = int(((g1n > 0) != (g0 > 0)).sum())
        entry["captured"] = num(float(c @ (port(edited, layer, "m") - port(run, layer, "m"))),
                                MEASURED)
        rows.append(entry)

    def ratio(x):
        return num(x / native_delta, MEASURED) if native_delta else None
    f3 = {k: float(c @ v) for k, v in chain.items()}
    return {"consumer": {"native_delta": num(native_delta, MEASURED), "text": reader.text},
            "direct": num(direct, EXACT, "P19"),
            "F1": num(f1, "derived-local"), "F1_ratio": ratio(f1),
            "F2": {str(k): num(v, "derived-local") for k, v in f2.items()},
            "F2_ratio": {str(k): ratio(v) for k, v in f2.items()},
            "F3": {str(k): num(v, "derived-local") for k, v in f3.items()},
            "F3_ratio": {str(k): ratio(v) for k, v in f3.items()},
            "layers": rows,
            "meaning": "F1 first order on actual layer inputs (direct + linear MLP); F2[k] the k "
                       "largest |Jacobian Delta g| gates per layer exact; F3[k] chained from the "
                       "direct write, re-reads not forecast"}


def _token_rows(run: Any, here: int, users: list[Binding]) -> list[Binding]:
    """One binding per token row holding exactly the cells no declared symbol holds."""
    layers, kv = run.adapter.layers, run.adapter.num_kv_heads
    out = []
    for s in range(here + 1):
        if any(b.whole(s) for b in users):
            continue
        cells = [(layer, k, port) for layer in range(layers) for k in range(kv)
                 for port in _PORTS if not any(b.covers(s, layer, k, port) for b in users)]
        if not cells:
            continue
        if len(cells) == layers * kv * len(_PORTS):
            out.append(Binding(f"t{s}", "token", [Source(s, text=f"t{s}")],
                               [f"t{s}.L*.k", f"t{s}.L*.v"]))
            continue
        out.append(Binding(f"t{s}", "token",
                           [Source(s, (layer,), (k,), f"cache.L{layer}.{port}[kv{k},row={s}]",
                                   (port,)) for layer, k, port in cells],
                           [f"t{s} (uncovered part: {len(cells)} of "
                            f"{layers * kv * len(_PORTS)} k/v cells)"]))
    return out


def _default_consumer(run: Any) -> Margin | None:
    row, stop = run.where.split(".", 1)
    if stop not in ("logits", "sample"):
        return None
    logits = f64(run.read(f"{row}.logits")).reshape(-1)
    first, second = (int(i) for i in logits.topk(2).indices)
    return Margin(int(row[1:]), first, second)


def _status_edits(run: Any, row: int, layers: int, here: tuple[int, int]) -> list[str]:
    """The edits ``Run.read_info`` reports on this row's ``k``/``v`` (own or upstream in the
    row); a prefilled row has none."""
    if row < first_row(run):
        return []
    found: list[str] = []
    for layer in range(layers):
        for port in ("k", "v"):
            address = f"t{row}.L{layer}.{port}"
            if run.parse(address) > here:
                continue
            info = run.read_info(address)
            if info["edited"]:
                found.append(f"{address} ({','.join(info['ops'])})")
            for edit in info["upstream_edits"]:
                if int(edit["row"]) == row:
                    found.append(f"t{row}.L{edit['layer']}.{edit['port']} ({edit['op']})")
    return list(dict.fromkeys(found))


@dataclass
class Table:
    """A symbol table: ``report`` (bounded JSON), ``records`` (sealed, one per page) and the
    float64 ``tensors`` behind it: ``mass[(row, layer)]`` (bindings x heads) with
    ``unbound_mass`` and ``total_mass`` per head, ``contribution`` per binding."""

    report: dict[str, Any]
    records: list[dict[str, Any]]
    tensors: dict[str, Any] = field(default_factory=dict)

    def to_frame(self, level: str = "binding", *, item: str | None = None,
                 foreign: bool = False) -> Any:
        """A long pandas DataFrame (``analysis.frame.to_frame``: binding, read or head)."""
        from .frame import to_frame

        return to_frame(self, level, item=item, foreign=foreign)

    def to_arrow(self, level: str = "binding", **kw: Any) -> Any:
        from .frame import to_arrow

        return to_arrow(self.to_frame(level, **kw))

    def to_parquet(self, path: str, level: str = "binding", **kw: Any) -> str:
        from .frame import to_parquet

        return to_parquet(self.to_frame(level, **kw), path)


def table(run: Any, *, consumer: Any = None, rows: str | None = None, min_mass: float = 0.1,
          readers: int = 3, offset: int = 0, limit: int = PAGE,
          session_id: str | None = None, tokenizer: Any = None,
          pending: dict[str, dict[str, Any]] | None = None,
          sets: list[Candidates] | None = None, content_layers: bool = False,
          hops: dict[str, dict[str, Any]] | None = None, sink: tuple[int, ...] | None = (0,),
          co: list[Any] | None = None, lexical_null: bool = False,
          switch: dict[str, str] | None = None, labels: dict[str, Any] | None = None,
          derive: dict[str, str] | None = None, foreign: bool = False,
          causal: bool | str = False, min_effect: float = 0.0,
          components: list[Any] | None = None, subspaces: list[Any] | None = None,
          neuronpedia: dict[str, Any] | None = None, store: Any = None) -> Table:
    """The symbol table at the cursor (see the module docstring); ``offset``/``limit`` page
    the bindings (declared symbols first, then token rows in order). ``pending``: declared
    symbols that cannot be bound on this run yet (name -> ``addresses``, ``reason``), listed
    ``unresolved``.

    Additions (review 2026-11-14; ``docs/SYMBOL-TABLE-SCHEMA.md`` lists every column):

    * ``sink``: sink rows for ``mass_nonsink`` (default ``(0,)``: t0; ``None`` or ``()``: off);
      token row 0 also gets ``sink_ratio``. ``contrast``, ``mask_delta`` and ``share`` are added
      under a scalar consumer.
    * ``sets`` built from words (``candidates.CandidateSet.candidates``) add ``truth`` and
      ``flags`` per set; ``lexical_null`` adds the truth token's own readout per set; ``co``
      (``candidates.CoReadout`` items) adds orthogonalised co-readouts with the whole subspace
      of another set removed; ``switch`` (target set -> set it switches on) adds the switched
      readout.
    * ``labels``: item-level labels (name -> value, ``supplied``), reported and attached to
      every binding; ``derive``: label name -> dotted path within a binding row (e.g.
      ``content.countries.best``), a derived label per binding.
    * ``foreign``: TransformerLens / nnsight / pyvene names per binding and per read.
    * ``causal`` (``True``, ``"mask"`` or ``"zero"``): the causal-liveness column
      (``analysis.liveness``), per binding on this page; without it every binding's ``causal``
      is ``unknown``.
    * ``subspaces``: ``subspace.SubspaceBinding`` items bound to a row: a ``subspaces`` section
      with basis authority, provenance, digest, the native cell's mass in the subspace, the
      direct consumer share of the in-subspace part (scalar consumer) and attached labels
      (e.g. measured IIA).
    * ``components``: ``components.ComponentBinding`` items (feature bindings): a separate
      ``components`` section with per-feature activation, direct effect, content, SAELens ids
      and a Neuronpedia link from the supplied ``neuronpedia`` id mapping.
    * A binding with more than 64 rows or definitions keeps the compact form for display and
      its full lists behind ``full`` (a sha256-checked JSON blob in ``store``, else held in
      memory on the run; ``core.bounded.load_json`` reads it).
    """
    from .reads import reader_rows

    if not 0 <= min_mass <= 1:
        raise Refusal("min_mass must be in 0..1")
    if not 1 <= limit <= PAGE:
        raise Refusal(f"limit must be in 1..{PAGE}")
    here_key = run.parse(run.where)
    layers = run.adapter.layers
    found_rows = reader_rows(run, Source(0), rows)
    if consumer is None:
        consumer = _default_consumer(run)
    reader = None if consumer is None else readout(run, consumer)
    users = list(declared(run).values())
    bindings = users + _token_rows(run, here_key[0], users)
    n = len(bindings)
    sink = tuple(sink or ())
    grid = _Grid(run, bindings, reader, found_rows, keep_writes=bool(sets), sink=sink)
    names = None
    if foreign:
        names = {"head_dim": run.adapter.head_dim, "kv_heads": run.adapter.num_kv_heads,
                 "layers": layers}
    item_labels = {k: label_claim(v) for k, v in (labels or {}).items()}
    entries = [_entry(run, binding, b, grid, reader, found_rows, layers, min_mass, readers,
                      here_key, tokenizer, names, item_labels)
               for b, binding in enumerate(bindings)]
    shown = entries[offset:offset + limit]
    if sink and 0 in sink:
        for b in range(offset, min(offset + limit, n)):
            if bindings[b].kind == "token" and bindings[b].cells[0].row == 0:
                entries[b]["view"]["sink_ratio"] = _sink_ratio(run, here_key[0])
    flagged = any(getattr(c, "info", None) is not None for c in (sets or ()))
    if flagged:
        for b in range(offset, min(offset + limit, n)):
            view = entries[b]["view"]
            if "token" in view:
                view["first_subword_of"] = sorted(
                    c.name for c in sets if getattr(c, "info", None) is not None
                    and c.info.multi_piece(view["token"]))
    if sets:
        basis = readout_basis(run, here_key[0])
        co_spaces = _subspaces(run, sets, co or (), basis)
        for b in range(offset, min(offset + limit, n)):
            per_layer: dict[int, Any] = {}
            for (j, layer), w in grid.write.items():
                if bool(grid.cover[(j, layer)][b].any()):
                    per_layer[layer] = per_layer.get(layer, 0) + w[b]
            view = entries[b]["view"]
            if not per_layer:
                view["content"] = {"state": "unresolved", "reason": "no read of this binding"}
                continue
            total = sum(per_layer.values())
            view["content"] = content(run, total, sets, basis, tokenizer=tokenizer,
                                      flags=flagged or lexical_null or bool(co) or bool(switch))
            view["content"]["meaning"] = ("readout of the binding's write into the reader "
                                          "row(s), summed over its reads, through rho * gamma * "
                                          "W_U (P19)")
            _content_extras(run, view["content"], total, sets, basis, co or (), co_spaces,
                            lexical_null, switch or {})
            if content_layers:
                view["content_lifetime"] = {
                    c.name: _content_strip(run, per_layer, layers, c, view["content"][c.name]["best"],
                                           basis) for c in sets}
    for b in range(offset, min(offset + limit, n)):
        view = entries[b]["view"]
        hop = (hops or {}).get(view["symbol"])
        if hop is None:
            continue
        native = float(hop["native_delta"])
        direct = float(hop["direct_delta"])
        view["hop"] = {"native_delta": num(native, MEASURED),
                       "direct_delta": num(direct, EXACT, "P19"),
                       "indirect_share": (num(1.0 - direct / native, MEASURED) if native else
                                          {"state": "unresolved", "reason": "native delta 0"}),
                       "replacement": hop.get("replacement", "supplied by the caller")}
        if hop.get("edge") is not None:
            view["hop"]["edge"] = hop["edge"]
    for b in range(offset, min(offset + limit, n)):
        view = entries[b]["view"]
        for name, path in (derive or {}).items():
            try:
                from .summary import leaf

                value = leaf(view, path)
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            view.setdefault("labels", {})[name] = label_claim(value, "derived")
    page_bindings = [bindings[b] for b in range(offset, min(offset + limit, n))]
    for binding, b in zip(page_bindings, range(offset, min(offset + limit, n))):
        rows_all = sorted({c.row for c in binding.cells})
        if len(rows_all) > _LIST_CAP or len(binding.addresses) > _LIST_CAP:
            from mdb.core.bounded import put_json

            entries[b]["view"]["full"] = put_json(
                {"rows": rows_all, "definitions": list(binding.addresses)}, store=store, run=run)
    if causal:
        from .liveness import liveness

        form = "mask" if causal is True else str(causal)
        found_causal = liveness(run, page_bindings, reader, form=form, min_effect=min_effect)
        for b in range(offset, min(offset + limit, n)):
            entries[b]["view"]["causal"] = found_causal[bindings[b].name]
    else:
        from .liveness import UNKNOWN

        for b in range(offset, min(offset + limit, n)):
            entries[b]["view"]["causal"] = dict(UNKNOWN)
    listed = sorted(range(offset, min(offset + limit, n)))
    keys = sorted(grid.mass)
    unbound_mass = {key: grid.total[key] - grid.mass[key][listed].sum(0) for key in keys}
    report: dict[str, Any] = {
        "where": run.where, "rows": found_rows,
        "consumer": (reader.text if reader is not None else
                     "none: per-layer o (norm of each read's share); give consumer= or run "
                     "until logits"),
        "min_mass": num(float(min_mass), SUPPLIED),
        "mass_meaning": "a binding's mass is a sum over the heads and layers that read it "
                        "(count reads; mean_mass per read)",
        "attribution": "a read (row, layer, head, source) belongs to the binding holding its "
                       "value cell v",
        "legend": LEGEND,
        "columns": COLUMN_NOTES,
        "bindings": [e["view"] for e in shown],
        "page": {"offset": offset, "limit": limit, "total": n},
    }
    if pending:
        report["unresolved_symbols"] = [
            {"symbol": name, "kind": "declared", "definitions": item["addresses"],
             "status": "unresolved", "reason": item["reason"]}
            for name, item in sorted(pending.items())][:PAGE]
    broken = grid.partition_error()
    listed_sum = sum((float(grid.mass[k][listed].sum()) for k in keys), 0.0)
    unbound: dict[str, Any] = {
        "symbol": "Unbound",
        "mass": refused(broken) if broken else num(
            sum(float(unbound_mass[k].sum()) for k in keys), EXACT),
        "meaning": "attention mass of the bindings not on this page, per head summed; the "
                   "consumer part no listed binding's reads carry, by part",
    }
    contribution = {}
    if broken:
        unbound["readout"] = refused(broken)
    elif reader is not None and reader.scalar:
        listed_effect = sum(entries[b]["contribution"] for b in listed
                            if entries[b]["contribution"] is not None)
        unbound["readout"] = num(float(f64(reader.native)) - listed_effect, MEASURED)
        unbound["parts"] = _unbound_parts(run, reader, listed_effect)
        report["consumer_total"] = claim(f64(reader.native), reader.native_authority
                                         or MEASURED)
        contribution = {b: e["contribution"] for b, e in enumerate(entries)}
    else:
        unbound["readout"] = {"state": "unresolved",
                              "reason": "no scalar consumer: the per-layer o shares have no "
                                        "common total"}
    report["unbound"] = unbound
    report["accounting"] = {
        "heads": sum(int(grid.total[k].numel()) for k in keys),
        "listed_mass": num(listed_sum, EXACT),
        "max_head_deviation": num(grid.deviation, MEASURED),
        "deviation_bound": num(grid.bound, BOUND),
        "partition": refused(broken) if broken else {
            "max_gap": num(grid.gap, EXACT),
            "meaning": "per head, the sum over all bindings of their mass minus sum_s p "
                       "(float64 summation order only)"},
        "within_bound": grid.deviation <= grid.bound and not broken,
        "meaning": "per head, bindings + Unbound = sum_s p (native); |sum_s p - 1| against "
                   "T u_dtype + 2 T u_f32 (softmax in float32, each weight cast to the dtype)",
    }
    if grid.missing:
        report["not_computed"] = grid.missing[:16]
    if item_labels:
        report["labels"] = item_labels
    if derive:
        report["derived_labels"] = dict(derive)
    if sink:
        report["sink"] = [f"t{r}" for r in sink]
    if sets:
        report["candidate_sets"] = [
            {**c.info.describe(), "truth": c.truth} if getattr(c, "info", None) is not None
            else {"name": c.name, "size": len(c.tokens), "truth": c.truth,
                  "source": "token ids"} for c in sets][:PAGE]
    if components:
        report["components"] = [_component_row(run, cb, consumer, sets, tokenizer, neuronpedia)
                                for cb in components][:PAGE]
    if subspaces:
        from .subspace import table_row

        report["subspaces"] = [table_row(run, sb, consumer) for sb in subspaces][:PAGE]
    records = _records(run, report, session_id)
    report["records"] = [r["id"] for r in records]
    tensors = {"mass": grid.mass, "effect": grid.effect, "write": grid.write, "total_mass": grid.total,
               "unbound_mass": unbound_mass, "contribution": contribution,
               "contrast": grid.contrast, "mask_delta": grid.mask_delta,
               "sink_mass": grid.sink_mass, "nonsink_total": grid.nonsink_total,
               "cover": grid.cover, "sources": grid.sources, "supplied": grid.supplied,
               "bindings": [b.name for b in bindings], "listed": listed,
               "binding_objects": bindings}
    return Table(report, records, tensors)


def _sink_ratio(run: Any, here: int) -> dict[str, Any]:
    """Row 0's value norm over the median row's, per layer (mean over KV heads), summarised as
    the median over layers (derived-exact from the cache)."""
    torch = _torch()
    ratios = []
    for layer in range(run.adapter.layers):
        try:
            v = f64(run.read_cache(layer, "v", staged=False))[0, :, :here + 1]   # (kv, T, dh)
        except (ValueError, IndexError, RuntimeError):
            continue
        norms = v.norm(dim=-1).mean(0)                                         # (T,)
        if norms.numel() < 2:
            continue
        median = float(norms.median())
        if median > 0:
            ratios.append(float(norms[0]) / median)
    if not ratios:
        return {"state": "unresolved", "reason": "no cache layer with two rows"}
    ratios_t = torch.tensor(ratios, dtype=torch.float64)
    return num(float(ratios_t.median()), EXACT)


def _subspaces(run: Any, sets: list[Candidates], co: Any, basis: tuple[float, Any]) -> dict:
    from .candidates import subspace

    by_name = {c.name: c for c in sets}
    out = {}
    for spec in co:
        if spec.target not in by_name:
            raise Refusal(f"co-readout target {spec.target} is not a candidate set of the table")
        if spec.project_out is not None:
            if spec.project_out not in by_name:
                raise Refusal(f"co-readout projects out {spec.project_out}, not a candidate set "
                              "of the table")
            tokens = by_name[spec.project_out].tokens
        elif spec.tokens:
            tokens = tuple(spec.tokens)
        else:
            raise Refusal("a co-readout names project_out (a set) or tokens")
        out[spec] = subspace(run, tokens, basis[1], spec.k)
    return out


def _content_extras(run: Any, found: dict[str, Any], write: Any, sets: list[Candidates],
                    basis: tuple[float, Any], co: Any, subspaces: dict, lexical: bool,
                    switch: dict[str, str]) -> None:
    """Lexical nulls, co-readouts and switched readouts of one binding's content."""
    from . import candidates as cands

    by_name = {c.name: c for c in sets}
    if lexical:
        for c in sets:
            if c.truth is not None and c.truth in c.tokens:
                found[c.name]["lexical_null"] = cands.lexical_null(run, c, c.truth, basis)
    for spec in co:
        target = found[spec.target]
        target.setdefault("co", {})[spec.project_out or "tokens"] = cands.co_readout(
            run, write, by_name[spec.target], subspaces[spec], basis, spec)
    for target, on in switch.items():
        if target not in by_name or on not in by_name:
            raise Refusal(f"switch {target}:{on} names a set the table does not read")
        found[target]["switched"] = cands.switched(
            by_name[target], found[on], by_name[on],
            lambda tokens: _scores(run, write, tokens, basis))


def _component_row(run: Any, cb: Any, consumer: Any, sets: Any, tokenizer: Any,
                   neuronpedia: dict[str, Any] | None) -> dict[str, Any]:
    """One feature binding's section row (``components.component_table`` plus names)."""
    from mdb.core.foreign_names import ct_node, neuronpedia_url

    from .components import component_table

    if consumer is None:
        consumer = _default_consumer(run)
    row = component_table(run, cb, consumer, sets=sets, tokenizer=tokenizer)
    mass = row.get("mass")
    if isinstance(mass, dict) and "inherited" in mass:      # record-safe: a claim holds NUM_KEYS
        row["mass_meaning"] = mass.pop("inherited")
    ref = (cb.basis or {}).get("saelens") or {}
    for entry in row["components"]:
        if entry.get("feature") is None:
            continue
        f = int(entry["feature"])
        entry["saelens"] = {"release": ref.get("release"), "sae_id": ref.get("sae_id"),
                            "hook_name": ref.get("hook_name"), "feature": f}
        entry["neuronpedia"] = neuronpedia_url(neuronpedia, ref.get("release"),
                                               ref.get("sae_id"), f)
        if (cb.basis or {}).get("transcoder"):
            entry["ct_node"] = ct_node(cb.layer, cb.row, f)
    row["kind"] = "feature"
    return row


def _remainder_bound(rest: Any) -> dict[str, Any]:
    """The remainder's rounding bound, or a refused bound when its premise fails (no number)."""
    if rest.bound is None:
        return {"bound": {"state": "refused", "reason": "the rounding bound's accumulation "
                          "premise fails on this run (see `explain`)"}, "within": None}
    return {"bound": num(float(rest.bound), BOUND), "within": bool(rest.within)}


def _unbound_parts(run: Any, reader: Any, listed_effect: float) -> dict[str, Any]:
    """The consumer minus the listed bindings' reads, by part (see the module docstring)."""
    from .circuits import terms, top_gap

    listed = terms(run, reader, 1, 1)
    native = float(f64(reader.native))
    attention = sum(float(t.value) for t in listed if t.kind == "attn")
    other = sum(float(t.value) for t in listed if t.kind != "attn")
    tail = top_gap(run, reader, f64(native - attention - other))
    edits = [t for t in tail if t.kind == "edit"]
    rest = tail[-1]
    parts: dict[str, Any] = {
        "embedding_mlp": num(other, EXACT, "P19"),
        "attention_not_listed": num(attention - listed_effect, EXACT, "P19"),
        "remainder": {"value": num(float(rest.value), MEASURED), **_remainder_bound(rest)},
        "meaning": "embedding, MLP and head-bias terms; the layers' attention output (native o, "
                   "read exactly) minus the listed reads (reads of bindings off this page, "
                   "heads an edit wrote, o_proj rounding); residual-stream edits; the measured "
                   "remainder (residual-add, final-norm and head rounding) with its bound",
    }
    if edits:
        labels = {t.authority for t in edits}
        parts["residual_edits"] = num(sum(float(t.value) for t in edits),
                                      SUPPLIED if SUPPLIED in labels else EXACT)
    return parts


class _Grid:
    """Per (reader row, layer): each binding's mass, effect and reads per head, from the value
    mix; reads are owned by the binding holding their value cell."""

    def __init__(self, run: Any, bindings: list[Binding], reader: Any, rows: list[int], *,
                 keep_writes: bool = False, sink: tuple[int, ...] = ()):
        torch = _torch()
        self.sink = tuple(int(s) for s in sink)
        # scalar consumer only: per binding and head, the contrast-form effect and the
        # direct-path mask prediction (review 2.5); per binding and head the mass from sink rows
        self.contrast: dict[tuple[int, int], Any] = {}
        self.mask_delta: dict[tuple[int, int], Any] = {}
        self.sink_mass: dict[tuple[int, int], Any] = {}
        self.nonsink_total: dict[tuple[int, int], Any] = {}
        self.write: dict[tuple[int, int], Any] = {}   # (row, layer) -> (bindings, d), on request
        n, heads, layers = len(bindings), run.adapter.num_heads, run.adapter.layers
        self.mass: dict[tuple[int, int], Any] = {}
        self.effect: dict[tuple[int, int], Any] = {}
        self.total: dict[tuple[int, int], Any] = {}
        self.reach: dict[tuple[int, int], str | None] = {}
        self.cover: dict[tuple[int, int], Any] = {}
        self.sources: dict[tuple[int, int], Any] = {}
        self.supplied: dict[tuple[int, int], Any] = {}
        self.blocked: dict[tuple[int, int], dict[int, str]] = {}
        self.missing: list[str] = []
        self.owners: list[str] = []
        self.deviation = self.bound = self.gap = self.tolerance = 0.0
        weights: dict[int, Any] = {}    # float64 o_proj by layer, cast once for all rows
        for j in rows:
            for layer in range(layers):
                try:
                    mix = head_mix(run, j, layer, weights, lazy=True)
                except Refusal as error:
                    self.missing.append(f"t{j}.L{layer}: {error}")
                    continue
                key = (j, layer)
                count = mix.p.shape[1]
                scalar = reader is not None and reader.scalar
                # per-read shares: the scalar readout's share (H, T) without building the
                # (H, T, d) tensor, else the o parts themselves
                parts = mix.scalar_parts(reader.vector) if scalar else mix.o_parts()
                owner = [self._owners(bindings, count, layer, kv)
                         for kv in range(run.adapter.num_kv_heads)]
                edited_p = run.read_info(f"t{j}.L{layer}.p")["kind"] != "native"
                m = torch.zeros(n, heads, dtype=torch.float64)
                cover = torch.zeros(n, heads, dtype=torch.bool)
                sources = torch.zeros(n, heads, dtype=torch.int64)
                supplied = torch.zeros(n, heads, dtype=torch.bool)
                width = () if reader is None or scalar else (parts.shape[-1],)
                e = torch.zeros(n, heads, *width, dtype=torch.float64)
                why = None if reader is None else (reader.reaches(layer, "attn") or (
                    None if reader.row == j else f"{reader.text} is in another row"))
                blocked = {}
                # each binding's (d,) write into the row (content readouts): the o parts,
                # built only when asked for (the scalar path never builds them otherwise)
                full = (parts if not scalar else mix.o_parts()) if keep_writes else None
                total_write = (torch.zeros(n, full.shape[-1], dtype=torch.float64)
                               if keep_writes else None)
                con = torch.zeros(n, heads, dtype=torch.float64) if scalar else None
                mdel = torch.zeros(n, heads, dtype=torch.float64) if scalar else None
                sinkm = torch.zeros(n, heads, dtype=torch.float64)
                den = torch.zeros(heads, dtype=torch.float64)
                sink_rows = [s for s in self.sink if s < count]
                is_sink = torch.zeros(count, dtype=torch.bool)
                if sink_rows:
                    is_sink[sink_rows] = True
                for h in range(heads):
                    reason = edited_block(mix, run, h)
                    if reason:
                        blocked[h] = reason
                    index = owner[mix.kappa[h]]
                    held = index >= 0
                    idx = index[held]
                    m[:, h] = torch.zeros(n, dtype=torch.float64).index_add_(
                        0, idx, mix.p[h][held])
                    tally = torch.bincount(idx, minlength=n)
                    sources[:, h] = tally
                    cover[:, h] = tally > 0
                    share = torch.zeros(n, *parts.shape[2:], dtype=torch.float64).index_add_(
                        0, idx, parts[h][held])
                    if total_write is not None:
                        total_write.index_add_(0, idx, full[h][held])
                    if edited_p:
                        for s in held.nonzero().reshape(-1).tolist():
                            if source_label(run, mix, h, s) == SUPPLIED:
                                supplied[int(index[s]), h] = True
                    ph = mix.p[h]
                    in_sink = held & is_sink
                    sinkm[:, h] = torch.zeros(n, dtype=torch.float64).index_add_(
                        0, index[in_sink], ph[in_sink])
                    den[h] = ph.sum() - ph[is_sink].sum()
                    if reader is None:
                        e[:, h] = torch.where(cover[:, h], share.norm(dim=-1), 0.0)
                    elif why is None and h not in blocked:
                        if scalar:
                            e[:, h] = torch.where(cover[:, h], share, 0.0)
                            # contrast p_s (x_s - xbar_h) = parts_s - p_s sum_s' parts_s'
                            c_all = parts[h] - ph * parts[h].sum()
                            con[:, h] = torch.zeros(n, dtype=torch.float64).index_add_(
                                0, idx, c_all[held])
                            mdel[:, h] = torch.zeros(n, dtype=torch.float64).index_add_(
                                0, idx, (-c_all / (1 - ph))[held])
                        else:
                            e[:, h] = torch.where(cover[:, h, None], share, 0.0)
                self.mass[key], self.cover[key], self.effect[key] = m, cover, e
                self.sink_mass[key], self.nonsink_total[key] = sinkm, den
                if scalar:
                    self.contrast[key], self.mask_delta[key] = con, mdel
                if total_write is not None:
                    self.write[key] = total_write
                self.sources[key], self.supplied[key] = sources, supplied
                self.reach[key], self.blocked[key] = why, blocked
                self.total[key] = mix.p.sum(1)
                self.deviation = max(self.deviation, float((mix.p.sum(1) - 1).abs().max()))
                self.bound = max(self.bound, count * unit_roundoff(mix.dtype) + 2 * count * _U32)
                self.gap = max(self.gap, float((m.sum(0) - mix.p.sum(1)).abs().max()))
                self.tolerance = max(self.tolerance, 4 * count * 2.0 ** -53
                                     * float(mix.p.abs().sum(1).max()))

    def _owners(self, bindings: list[Binding], count: int, layer: int, kv: int) -> Any:
        """The binding holding each value cell ``cache.L<layer>.v[kv<kv>,row=s]``, ``s <
        count`` (-1: none); a cell held by zero or several bindings is listed in ``owners``."""
        torch = _torch()
        held: list[list[int]] = [[] for _ in range(count)]
        for b, binding in enumerate(bindings):
            for row in {c.row for c in binding.cells}:
                if row < count and binding.covers(row, layer, kv, "v"):
                    held[row].append(b)
        out = torch.full((count,), -1, dtype=torch.int64)
        for s, found in enumerate(held):
            if len(found) != 1:
                self.owners.append(f"cache.L{layer}.v[kv{kv},row={s}] is held by "
                                   f"{len(found)} bindings")
            if found:
                out[s] = found[0]
        return out

    def partition_error(self) -> str | None:
        """Why the bindings do not partition the heads' mass, or None."""
        if self.owners:
            return "the bindings do not partition the value cells: " + self.owners[0]
        if self.gap > self.tolerance:
            return (f"per head the bindings' mass differs from sum_s p by {self.gap:.3g} "
                    f"(> {self.tolerance:.3g}, float64 summation)")
        return None


def _torch():
    import torch

    return torch


def _entry(run, binding, b, grid, reader, found_rows, layers, min_mass, readers, here_key,
           tokenizer, foreign=None, item_labels=None) -> dict[str, Any]:
    """One binding's row of the table (``view``) and its float64 contribution (None when an
    edit blocks one of its reads)."""
    strips, first, last = [], None, None
    total_mass, count, supplied = 0.0, 0, False
    contrast_sum, mask_sum, nonsink, nonsink_ok = 0.0, 0.0, 0.0, True
    contribution: float | None = 0.0
    blocked_by = None
    scalar = reader is not None and reader.scalar
    reads = []
    for j in found_rows:
        marks = []
        for layer in range(layers):
            key = (j, layer)
            if key not in grid.mass:
                marks.append("?")
                continue
            covered = grid.cover[key][b]
            if not bool(covered.any()):
                marks.append("-")
                continue
            m = grid.mass[key][b]
            total_mass += float(m.sum())
            if grid.sink:
                den = grid.nonsink_total[key]
                own = m - grid.sink_mass[key][b]
                live = covered & (own != 0)
                if bool((den[live] <= 0).any()):
                    nonsink_ok = False
                elif bool(live.any()):
                    nonsink += float((own[live] / den[live]).sum())
            count += int(covered.sum())
            supplied = supplied or bool(grid.supplied[key][b].any())
            above = bool((m[covered] >= min_mass).any())
            marks.append("#" if above else ".")
            if above:
                first = first or f"t{j}.L{layer}"
                last = f"t{j}.L{layer}"
            e = grid.effect[key][b]
            for h in range(len(covered)):
                if not covered[h]:
                    continue
                reason = grid.blocked[key].get(h)
                value = e[h]
                if reason and reader is not None:
                    shown, rank = refused(reason), -1.0
                    if scalar and grid.reach[key] is None:
                        blocked_by = blocked_by or reason
                elif scalar and grid.reach[key] is not None:
                    shown, rank = refused(grid.reach[key]), -1.0
                elif reader is None:
                    shown, rank = {"norm": num(float(value), EXACT, "P07")}, float(value)
                else:
                    shown = claim(value, EXACT, "P19" if reader.scalar else "P07")
                    rank = size(value)
                    if scalar:
                        contribution += float(value)
                        contrast_sum += float(grid.contrast[key][b, h])
                        mask_sum += float(grid.mask_delta[key][b, h])
                label = (SUPPLIED if grid.supplied[key][b, h] else
                         MEASURED if int(grid.sources[key][b, h]) == 1 else EXACT)
                read = {"name": f"t{j}.L{layer}.h{h}", "mass": num(float(m[h]), label),
                        "effect": shown}
                if scalar and key in grid.contrast and not (reason and reader is not None) \
                        and grid.reach[key] is None:
                    read["contrast"] = num(float(grid.contrast[key][b, h]), EXACT, "P19")
                if foreign is not None:
                    from mdb.core.foreign_names import read_names

                    read["foreign"] = read_names(j, layer, h)
                reads.append((rank, read))
        strips.append(f"t{j}:{''.join(marks)}")
    reads.sort(key=lambda r: -r[0])
    edits = []
    for row in sorted({c.row for c in binding.cells}):
        edits += _status_edits(run, row, layers, here_key)
    pending = any(c == "?" for strip in strips for c in strip.split(":", 1)[1])
    if edits:
        status = "edited"
    elif first is not None:
        status = "read"
    elif pending:
        status = "live"
    else:
        status = "unread"
    label = SUPPLIED if supplied else EXACT
    view: dict[str, Any] = {
        "symbol": binding.name, "kind": binding.kind,
        "rows": _compact_rows(sorted({c.row for c in binding.cells})),
        "definitions": _compact_list(binding.addresses),
        "lifetime": {"strip": " ".join(strips), "first": first, "last": last},
        "mass": num(total_mass, label),
        "count": count,
        "mean_mass": (num(total_mass / count, label) if count else
                      {"state": "unresolved", "reason": "no read of this binding"}),
        "readers": [r[1] for r in reads[:readers]],
        "status": status,
    }
    if binding.kind == "token":
        row = binding.cells[0].row
        token = _token_at(run, row)
        view["position"] = row
        view["is_sink"] = row in grid.sink
        if token is not None:
            view["token"] = token
            if tokenizer is not None:
                view["text"] = tokenizer.decode([token])
            view.update(token_meta(run, row, tokenizer))
    if blocked_by is not None:
        contribution = None
        view["to_consumer"] = refused(blocked_by)
    elif scalar:
        view["to_consumer"] = num(contribution, EXACT, "P19")
        view["contrast"] = num(contrast_sum, EXACT, "P19")
        if math.isfinite(mask_sum):
            view["mask_delta"] = num(mask_sum, EXACT, "P19")
        else:
            view["mask_delta"] = refused("a read has p = 1: masking its only source does not "
                                         "renormalise")
        native = reader.native
        if native is not None and float(f64(native)) != 0.0:
            total = float(f64(native))
            view["share"] = num(contribution / total, EXACT)
            view["share_of_abs"] = num(contribution / abs(total), EXACT)
    else:
        contribution = None
        view["to_consumer"] = {"state": "unresolved", "reason": "no scalar consumer"}
    if grid.sink:
        if not nonsink_ok:
            view["mass_nonsink"] = refused("a head reads only sink rows (no non-sink mass)")
        else:
            view["mass_nonsink"] = num(nonsink, label)
    if binding.slot is not None:
        view["slot"] = binding.slot
        view["role"] = binding.role
        if binding.map_digest is not None:
            view["map_digest"] = binding.map_digest
    if binding.labels or item_labels:
        view["labels"] = {**(item_labels or {}),
                          **{k: label_claim(v) for k, v in binding.labels.items()}}
    if foreign is not None:
        from mdb.core.foreign_names import binding_names

        view["foreign"] = binding_names(binding.cells, **foreign)
    if edits:
        view["edited_by"] = edits[:8]
    if len(reads) > readers:
        view["omitted"] = len(reads) - readers
    return {"view": view, "contribution": contribution}


_SPECIAL = ("<|", "<s>", "</s>", "<pad>", "<unk>")


def token_meta(run: Any, row: int, tokenizer: Any) -> dict[str, Any]:
    """Per-row token metadata (review 2.6, structural from the tokenizer): ``token_class``
    (``special``, ``bos_or_t0``, ``whitespace``, ``punct``, ``digit``, ``word_initial`` or
    ``continuation``) and ``pieces_of_word`` (index of this piece within its whitespace-delimited
    word, and the word's piece count). Without a tokenizer only ``bos_or_t0`` is known."""
    if row == 0:
        cls = "bos_or_t0"
    elif tokenizer is None:
        return {}
    else:
        cls = _token_class(tokenizer.decode([_token_at(run, row)]))
    out: dict[str, Any] = {"token_class": cls}
    if tokenizer is None:
        return out
    texts = [tokenizer.decode([_token_at(run, r)]) for r in range(max(0, row - 32),
                                                                     row + 1)]
    start = len(texts) - 1
    while start > 0 and not texts[start][:1].isspace() and _word_piece(texts[start]):
        start -= 1
    index = len(texts) - 1 - start
    end, pieces = row, index + 1
    while True:
        nxt = _token_at(run, end + 1) if end + 1 < len(run.tokens) else None
        if nxt is None:
            break
        text = tokenizer.decode([nxt])
        if text[:1].isspace() or not _word_piece(text):
            break
        end, pieces = end + 1, pieces + 1
    out["pieces_of_word"] = {"index": index, "count": pieces}
    return out


def _word_piece(text: str) -> bool:
    return bool(text) and any(ch.isalnum() for ch in text)


def _token_class(text: str) -> str:
    if any(text.startswith(p) for p in _SPECIAL):
        return "special"
    if not text.strip():
        return "whitespace"
    body = text.strip()
    if body.isdigit():
        return "digit"
    if not any(ch.isalnum() for ch in body):
        return "punct"
    return "word_initial" if text[:1].isspace() else "continuation"


def label_claim(value: Any, authority: str = SUPPLIED) -> dict[str, Any]:
    """A label as a record leaf: a number is a claim ``{value, authority}``; any other value
    is ``{label, source}`` (``source``: ``supplied`` or ``derived``). ``value`` may already be
    such a leaf, or a ``(value, authority)`` pair."""
    if isinstance(value, dict) and ("label" in value or "value" in value):
        return dict(value)
    if isinstance(value, tuple) and len(value) == 2:
        value, authority = value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return num(value, authority if authority in (SUPPLIED, EXACT, MEASURED, BOUND,
                                                     "derived-local") else EXACT)
    return {"label": value, "source": "derived" if authority == "derived" else str(authority)}


_LIST_CAP = 64     # records hold lists of at most 64 items (core.pointers)


def _compact_rows(rows: list[int]) -> list[Any]:
    """The rows as given when there are at most ``_LIST_CAP``; else half-open ranges ``"a:b"``
    (a long span such as a whole chain of thought is a few ranges, not thousands of ids)."""
    if len(rows) <= _LIST_CAP:
        return rows
    ranges, start = [], rows[0]
    for prev, cur in zip(rows, rows[1:] + [None]):
        if cur != prev + 1 if cur is not None else True:
            ranges.append(f"{start}:{prev + 1}")
            start = cur
    if len(ranges) > _LIST_CAP:
        ranges = ranges[:_LIST_CAP - 1] + [f"... +{len(ranges) - _LIST_CAP + 1} more ranges"]
    return ranges


def _compact_list(items: list[str]) -> list[str]:
    """At most ``_LIST_CAP`` items: the first ones and a count of the rest."""
    if len(items) <= _LIST_CAP:
        return items
    return items[:_LIST_CAP - 1] + [f"... +{len(items) - _LIST_CAP + 1} more"]


def _token_at(run: Any, row: int) -> int | None:
    if row < len(run.tokens):
        return int(run.tokens[row])
    held = run.session.read("tokens")[0]
    return int(held[row]) if row < held.numel() else None


def _records(run: Any, report: dict[str, Any], session_id: str | None) -> list[dict[str, Any]]:
    """The table sealed as one ``symbol-table`` record (this page)."""
    body = {"schema": SCHEMA, **{k: report[k] for k in ("where", "consumer", "rows", "min_mass",
                                                         "bindings", "unbound", "page",
                                                         "accounting")}}
    for key in ("labels", "derived_labels", "candidate_sets", "components", "subspaces"):
        if key in report:
            body[key] = report[key]
    if "consumer_total" in report:
        body["consumer_total"] = report["consumer_total"]
    body["procedure"] = procedure(run)

    def seal() -> dict[str, Any]:
        return seal_record(session_id=session_id or run_session_id(run), kind="symbol-table",
                           model_identity=run.adapter.model_identity,
                           execution=dict(run.adapter.execution), authority=EXACT, body=body)

    try:
        return [seal()]
    except RecordError:
        if not body.get("components"):
            raise
    # a feature binding of a k=50 SAE (about 50 rows with content) exceeds the record bound:
    # each section keeps its totals and partition, its rows go behind a sha256 pointer
    # (``core.bounded.load_json``); ``report["components"]`` keeps every row
    from mdb.core.bounded import put_json

    body["components"] = [
        {**{k: v for k, v in section.items() if k != "components"},
         "components_full": put_json(section.get("components", []), run=run)}
        for section in body["components"]]
    return [seal()]


__all__ = ["Binding", "Candidates", "LEGEND", "content", "gate_switch_forecast", "hop_edge", "readout_basis", "PAGE", "STATUSES", "SYMBOL_TABLE", "Table", "declare",
           "declared", "table"]
