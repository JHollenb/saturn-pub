"""Sparse-autoencoder feature bases on residual ports, with the error term always explicit.

A ``FeatureBasis`` is one SAE bound to one residual site of the port program: the output of
decoder layer ``L`` (``L<L>.h_out``; the same value is ``L<L+1>.h`` when no edit sits between
them). Qwen-Scope's ``hook_point: resid_post`` is that site. For a native value ``x`` at the
site the basis gives the exact identity

    x = b_dec + sum_{f in S} a_f d_f + e

where ``S`` is the set of active features (top-k of ``relu(W_enc x + b_enc)``, the published
selection rule, computed in float32 exactly as the published encoder does), ``a_f`` the
activations recomputed in float64 from the native bits (rows of ``W_enc`` sliced before the
cast), ``d_f = W_dec[:, f]`` the decoder directions (columns sliced before the cast) and
``e = x - b_dec - sum a_f d_f`` the error, defined as the remainder so the identity closes to
float64 roundoff by construction. The error is a listed term everywhere: never dropped.

Direct effects on a margin ``M[a,b]``: the residual at the site reaches the final state through
the skip path unchanged, so ``M = rho c.x + sum_{l>L} rho c.(o_l + m_l) + gaps`` with ``rho c``
the readout of ``reads.readout`` (P19). Splitting ``rho c.x`` over the identity above gives each
feature's, the bias's and the error's direct share; the rest of the margin is the later layers'
terms and the measured remainder. ``margin_split`` returns all of it with the closure.

Labels: the native value is ``measured``; activations, directions, reconstruction, error and
direct effects are ``derived-exact`` (float64 identities over the native value and the SAE's
stored float32 weights, which are ``supplied``); logit-lens token lists are ``derived-local``
(float32 ranking only, never a decomposition).

Memory: weights stay in their stored float32 on the host; nothing is cast whole. A set of
layers (``SaeSet``) loads one layer file when it is first used and keeps at most ``cache``
layers loaded (least recently used dropped).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .finite import EXACT, LOCAL, MEASURED, SUPPLIED, f64

SCHEMA = "mdb-sae-v1"
SITE_PORTS = ("h_out", "h")
_HOOKS = {"resid_post": "h_out", "hook_resid_post": "h_out", "h_out": "h_out",
          "resid_pre": "h", "hook_resid_pre": "h", "h": "h"}


def _torch():
    import torch

    return torch


class SaeError(ValueError):
    """A basis that cannot be loaded or applied (shape, site, file): a refusal, not a crash."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(directory: str) -> dict[str, str]:
    """``name -> sha256`` from a ``manifest.txt`` (``NAME SHA256 SIZE`` lines) beside the files,
    written by the transfer that verified them."""
    path = os.path.join(directory, "manifest.txt")
    out: dict[str, str] = {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 2 and re.fullmatch(r"[0-9a-f]{64}", parts[1]):
                out[parts[0]] = parts[1]
    return out


@dataclass
class FeatureBasis:
    """One SAE at one site. Tensors are the stored ones (float32 on the host):
    ``W_enc (d_sae, d)``, ``b_enc (d_sae,)``, ``W_dec (d, d_sae)``, ``b_dec (d,)``."""

    name: str
    layer: int
    port: str
    k: int
    W_enc: Any
    b_enc: Any
    W_dec: Any
    b_dec: Any
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.port not in SITE_PORTS:
            raise SaeError(f"an SAE site is one of {', '.join(SITE_PORTS)}, not {self.port}")
        d_sae, d = (int(x) for x in self.W_enc.shape)
        if tuple(self.W_dec.shape) == (d_sae, d):           # stored (d_sae, d): transpose view
            self.W_dec = self.W_dec.T
        if (tuple(self.W_dec.shape) != (d, d_sae) or tuple(self.b_enc.shape) != (d_sae,)
                or tuple(self.b_dec.shape) != (d,)):
            raise SaeError(f"SAE {self.name} L{self.layer}: shapes W_enc {tuple(self.W_enc.shape)}"
                           f" W_dec {tuple(self.W_dec.shape)} b_enc {tuple(self.b_enc.shape)} "
                           f"b_dec {tuple(self.b_dec.shape)} do not form one basis")
        if not 1 <= self.k <= d_sae:
            raise SaeError(f"k = {self.k} is outside 1..{d_sae}")

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def site(self) -> str:
        return f"L{self.layer}.{self.port}"

    def covers(self, layer: int, port: str) -> bool:
        """True when ``L<layer>.<port>`` holds the value this basis was trained on."""
        if self.port == "h_out":
            return (layer, port) in ((self.layer, "h_out"), (self.layer + 1, "h"))
        return (layer, port) in ((self.layer, "h"), (self.layer - 1, "h_out"))

    # --- the identity ---------------------------------------------------------------------

    def _vector(self, x: Any) -> Any:
        torch = _torch()
        flat = x.detach().reshape(-1)
        if flat.numel() != self.d_model:
            raise SaeError(f"SAE {self.name} expects {self.d_model} values, got {flat.numel()}")
        if not flat.is_floating_point():
            raise SaeError("SAE input must be floating point")
        return flat.to("cpu"), torch

    def select(self, x: Any) -> tuple[list[int], Any]:
        """The active set by the published rule: ``topk(relu(W_enc x + b_enc), k)`` in float32
        (``x`` widened to float32 exactly from bf16/fp16); entries whose float32 value is 0 are
        not active. Returns ``(ids, pre32 of those ids)`` in descending order."""
        flat, torch = self._vector(x)
        pre = torch.addmv(self.b_enc, self.W_enc, flat.to(self.W_enc.dtype))
        values, ids = torch.topk(torch.relu(pre), self.k)
        keep = values > 0
        return [int(i) for i in ids[keep].tolist()], values[keep]

    def encode(self, x: Any) -> dict[str, Any]:
        """``{ids, acts}``: the active set and its float64 activations
        ``relu(W_enc[ids] x + b_enc[ids])`` (rows sliced, then cast; ``x`` cast once)."""
        torch = _torch()
        flat, _ = self._vector(x)
        ids, _ = self.select(flat)
        x64 = flat.to(torch.float64)
        index = torch.tensor(ids, dtype=torch.long)
        rows = self.W_enc.index_select(0, index).to(torch.float64)
        acts = torch.relu(rows @ x64 + self.b_enc.index_select(0, index).to(torch.float64))
        return {"ids": ids, "acts": acts}

    def direction(self, feature: int) -> Any:
        """``d_f`` in float64 (one column sliced, then cast)."""
        torch = _torch()
        if not 0 <= int(feature) < self.d_sae:
            raise SaeError(f"feature {feature} is outside 0..{self.d_sae - 1}")
        return self.W_dec[:, int(feature)].to(torch.float64)

    def directions(self, ids: Sequence[int]) -> Any:
        torch = _torch()
        index = torch.tensor(list(ids), dtype=torch.long)
        return self.W_dec.index_select(1, index).to(torch.float64)       # (d, n)

    def bias(self) -> Any:
        return self.b_dec.to(_torch().float64)

    def decompose(self, x: Any) -> Decomposition:
        """The exact identity ``x = b_dec + D a + e`` at a native value ``x``."""
        torch = _torch()
        flat, _ = self._vector(x)
        x64 = flat.to(torch.float64)
        enc = self.encode(flat)
        D = self.directions(enc["ids"])
        parts = D * enc["acts"]                     # (d, n): a_f d_f per column
        bias = self.bias()
        recon = bias + parts.sum(dim=1)
        error = x64 - recon
        return Decomposition(self, x64, enc["ids"], enc["acts"], D, parts, bias, recon, error)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "layer": self.layer, "site": self.site, "k": self.k,
                "d_model": self.d_model, "d_sae": self.d_sae, "source": dict(self.source)}


@dataclass
class Decomposition:
    basis: FeatureBasis
    x: Any
    ids: list[int]
    acts: Any
    D: Any
    parts: Any
    bias: Any
    recon: Any
    error: Any

    def stats(self) -> dict[str, Any]:
        """Norm accounting of the identity (``derived-exact`` except the native norm)."""
        x2 = float(self.x.pow(2).sum())
        e2 = float(self.error.pow(2).sum())
        closure = float((self.bias + self.parts.sum(dim=1) + self.error - self.x).abs().max())
        return {"x_norm": _n(math.sqrt(x2), MEASURED), "error_norm": _n(math.sqrt(e2), EXACT),
                "recon_norm": _n(float(self.recon.norm()), EXACT),
                "bias_norm": _n(float(self.bias.norm()), EXACT),
                "error_fraction_of_norm": _n(math.sqrt(e2 / x2) if x2 else 0.0, EXACT),
                "error_fraction_of_energy": _n(e2 / x2 if x2 else 0.0, EXACT),
                "active": len(self.ids),
                "identity_closure": _n(closure, EXACT)}


def _n(value: float, authority: str, proof: str | None = None) -> dict[str, Any]:
    from mdb.core.records import num

    return num(float(value), authority, proof=proof)


# --------------------------------------------------------------------------- sets of layers


@dataclass
class SaeSet:
    """A named family of per-layer SAEs read from a directory (Qwen-Scope layout:
    ``config.json`` + ``layer{L}.sae.pt``) or from one file bound to one layer."""

    name: str
    path: str
    port: str
    k: int
    layers: tuple[int, ...]
    files: dict[int, str]
    cache: int = 4
    config: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None
    verify: bool = False
    _loaded: OrderedDict = field(default_factory=OrderedDict, repr=False)
    hashes: dict[int, dict[str, Any]] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "site_port": self.port, "k": self.k,
                "layers": list(self.layers), "cache": self.cache,
                "loaded": list(self._loaded), "revision": self.revision,
                "hook_point": self.config.get("hook_point"),
                "repo": self.config.get("hf_repo_id"),
                "files": {str(layer): self.hashes.get(layer, {"file": os.path.basename(p)})
                          for layer, p in sorted(self.files.items())
                          if layer in self._loaded or layer in self.hashes}}

    def site_of(self, layer: int) -> str:
        return f"L{layer}.{self.port}"

    def for_site(self, layer: int, port: str) -> FeatureBasis | None:
        """The basis whose site holds ``L<layer>.<port>``, or None."""
        candidates = [layer] if port == self.port else (
            [layer - 1] if (self.port, port) == ("h_out", "h") else
            [layer + 1] if (self.port, port) == ("h", "h_out") else [])
        for own in candidates:
            if own in self.files:
                return self.get(own)
        return None

    def get(self, layer: int) -> FeatureBasis:
        torch = _torch()
        if layer not in self.files:
            raise SaeError(f"SAE {self.name} has no layer {layer} (has "
                           f"{', '.join(map(str, sorted(self.files)))})")
        if layer in self._loaded:
            self._loaded.move_to_end(layer)
            return self._loaded[layer]
        path = self.files[layer]
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:  # noqa: BLE001 - reported as a refusal
            raise SaeError(f"cannot load {path}: {type(error).__name__}: {error}") from None
        missing = [key for key in ("W_enc", "W_dec", "b_enc", "b_dec") if key not in state]
        if missing:
            raise SaeError(f"{path} lacks {', '.join(missing)}")
        info: dict[str, Any] = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
        listed = _manifest(os.path.dirname(path)).get(os.path.basename(path))
        if listed:
            info["sha256_manifest"] = listed
        if self.verify:
            info["sha256"] = _sha256(path)
            if listed:
                info["manifest_match"] = info["sha256"] == listed
        self.hashes[layer] = info
        source = {"set": self.name, "path": path, "revision": self.revision,
                  "repo": self.config.get("hf_repo_id"), **info}
        basis = FeatureBasis(self.name, layer, self.port, self.k, state["W_enc"],
                             state["b_enc"], state["W_dec"], state["b_dec"], source)
        self._loaded[layer] = basis
        while len(self._loaded) > max(1, self.cache):
            self._loaded.popitem(last=False)
        return basis


def open_set(name: str, path: str, *, layers: Sequence[int] | None = None, k: int | None = None,
             port: str | None = None, cache: int = 4, verify: bool = False,
             layer: int | None = None) -> SaeSet:
    """An ``SaeSet`` over a directory (``config.json`` gives ``k``, ``hook_point`` and the
    layers; files ``layer{L}.sae.pt``) or one ``.pt`` file (``layer`` and ``k`` required).
    Nothing is loaded until a layer is used."""
    path = os.path.abspath(os.path.expanduser(path))
    config: dict[str, Any] = {}
    files: dict[int, str] = {}
    revision = None
    if os.path.isdir(path):
        cfg = os.path.join(path, "config.json")
        if os.path.isfile(cfg):
            config = json.load(open(cfg, encoding="utf-8"))
        rev = os.path.join(path, "REVISION")
        if os.path.isfile(rev):
            revision = open(rev, encoding="utf-8").read().strip() or None
        for entry in os.listdir(path):
            match = re.fullmatch(r"layer(\d+)\.sae\.pt", entry)
            if match:
                files[int(match[1])] = os.path.join(path, entry)
        if not files:
            raise SaeError(f"{path} holds no layer<L>.sae.pt files")
    elif os.path.isfile(path):
        if layer is None:
            raise SaeError("a single SAE file needs layer=L")
        files[int(layer)] = path
    else:
        raise SaeError(f"no SAE at {path}")
    if layers is not None:
        wanted = set(int(x) for x in layers)
        absent = sorted(wanted - set(files))
        if absent:
            raise SaeError(f"{path} has no layer(s) {', '.join(map(str, absent))}")
        files = {key: value for key, value in files.items() if key in wanted}
    hook = port or _HOOKS.get(str(config.get("hook_point", "")), None)
    if hook is None:
        raise SaeError(f"unknown hook point {config.get('hook_point')!r}: give port=h_out|h")
    k = int(k if k is not None else config.get("k", 0) or 0)
    if k < 1:
        raise SaeError("the SAE's k (top-k) is not in config.json: give k=K")
    return SaeSet(name, path, hook, k, tuple(sorted(files)), files, cache=int(cache),
                  config=config, revision=revision, verify=verify)


# --------------------------------------------------------------------------- analysis


def logit_lens(run: Any, vector: Any, top: int = 5) -> list[int]:
    """Token ids whose readout ``W_U (gamma . v)`` is largest for direction ``v``: float32
    ranking on the head's device, head read in row chunks (sliced, then cast), never whole."""
    torch = _torch()
    adapter = run.adapter
    weight = adapter.model.lm_head.weight
    norm = getattr(adapter.backbone, adapter.spec.final_norm)
    v = (f64(norm.weight).reshape(-1) * f64(vector).reshape(-1)).to(torch.float32)
    v = v.to(weight.device)
    best_vals, best_ids = None, None
    chunk = 16384
    for start in range(0, int(weight.shape[0]), chunk):
        part = weight[start:start + chunk].to(torch.float32) @ v
        vals, ids = torch.topk(part, min(top, int(part.numel())))
        ids = ids + start
        if best_vals is None:
            best_vals, best_ids = vals, ids
        else:
            joined_v, joined_i = torch.cat((best_vals, vals)), torch.cat((best_ids, ids))
            order = torch.topk(joined_v, min(top, int(joined_v.numel()))).indices
            best_vals, best_ids = joined_v[order], joined_i[order]
    return [int(i) for i in best_ids.tolist()]


def margin_split(run: Any, reader: Any, basis: FeatureBasis, layer: int, port: str,
                 dec: Decomposition | None = None) -> dict[str, Any]:
    """Exact split of a scalar consumer (margin or logit of ``reader.row``) at the site value
    ``L<layer>.<port>`` of that row.

    ``consumer = rho c.x  +  sum over later residual terms  +  remainder``; ``rho c.x`` is split
    over ``b_dec``, every active feature and the error. Returns float64 numbers (``direct``,
    ``features`` (id, act, effect), ``bias``, ``error``, ``downstream`` list of
    ``(name, address, value)``, ``remainder``, ``native``) — the caller labels them."""
    torch = _torch()
    row = reader.row
    address = f"t{row}.L{layer}.{port}"
    x = run.read(address)
    dec = dec or basis.decompose(x)
    w = reader.vector
    direct = float(w @ dec.x)
    effects = (w @ dec.D) * dec.acts                         # per feature, float64
    bias = float(w @ dec.bias)
    error = float(w @ dec.error)
    last = run.adapter.layers - 1
    downstream = []
    first = layer + 1 if port == "h_out" else layer
    for l in range(first, last + 1):
        for part, name in (("o", "attn"), ("m", "mlp")):
            value = float(reader.apply(run.read(f"t{row}.L{l}.{part}")))
            downstream.append((f"L{l}.{name}", f"t{row}.L{l}.{part}", value))
    native = float(reader.native)
    listed = direct + sum(v for _, _, v in downstream) + float(reader.bias)
    remainder = native - listed
    return {"address": address, "decomposition": dec, "direct": direct,
            "features": [(int(i), float(a), float(e))
                         for i, a, e in zip(dec.ids, dec.acts.tolist(), effects.tolist())],
            "bias": bias, "error": error, "downstream": downstream,
            "head_bias": float(reader.bias), "remainder": remainder, "native": native,
            "identity_gap": direct - (bias + error + float(effects.sum())),
            "torch": torch.__version__}


def edit_vector(basis: FeatureBasis, terms: Sequence[tuple], values: Any) -> Any:
    """``sum_t scale_t * vector_t`` in float64. A term is ``(kind, feature, coef, scale,
    origin)``: ``feature`` (``d_f``; with ``coef`` times the feature's float64 activation at the
    site's value), ``error`` (``e``), ``recon`` (``b_dec + D a``) or ``bias`` (``b_dec``).
    ``values`` maps an origin (None: this run) to the site's value there (a bare tensor stands
    for ``{None: tensor}``); terms that need no value ignore it."""
    torch = _torch()
    if not isinstance(values, dict):
        values = {None: values}
    total = torch.zeros(basis.d_model, dtype=torch.float64)
    decs: dict[Any, Decomposition] = {}
    for term in terms:
        kind, feature, coef, scale = term[:4]
        origin = term[4] if len(term) > 4 else None
        needs = kind in ("error", "recon") or coef
        if needs and origin not in decs:
            if values.get(origin) is None:
                where = f"@{origin}" if origin else ""
                raise SaeError(f"{kind}{where}{'(coef)' if coef else ''} needs the value at the "
                               "site")
            decs[origin] = basis.decompose(values[origin])
        dec = decs.get(origin)
        if kind == "feature":
            vector = basis.direction(feature)
            if coef:
                acts = dict(zip(dec.ids, dec.acts.tolist()))
                vector = vector * float(acts.get(int(feature), 0.0))
        elif kind == "error":
            vector = dec.error
        elif kind == "recon":
            vector = dec.recon
        elif kind == "bias":
            vector = basis.bias()
        else:
            raise SaeError(f"unknown feature term {kind!r}")
        total = total + float(scale) * vector
    return total


_TERM = re.compile(
    r"(?P<sign>[+-]?)(?:(?P<pre>[0-9.eE+-]+)\*)?"
    r"(?:(?P<feat>feature)\[(?P<id>\d+)(?P<coef>,coef)?\]|(?P<word>error|recon|bias))"
    r"(?:@(?P<origin>[A-Za-z_][\w-]*))?"
    r"(?:\*(?P<post>[-+]?[0-9.]+(?:[eE][-+]?\d+)?))?")


def parse_terms(text: str) -> list[tuple]:
    """``feature[ID]``, ``feature[ID,coef]``, ``error``, ``recon``, ``bias``, each optionally
    ``@RUN`` (the value at the same address of the named run) and scaled ``X*`` / ``*X``,
    joined by ``+`` or ``-``; no spaces, never evaluated. ``feature[12,coef]*-1`` is minus the
    feature's own contribution ``a_12 d_12``; ``recon@rome-recon`` swaps in the other run's
    reconstruction and keeps this run's error."""
    text = text.strip()
    terms, at = [], 0
    while at < len(text):
        match = _TERM.match(text, at)
        if match is None or match.end() == at:
            raise SaeError(f"cannot read feature term at {text[at:]!r}: write feature[ID], "
                           "feature[ID,coef], error, recon or bias, [@RUN], with *X, joined by "
                           "+ or -")
        scale = 1.0
        for key in ("pre", "post"):
            if match[key]:
                try:
                    scale *= float(match[key])
                except ValueError:
                    raise SaeError(f"bad scale {match[key]!r}") from None
        if match["sign"] == "-":
            scale = -scale
        if not math.isfinite(scale):
            raise SaeError("a feature scale must be finite")
        kind = "feature" if match["feat"] else match["word"]
        if kind == "bias" and match["origin"]:
            raise SaeError("bias@RUN is the same b_dec: drop @RUN")
        terms.append((kind, int(match["id"]) if match["id"] else None, bool(match["coef"]),
                      scale, match["origin"]))
        at = match.end()
    if not terms:
        raise SaeError("empty feature expression")
    return terms


def is_expression(value: Any) -> bool:
    return isinstance(value, str) and bool(re.match(r"[+-]?([0-9.eE+-]+\*)?"
                                                     r"(feature\[|error\b|recon\b|bias\b)",
                                                     value.strip()))


def circuit(run: Any, reader: Any, split: dict[str, Any], basis: FeatureBasis, *, top: int = 4,
            session_id: str | None = None) -> Any:
    """A circuit record (kind ``circuit``, ``circuits.ledger``) of a scalar consumer with the
    site's residual expanded over the SAE: consumer -> [site residual -> features (top, ``+N
    more``), SAE bias, SAE error], the later layers' attn/mlp terms (top, ``+N more``), the
    head bias and the measured remainder. The error node is fixed: never folded or dropped."""
    from mdb.core.pointers import RecordError, check_bounded
    from mdb.core.records import num, seal_record

    from . import circuits as C
    from .forecast import Refusal, run_session_id
    from .reads import claim, size

    if not 1 <= top <= 8:
        raise Refusal("top must be in 1..8")
    address = split["address"]
    feats = [C.Node(f"f{i}", "feature", f"{address}{{sae={basis.name},feature={i}}}",
                    _scalar(e), EXACT, "P19",
                    note=f"a_f {a:.6g} x rho c.d_f (SAE {basis.name} L{basis.layer})")
             for i, a, e in split["features"]]
    children = C._group(feats, top, f"{address} features")
    children.append(C.Node("sae bias", "bias", f"{address}{{sae={basis.name},b_dec}}",
                           _scalar(split["bias"]), EXACT, "P19", note="rho c.b_dec"))
    children.append(C.Node("sae error", "error", f"{address}{{sae={basis.name},error}}",
                           _scalar(split["error"]), EXACT, "P19",
                           note="rho c.(native - reconstruction): what the SAE drops"))
    site = C.Node(f"{address.split('.', 1)[1]} via SAE", "residual", address,
                  _scalar(split["direct"]), EXACT, "P19", children=children,
                  note="the site's residual on the skip path: rho c.x")
    later = [C.Node(name, "attn" if name.endswith("attn") else "mlp", addr, _scalar(v), EXACT,
                    "P19", port=addr, index=()) for name, addr, v in split["downstream"]]
    nodes = [site] + C._group(later, top, f"after {address}")
    if split["head_bias"]:
        nodes.append(C.Node("head bias", "bias", "lm_head.bias", _scalar(split["head_bias"]),
                            EXACT, "P19"))
    nodes.append(C._remainder("remainder", reader.text, _scalar(split["remainder"]),
                              "native consumer minus the listed terms: rounding of the residual "
                              "adds after the site, final norm and head (not bounded here; "
                              "explain bounds them)"))
    native = _scalar(split["native"])
    root = C.Node(reader.text, "consumer", reader.text, native,
                  reader.native_authority or MEASURED, children=nodes)
    total = sum((n.value for n in nodes[1:]), nodes[0].value)
    closure = native - total
    scale = sum(size(n.value) for n in nodes) + size(native)
    entries = C.ledger(run)
    ident = max(entries, default=0) + 1
    ids = [0]
    tree = root.view(ids)
    body = {"schema": C.SCHEMA, "consumer": reader.text, "kind": reader.kind,
            "where": run.where, "k": top, "depth": 1, "count": ids[0], "root": tree,
            "sae": {"name": basis.name, "site": address,
                    "revision": basis.source.get("revision"),
                    "file_sha256": basis.source.get("sha256")
                    or basis.source.get("sha256_manifest")},
            "closure": {"value": claim(closure, EXACT, "P19"),
                        "bound": num(float(64 * 2.0 ** -53 * scale), "bound"),
                        "within": bool(size(closure) <= 64 * 2.0 ** -53 * scale),
                        "by_construction": True,
                        "meaning": "native consumer minus the sum of the depth-1 nodes (float64 "
                                   "summation only); the SAE identity closes by defining the "
                                   "error as native minus reconstruction"},
            "remainders_within": None,
            "edits": {"count": 0, "meaning": "edit displacements stay inside the terms that "
                                             "carry them in an SAE circuit"},
            "premises": list(reader.premises)}
    try:
        check_bounded(body, "circuit record")
        record = seal_record(session_id=session_id or run_session_id(run), kind="circuit",
                             model_identity=run.adapter.model_identity,
                             execution=dict(run.adapter.execution), authority=EXACT, body=body)
    except RecordError as error:
        raise Refusal(f"the SAE circuit record does not fit one record ({error})") from None
    built = C.Circuit(ident, record, root, reader.target, run.where, complement=None)
    entries[ident] = built
    return built


def _scalar(value: float) -> Any:
    return _torch().tensor(float(value), dtype=_torch().float64)


def unit_vector(run: Any, row: int, text: str) -> tuple[Any, str]:
    """The residual-stream write of one mdb unit, float64: ``L<l>.f[i=N]`` the neuron's
    ``W_down[:, N]`` (a weight column: per unit of ``f``), ``L<l>.A[hH]`` head ``H``'s write
    ``W_o,h A_h`` at ``row``, ``L<l>.A[hH,src=S]`` the part of it read from source row ``S``
    (``p[h,S] W_o,h v[kappa(h),S]``, P06/P07). Returns ``(vector, meaning)``."""
    from .reads import head_mix, layer_module

    torch = _torch()
    match = re.fullmatch(r"L(?P<l>\d+)\.(?:f\[i=(?P<i>\d+)\]|A\[h(?P<h>\d+)(?:,src=(?P<s>\d+))?\])",
                         text.strip())
    if match is None:
        raise SaeError(f"match= takes L<l>.f[i=N], L<l>.A[hH] or L<l>.A[hH,src=S], not {text!r}")
    layer = int(match["l"])
    if not 0 <= layer < run.adapter.layers:
        raise SaeError(f"no layer {layer}")
    module = layer_module(run.adapter, layer)
    if match["i"] is not None:
        weight = module.mlp.down_proj.weight
        i = int(match["i"])
        if not 0 <= i < int(weight.shape[1]):
            raise SaeError(f"neuron {i} is outside 0..{int(weight.shape[1]) - 1}")
        return weight[:, i].detach().to("cpu").to(torch.float64), \
            f"W_down[:, {i}] of L{layer} (the write per unit of f[i={i}])"
    mix = head_mix(run, row, layer)
    h = int(match["h"])
    if not 0 <= h < int(mix.p.shape[0]):
        raise SaeError(f"head {h} is outside 0..{int(mix.p.shape[0]) - 1}")
    if match["s"] is None:
        return mix.o_parts()[h].sum(dim=0).to("cpu"), f"W_o,h{h} A_h{h} of L{layer} at t{row}"
    s = int(match["s"])
    if not 0 <= s < int(mix.p.shape[1]):
        raise SaeError(f"source {s} is outside 0..{int(mix.p.shape[1]) - 1}")
    return mix.o_parts()[h, s].to("cpu"), \
        f"p[h{h},{s}] W_o,h{h} v[kv{mix.kappa[h]},{s}] of L{layer} at t{row}"


def match_unit(basis: FeatureBasis, vector: Any, top: int = 8) -> dict[str, Any]:
    """Decoder directions closest to ``vector``: float32 cosine ranking over all features, then
    float64 cosines of the top ones (columns sliced, then cast) and the fraction of ``vector``
    left outside their span (least squares, float64)."""
    torch = _torch()
    v = f64(vector).reshape(-1)
    v32 = v.to(torch.float32)
    norms = basis.W_dec.norm(dim=0)
    cos32 = (v32 @ basis.W_dec) / (norms * v32.norm()).clamp_min(1e-30)
    order = torch.topk(cos32.abs(), top).indices.tolist()
    D = basis.directions(order)                                  # (d, top) float64
    cos = (v @ D) / (D.norm(dim=0) * v.norm())
    coef = torch.linalg.lstsq(D, v.unsqueeze(1)).solution.reshape(-1)
    residual = v - D @ coef
    return {"ids": [int(i) for i in order], "cos": cos.tolist(), "coef": coef.tolist(),
            "outside_fraction": float(residual.norm() / v.norm()) if float(v.norm()) else 0.0,
            "norm": float(v.norm())}


def trace_feature(run: Any, basis: FeatureBasis, feature: int, row: int, *,
                  top: int = 4) -> dict[str, Any]:
    """A feature's life along one token row, read with its encoder row as a linear probe
    ``pre_f(x) = W_enc[f].x + b_enc[f]`` (float64; the row sliced, then cast) at every residual
    port ``h``, ``r``, ``h_out`` of every layer, and its writers: the exact linear shares
    ``W_enc[f].o_l`` (attention) and ``W_enc[f].m_l`` (MLP) of each layer, the top heads
    (``W_enc[f].W_o,h A_h``) of the strongest attention writer and the top neurons
    (``W_enc[f].W_down[:, i] f_i``) of the strongest MLP writer. ``cutoff`` is the smallest
    active pre-activation at the SAE's own site (the top-k threshold there): above it at the
    site means active. Off the site the probe is the same linear map on another layer's
    stream (no claim that the SAE is trained there)."""
    from .reads import head_mix, layer_module

    torch = _torch()
    w = basis.W_enc[int(feature)].to(torch.float64)
    b = float(basis.b_enc[int(feature)].to(torch.float64))
    layers = run.adapter.layers
    probe, writers = [], []
    for layer in range(layers):
        ports = {name: f64(run.read(f"t{row}.L{layer}.{name}")).reshape(-1)
                 for name in ("h", "r", "h_out", "o", "m")}
        probe.append({"layer": layer, **{name: float(w @ ports[name]) + b
                                         for name in ("h", "r", "h_out")}})
        writers.append({"layer": layer, "attn": float(w @ ports["o"]),
                        "mlp": float(w @ ports["m"])})
    site_x = run.read(f"t{row}.L{basis.layer}.{basis.port}")
    ids, pre32 = basis.select(site_x)
    cutoff = float(pre32[-1]) if len(ids) else 0.0
    active = int(feature) in ids
    order = ["h", "r", "h_out"]
    above = [(p["layer"], name) for p in probe for name in order if p[name] > cutoff]
    attn_layer = max(writers, key=lambda x: abs(x["attn"]))["layer"]
    mlp_layer = max(writers, key=lambda x: abs(x["mlp"]))["layer"]
    mix = head_mix(run, row, attn_layer)
    head_parts = mix.o_parts().sum(dim=1).to("cpu") @ w                  # (H,)
    heads = sorted(((int(h), float(v)) for h, v in enumerate(head_parts.tolist())),
                   key=lambda x: -abs(x[1]))[:top]
    down = layer_module(run.adapter, mlp_layer).mlp.down_proj.weight
    f_act = f64(run.read(f"t{row}.L{mlp_layer}.f")).reshape(-1)
    unit = (w.to(down.device).to(torch.float32) @ down.to(torch.float32)).to("cpu")   # ranking
    approx = unit.to(torch.float64) * f_act
    cand = torch.topk(approx.abs(), min(4 * top, approx.numel())).indices.tolist()
    exact = [(int(i), float(w @ down[:, i].detach().to("cpu").to(torch.float64)) * float(f_act[i]))
             for i in cand]
    neurons = sorted(exact, key=lambda x: -abs(x[1]))[:top]
    return {"feature": int(feature), "site": basis.site, "cutoff": cutoff, "active": active,
            "enc_bias": b, "probe": probe, "writers": writers,
            "first_above": above[0] if above else None,
            "last_above": above[-1] if above else None,
            "attn_layer": attn_layer, "heads": heads, "mlp_layer": mlp_layer,
            "neurons": neurons}


__all__ = ["Decomposition", "FeatureBasis", "circuit", "match_unit", "trace_feature", "unit_vector", "SaeError", "SaeSet", "edit_vector", "is_expression",
           "logit_lens", "margin_split", "open_set", "parse_terms", "SCHEMA", "SUPPLIED", "LOCAL"]
